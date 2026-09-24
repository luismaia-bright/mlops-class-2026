"""
Score every Prompt Mode against the Evaluation Set and promote the winner.

This is the LLMOps cycle of class 3, and it is deliberately the same shape as the MLOps
cycle of class 2:

    class 2   train models   -> register versions -> evaluate -> alias @champion -> API serves
    class 3   write prompts  -> register versions -> evaluate -> alias @champion -> BonsAI serves

The thing being versioned changed. The discipline did not.

Run it:
    docker compose exec bonsai python -m src.evaluate_prompts
"""

import argparse
import logging
import os
import time

import mlflow
import mlflow.genai
from rouge_score import rouge_scorer

from src import evaluation_set, llm_client
from src.prompt_modes import PROMPT_MODES, PROMPT_NAME, REFUSAL

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Bonsai-Care-Prompt-Engineering"

# A mode must clear this to be promoted. Serving a prompt that answers questions it should
# refuse is worse than serving yesterday's prompt.
MIN_REFUSAL_ACCURACY = 1.0
MIN_OVERALL_SCORE = 0.30


def looks_like_refusal(response: str) -> bool:
    """
    Did BonsAI decline? We accept the exact sentence we asked for, and also a clear
    paraphrase — models rarely repeat a canned line verbatim, and marking a polite,
    correct refusal as a failure would push us to promote the wrong mode.
    """
    text = response.lower()
    if REFUSAL.lower() in text:
        return True
    declines = ("only provide information related to bonsai", "only answer questions about bonsai",
                "only help with bonsai", "specialise", "specialize", "i can only")
    return any(phrase in text for phrase in declines) and "bonsai" in text


class RateLimiter:
    """
    Keep calls under the provider's requests-per-minute ceiling.

    The Gemini free tier allows 5 requests per minute per model. Firing an evaluation as
    fast as the code can loop gets most calls rejected with 429, and — this is the part
    that matters — a rejected call still produces a score of zero. The run then looks like
    a prompt-quality result when it is really a quota result.

    Rate limiting is not politeness. It is what makes the numbers mean anything.
    """

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.last_call = 0.0

    def wait(self):
        if not self.interval:
            return
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


def score_mode(client, model, max_tokens, mode_name, template, limiter, cases=None):
    """
    Run one Prompt Mode over the Evaluation Set and return its metrics.

    `cases` defaults to the whole set. The notebook passes a sample when it is
    demonstrating the mechanics — see evaluation_set.sample().
    """
    cases = cases if cases is not None else evaluation_set.all_cases()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1, rougeL, mentioned, actionable = [], [], [], []
    refusals_correct = 0
    off_topic_total = 0
    truncated = 0
    failures = 0
    latencies = []

    for case in cases:
        prompt = template.replace("{{query}}", case["query"])

        limiter.wait()
        started = time.time()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            choice = completion.choices[0]
            answer = choice.message.content or ""
            if choice.finish_reason == "length":
                truncated += 1
        except Exception as exc:
            # A failed call is a real result, not something to paper over. It counts
            # against the mode rather than being silently dropped.
            logger.warning("    call failed for %r: %s", case["query"][:40], exc)
            failures += 1
            latencies.append(time.time() - started)
            if case["kind"] == "off_topic":
                off_topic_total += 1
            continue
        latencies.append(time.time() - started)

        if case["kind"] == "off_topic":
            off_topic_total += 1
            if looks_like_refusal(answer):
                refusals_correct += 1
            continue

        scores = scorer.score(case["reference"], answer)
        rouge1.append(scores["rouge1"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)

        lowered = answer.lower()
        mentioned.append(
            sum(word in lowered for word in case["must_mention"]) / len(case["must_mention"])
        )
        actionable.append(float(any(word in lowered for word in evaluation_set.ACTION_WORDS)))

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "rouge1": mean(rouge1),
        "rougeL": mean(rougeL),
        "key_point_coverage": mean(mentioned),
        "actionability": mean(actionable),
        "refusal_accuracy": refusals_correct / off_topic_total if off_topic_total else 0.0,
        "avg_latency_seconds": mean(latencies),
        "truncated_responses": truncated,
        "failed_calls": failures,
    }

    # One number to rank by. Weighted towards saying the right things and refusing the
    # wrong ones, because word overlap with a single reference answer is a weak signal for
    # free-form advice — ROUGE is here so you can see that for yourself.
    metrics["overall_score"] = (
        0.40 * metrics["key_point_coverage"]
        + 0.30 * metrics["refusal_accuracy"]
        + 0.20 * metrics["rougeL"]
        + 0.10 * metrics["actionability"]
    )
    metrics["mode"] = mode_name
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate BonsAI Prompt Modes")
    parser.add_argument("--promote", action="store_true",
                        help="move the @champion alias onto the winner")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument("--rpm", type=int, default=int(os.getenv("LLM_REQUESTS_PER_MINUTE", 5)),
                        help="requests per minute; the Gemini free tier allows 5 per model")
    args = parser.parse_args()

    if not llm_client.is_configured():
        raise SystemExit(
            "No GEMINI_API_KEY. Put it in docker/.env next to docker-compose.yml.\n"
            "There is deliberately no offline mode: inventing scores would teach you to "
            "trust a number that measured nothing."
        )

    client = llm_client.build_client()
    model = llm_client.get_model()
    max_tokens = llm_client.get_max_tokens()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Record every scoring call, not just the score it produced.
    #
    # A metric tells you a mode scored 0.31. It cannot tell you whether that was four
    # sensible answers and six refusals, or ten truncated ones. The traces land inside the
    # nested run for each mode, so the evidence sits next to the number it produced — and
    # when a result looks wrong, the first question ("what did we actually send?") has an
    # answer instead of a rerun.
    mlflow.openai.autolog()

    cases = evaluation_set.all_cases()
    limiter = RateLimiter(args.rpm)

    total_calls = len(PROMPT_MODES) * len(cases)
    estimate = total_calls * 60 / args.rpm / 60 if args.rpm else 0
    logger.info("Evaluating %d Prompt Modes over %d cases with %s",
                len(PROMPT_MODES), len(cases), model)
    logger.info("%d calls at %d requests/minute — about %.0f minutes\n",
                total_calls, args.rpm, estimate)

    results = []
    versions = {}

    with mlflow.start_run(run_name="prompt-mode-comparison"):
        mlflow.log_params({
            "requests_per_minute": args.rpm,
            "model": model,
            "max_tokens": max_tokens,
            "evaluation_cases": len(cases),
            "on_topic_cases": len(evaluation_set.ON_TOPIC),
            "off_topic_cases": len(evaluation_set.OFF_TOPIC),
        })

        for mode_name, mode in PROMPT_MODES.items():
            logger.info("  %s", mode_name)

            # Every mode becomes a version of the same registered prompt, so that one
            # @champion alias can point at whichever one wins.
            version = mlflow.genai.register_prompt(
                name=PROMPT_NAME,
                template=mode["template"],
                commit_message=f"{mode_name}: {mode['description']}",
                tags={"mode": mode_name},
            )
            versions[mode_name] = version.version

            with mlflow.start_run(run_name=mode_name, nested=True):
                metrics = score_mode(client, model, max_tokens, mode_name, mode["template"], limiter)
                mlflow.log_param("mode", mode_name)
                mlflow.log_param("prompt_version", version.version)
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

            results.append(metrics)
            logger.info("    score %.3f | key points %.2f | refusals %.2f | %.1fs/case",
                        metrics["overall_score"], metrics["key_point_coverage"],
                        metrics["refusal_accuracy"], metrics["avg_latency_seconds"])

        results.sort(key=lambda m: m["overall_score"], reverse=True)
        winner = results[0]

        logger.info("\nRanking:")
        for rank, metrics in enumerate(results, start=1):
            logger.info("  %d. %-12s %.3f", rank, metrics["mode"], metrics["overall_score"])

        mlflow.log_param("best_mode", winner["mode"])
        mlflow.log_metric("best_overall_score", winner["overall_score"])

        # The gate. A mode that answers questions it was told to refuse does not get
        # promoted no matter how well it scores on everything else.
        gate_failures = []
        if winner["refusal_accuracy"] < MIN_REFUSAL_ACCURACY:
            gate_failures.append(
                f"refusal_accuracy {winner['refusal_accuracy']:.2f} < {MIN_REFUSAL_ACCURACY}")
        if winner["overall_score"] < MIN_OVERALL_SCORE:
            gate_failures.append(
                f"overall_score {winner['overall_score']:.3f} < {MIN_OVERALL_SCORE}")

        mlflow.log_metric("gate_passed", 0 if gate_failures else 1)

        if gate_failures:
            logger.info("\nGate FAILED for %s: %s", winner["mode"], "; ".join(gate_failures))
            logger.info("Not promoting. BonsAI keeps serving the current champion.")
            return

        logger.info("\nGate passed for %s.", winner["mode"])

        if not args.promote:
            logger.info("Run again with --promote to move the @champion alias.")
            return

        mlflow.genai.set_prompt_alias(PROMPT_NAME, "champion", versions[winner["mode"]])
        mlflow.log_param("promoted_version", versions[winner["mode"]])
        logger.info("@champion -> %s version %s", PROMPT_NAME, versions[winner["mode"]])
        logger.info("Tell BonsAI to pick it up:  curl -X POST http://localhost:3000/prompt/reload")


if __name__ == "__main__":
    main()
