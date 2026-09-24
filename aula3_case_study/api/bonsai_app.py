"""
BonsAI Chat Bot - A specialized bonsai care assistant
Interactive web interface for bonsai plant care assistance using MLflow and Gemini
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
import mlflow
import requests
import json
import logging
import os
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Any, List

from src import llm_client, prompt_modes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
EXPERIMENT_NAME = 'Bonsai-Care-Prompt-Engineering'

# Live traffic is traced into its own experiment, separate from the evaluation runs.
#
# The two answer different questions. EXPERIMENT_NAME answers "which Prompt Mode scored
# best on our Evaluation Set" — four runs, made on purpose, before deploying. This one
# answers "what did customers actually ask, and what did we actually answer" — one trace
# per question, forever. Keeping them apart means a week of chat traffic does not bury
# the four runs the class compares.
CHAT_EXPERIMENT_NAME = os.getenv('MLFLOW_CHAT_EXPERIMENT', 'Bonsai-Care-Chat')

# How many previous turns of a conversation get replayed to the model. Each turn is sent
# again on every question, so this is a direct cost and latency knob: 6 turns of context
# on a 20-question conversation is the difference between a cheap demo and an expensive
# one. It is also why a bot "forgets" — the limit is where memory stops.
MAX_HISTORY_TURNS = int(os.getenv('CHAT_HISTORY_TURNS', 6))

# How many conversations to keep in memory before the oldest is dropped.
#
# In-process and lost on restart, which is wrong for production and right for a class:
# a real deployment puts this in Redis, and the point of the lesson is the trace in
# MLflow, not the store. The cap stops a long demo from growing without bound.
MAX_CONVERSATIONS = int(os.getenv('CHAT_MAX_CONVERSATIONS', 200))

# Which LLM answers, and how, is decided in one place: src/llm_client.py
MODEL_NAME = llm_client.get_model()
MAX_TOKENS = llm_client.get_max_tokens()

# None when GEMINI_API_KEY is not set. The app still starts and says so on /health,
# instead of refusing to boot with a stack trace nobody can read.
llm = llm_client.build_client()

if llm is None:
    logger.warning("No GEMINI_API_KEY set — BonsAI will start but cannot answer questions")

# The Prompt Modes live in src/prompt_modes.py, shared with the evaluation pipeline.
# If the service kept its own copy, the pipeline would score one set of prompts while
# customers got another, and the evaluation would prove nothing.
PROMPT_NAME = prompt_modes.PROMPT_NAME
BONSAI_PROMPTS = prompt_modes.PROMPT_MODES

# Global variables
current_prompt_template = BONSAI_PROMPTS["basic"]
model_info = {}
tracing_enabled = False

# Conversations in flight, newest last: {session_id: [{"role": ..., "content": ...}, ...]}
#
# Flask's development server handles requests on threads, so two browser tabs can land
# here at the same time. The lock is not ceremony: without it, two questions arriving
# together can interleave and drop a turn.
_conversations: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
_conversations_lock = threading.Lock()


def setup_tracing() -> bool:
    """
    Turn on MLflow tracing for every LLM call this service makes.

    Two lines do the work. `set_experiment` decides where traces land; `openai.autolog`
    patches the OpenAI SDK so each `chat.completions.create` records itself — the prompt
    that went in, the answer that came out, the token counts, the latency, and the error
    if it failed.

    Note that we talk to Gemini, not OpenAI. The autologger instruments the *SDK*, and
    Gemini is reached through its OpenAI-compatible endpoint (see src/llm_client.py), so
    the same integration covers it. Point LLM_BASE_URL at a third provider and tracing
    keeps working, for the same reason.

    Returns False rather than raising when MLflow is unreachable: a tracking server that
    is down should cost you observability, not the service.
    """
    global tracing_enabled

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(CHAT_EXPERIMENT_NAME)
        mlflow.openai.autolog()
        tracing_enabled = True
        logger.info(f"✅ Tracing into MLflow experiment '{CHAT_EXPERIMENT_NAME}'")
    except Exception as e:
        tracing_enabled = False
        logger.warning(f"⚠️ Tracing is off — could not reach MLflow at {MLFLOW_TRACKING_URI}: {e}")

    return tracing_enabled


def get_history(session_id: str) -> List[Dict[str, str]]:
    """The last MAX_HISTORY_TURNS exchanges of this conversation, oldest first."""
    with _conversations_lock:
        return list(_conversations.get(session_id, []))


def remember_turn(session_id: str, question: str, answer: str) -> None:
    """Append one exchange to a conversation, trimming both it and the store."""
    with _conversations_lock:
        history = _conversations.pop(session_id, [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        # Two messages per turn, so the window is twice the turn count.
        _conversations[session_id] = history[-(MAX_HISTORY_TURNS * 2):]

        while len(_conversations) > MAX_CONVERSATIONS:
            _conversations.popitem(last=False)

# HTML Template for the chat interface
CHAT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BonsAI Chat - Your Bonsai Care Expert</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        
        .chat-header h1 {
            font-size: 1.8em;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            background: #4CAF50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.bot {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .message.user .message-content {
            background: #007bff;
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        
        .message-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            font-size: 0.8em;
        }
        
        .message.user .message-avatar {
            background: #007bff;
            order: 2;
        }
        
        .message.bot .message-avatar {
            background: #4CAF50;
            order: 1;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        .input-group input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 0.9em;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .input-group input:focus {
            border-color: #4CAF50;
        }
        
        .input-group button {
            padding: 12px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s;
        }
        
        .input-group button:hover {
            background: #45a049;
        }
        
        .input-group button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .welcome-message {
            text-align: center;
            color: #666;
            padding: 40px 20px;
        }
        
        .welcome-message h3 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        
        .quick-questions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
            justify-content: center;
        }
        
        .quick-question {
            background: #e8f5e8;
            color: #2e7d2e;
            padding: 8px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            cursor: pointer;
            transition: background 0.3s;
            border: none;
        }
        
        .quick-question:hover {
            background: #4CAF50;
            color: white;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px 16px;
            background: white;
            border-radius: 18px;
            border: 1px solid #e0e0e0;
            width: fit-content;
            margin-bottom: 15px;
        }
        
        .typing-dots {
            display: inline-block;
        }
        
        .typing-dots span {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #4CAF50;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        .error-message {
            color: #dc3545;
            text-align: center;
            padding: 10px;
            background: #f8d7da;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="status-indicator"></div>
            <h1>🌿 BonsAI Chat</h1>
            <p>Your specialized bonsai care expert assistant</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <h3>Welcome to BonsAI!</h3>
                <p>I'm your specialized bonsai care expert. Ask me anything about bonsai care, styling, watering, fertilizing, or any bonsai-related questions!</p>
                <div class="quick-questions">
                    <button class="quick-question" onclick="sendQuickQuestion(this)">How often should I water my Juniper bonsai?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">What soil mix is best for Ficus bonsai?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">My bonsai leaves are yellowing, help!</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">When should I repot my bonsai?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">How to wire bonsai branches?</button>
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            BonsAI is thinking...
        </div>
        
        <div class="chat-input">
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Ask me about your bonsai..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()" id="sendButton">Send</button>
            </div>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        
        function scrollToBottom() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = isUser ? 'You' : '🌿';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = content.replace(/\\n/g, '<br>');
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            
            chatMessages.appendChild(messageDiv);
            scrollToBottom();
        }
        
        function showTyping() {
            typingIndicator.style.display = 'block';
            scrollToBottom();
        }
        
        function hideTyping() {
            typingIndicator.style.display = 'none';
        }
        
        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = message;
            chatMessages.appendChild(errorDiv);
            scrollToBottom();
        }
        
        // One id per browser tab, kept in sessionStorage so a reload continues the same
        // conversation and a new tab starts a new one. Every message carries it, and the
        // server hands it to MLflow as the trace's session id — which is how a dozen
        // separate traces become one readable conversation in the UI.
        const sessionId = (() => {
            let id = sessionStorage.getItem('bonsai_session_id');
            if (!id) {
                id = 'session-' + Math.random().toString(16).slice(2, 14);
                sessionStorage.setItem('bonsai_session_id', id);
            }
            return id;
        })();

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Clear welcome message if it exists
            const welcomeMessage = document.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
            
            // Add user message
            addMessage(message, true);
            messageInput.value = '';
            sendButton.disabled = true;
            showTyping();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: message, session_id: sessionId })
                });
                
                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }
                
                const data = await response.json();
                hideTyping();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    addMessage(data.response);
                }
                
            } catch (error) {
                hideTyping();
                showError('Sorry, I encountered an error. Please try again.');
                console.error('Error:', error);
            } finally {
                sendButton.disabled = false;
                messageInput.focus();
            }
        }
        
        function sendQuickQuestion(button) {
            messageInput.value = button.textContent;
            sendMessage();
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Focus on input when page loads
        messageInput.focus();
    </script>
</body>
</html>
"""

def use_mode(mode_name: str) -> bool:
    """Serve one of the locally defined Prompt Modes."""
    global current_prompt_template, model_info

    if mode_name not in BONSAI_PROMPTS:
        logger.warning(f"Unknown prompt mode: {mode_name}")
        return False

    mode = BONSAI_PROMPTS[mode_name]
    current_prompt_template = {
        "name": mode_name,
        "template": mode["template"],
        "description": mode["description"],
    }
    model_info = {
        "prompt_name": PROMPT_NAME,
        "mode": mode_name,
        "version": None,
        "source": "local",
        "status": "local_fallback",
        "description": mode["description"],
        "loaded_at": datetime.now().isoformat(),
    }
    logger.info(f"Serving local Prompt Mode: {mode_name}")
    return True


def load_champion_prompt() -> bool:
    """
    Serve whichever Prompt Mode currently carries the @champion alias.

    This is the whole point of the registry: the evaluation pipeline decides which mode
    wins and moves the alias, and BonsAI picks it up without being redeployed. Exactly
    what class 2 does with the model registry, applied to prompts.

    Falls back to the local 'basic' mode when no champion exists yet — a fresh stack has
    not run the evaluation, and an assistant that refuses to answer at all would be worse
    than one answering with a reasonable default.
    """
    global current_prompt_template, model_info

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@champion")

        mode_name = (prompt.tags or {}).get("mode", "unknown")
        current_prompt_template = {
            "name": mode_name,
            "template": prompt.template,
            "description": f"champion, version {prompt.version}",
        }
        model_info = {
            "prompt_name": PROMPT_NAME,
            "mode": mode_name,
            "version": prompt.version,
            "source": "mlflow",
            "status": "champion",
            "description": f"prompts:/{PROMPT_NAME}@champion",
            "loaded_at": datetime.now().isoformat(),
        }
        logger.info(f"Serving @champion: {PROMPT_NAME} v{prompt.version} ({mode_name})")
        return True

    except Exception as e:
        logger.warning(f"No @champion in the registry yet ({e}); falling back to local basic")
        return use_mode("basic")


@mlflow.trace(span_type="LLM")
def query_llm(prompt: str, history: List[Dict[str, str]] | None = None) -> str:
    """
    Send a prompt to the configured LLM and return BonsAI's reply.

    The decorator opens a span around this function, and the autologger opens another one
    inside it for the HTTP call itself. That nesting is the point: the outer span shows
    what BonsAI decided to send, the inner one shows what the provider did with it. When
    an answer is wrong, those are two different suspects.
    """
    if llm is None:
        logger.error("LLM client not initialized — GEMINI_API_KEY is not set")
        return "Sorry, I'm not configured right now: no API key was provided."

    # Earlier turns first, then the templated question. The model sees the conversation.
    messages = list(history or [])
    messages.append({"role": "user", "content": prompt})

    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            # Generous on purpose: Gemini spends part of this budget reasoning before it
            # writes anything, and an exhausted budget returns a half-finished sentence
            # rather than an error.
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )

        if not response.choices:
            logger.error("LLM returned no choices")
            return "Sorry, I couldn't generate a response right now."

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning(
                "Reply hit the token limit (max_tokens=%s) and was cut off", MAX_TOKENS
            )

        return choice.message.content or "Sorry, I couldn't generate a response right now."

    except Exception as e:
        logger.error(f"Error querying the LLM: {str(e)}")
        return "Sorry, I encountered an error while processing your bonsai question. Please try again."

@app.route('/', methods=['GET'])
def chat_interface():
    """Serve the main chat interface"""
    return render_template_string(CHAT_HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "bonsai-chat-bot",
        "bot_name": "BonsAI",
        "model_info": model_info,
        # Reports how the LLM is configured, and never the key itself.
        "llm": llm_client.describe(),
        # If traces are missing, this is the first thing to look at.
        "tracing": {
            "enabled": tracing_enabled,
            "experiment": CHAT_EXPERIMENT_NAME,
            "tracking_uri": MLFLOW_TRACKING_URI,
            "active_sessions": len(_conversations),
            "history_turns": MAX_HISTORY_TURNS,
        },
    })

@mlflow.trace(name="bonsai_chat_turn", span_type="CHAIN")
def answer_question(user_query: str, session_id: str, turn: int) -> str:
    """
    One turn of a conversation: template the question, ask the model, remember the answer.

    This is the traced unit. The decorator makes it the root span, and every span opened
    underneath — query_llm, and the provider call the autologger captures inside it —
    hangs off this one. Open a trace in MLflow and you see the whole turn in one tree.

    `update_current_trace` is what turns a pile of traces into conversations. The session
    id is what the MLflow UI groups by, so tagging it here is the difference between 40
    unrelated questions and 8 conversations you can read end to end. The prompt version
    rides along as a tag, which is what lets you ask the question that matters after a
    deployment: did the answers get worse when we moved the champion alias?
    """
    history = get_history(session_id)

    mlflow.update_current_trace(
        session_id=session_id,
        tags={
            "prompt_mode": current_prompt_template['name'],
            "prompt_source": model_info.get("source", "unknown"),
            "prompt_version": str(model_info.get("version", "local")),
            "turn": str(turn),
        },
    )

    # Apply BonsAI prompt template
    # Templates use the registry's {{query}} syntax, so a plain replace — not
    # str.format, which would choke on the JSON braces inside some prompts.
    formatted_prompt = current_prompt_template['template'].replace('{{query}}', user_query)

    ai_response = query_llm(formatted_prompt, history=history)
    remember_turn(session_id, user_query, ai_response)

    return ai_response


@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint for BonsAI assistance"""
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request"
            }), 400

        user_query = data['query']

        # Validate input
        if not user_query.strip():
            return jsonify({
                "error": "Query cannot be empty"
            }), 400

        # The browser sends the same id for every message in a tab (see sendMessage in the
        # template). A caller that sends none — curl, or the test suite — gets a fresh
        # one, so every turn is still attached to a session, just a session of one.
        session_id = (data.get('session_id') or '').strip() or f"session-{uuid.uuid4().hex[:12]}"
        turn = len(get_history(session_id)) // 2 + 1

        # Query the LLM
        logger.info(f"🌿 BonsAI processing query: {user_query[:50]}... [session {session_id} turn {turn}]")
        ai_response = answer_question(user_query, session_id, turn)

        # Prepare response
        response_data = {
            "query": user_query,
            "response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "bot_name": "BonsAI",
            "model": MODEL_NAME,
            "prompt_template": current_prompt_template['name'],
            "session_id": session_id,
            "turn": turn,
        }

        logger.info(f"✅ BonsAI response generated successfully")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Error in BonsAI chat endpoint: {str(e)}")
        return jsonify({
            "error": "I'm having trouble right now. Please try again.",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/prompt/switch', methods=['POST'])
def switch_prompt():
    """Switch to a different prompt template"""
    try:
        data = request.get_json()
        prompt_type = data.get('prompt_type', 'basic')
        
        if use_mode(prompt_type):
            return jsonify({
                "status": "success",
                "message": f"Switched to {prompt_type} prompt",
                "current_prompt": current_prompt_template,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": "Failed to switch prompt template"
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Error switching prompt: {str(e)}")
        return jsonify({
            "error": "Internal server error"
        }), 500

@app.route('/prompt/reload', methods=['POST'])
def reload_prompt_from_mlflow():
    """
    Re-read prompts:/bonsai-care@champion from the registry.

    Call this after the evaluation pipeline promotes a new Prompt Mode, and BonsAI starts
    answering with it. No redeploy, no restart.
    """
    try:
        if load_champion_prompt():
            return jsonify({
                "status": "success",
                "message": f"Reloaded @champion from MLflow",
                "current_prompt": current_prompt_template,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": "Failed to reload prompt from MLflow"
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Error reloading prompt: {str(e)}")
        return jsonify({
            "error": "Internal server error"
        }), 500

@app.route('/prompt/info', methods=['GET'])
def prompt_info():
    """Get information about the current prompt template"""
    return jsonify({
        "current_prompt": current_prompt_template,
        "available_prompts": list(BONSAI_PROMPTS.keys()),
        "model_info": model_info,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/evaluate', methods=['POST'])
def evaluate_response():
    """Evaluate a BonsAI response for feedback"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data or 'response' not in data or 'rating' not in data:
            return jsonify({
                "error": "Missing required fields: 'query', 'response', 'rating'"
            }), 400
        
        # Log evaluation data
        evaluation_data = {
            "query": data['query'],
            "response": data['response'], 
            "rating": data['rating'],
            "feedback": data.get('feedback', ''),
            "timestamp": datetime.now().isoformat(),
            "bot_name": "BonsAI",
            "prompt_template": current_prompt_template['name']
        }
        
        logger.info(f"📊 BonsAI evaluation received: Rating {data['rating']}/5")
        
        return jsonify({
            "status": "evaluation_recorded",
            "message": "Thank you for your feedback! This helps BonsAI learn.",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error in evaluate endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error"
        }), 500

@app.route('/bonsai/info', methods=['GET'])
def bonsai_info():
    """Get BonsAI bot information and capabilities"""
    return jsonify({
        "bot_name": "BonsAI",
        "description": "Specialized bonsai care expert assistant",
        "capabilities": [
            "Bonsai care advice",
            "Species-specific guidance",
            "Watering schedules",
            "Soil recommendations",
            "Styling techniques",
            "Problem diagnosis",
            "Emergency care"
        ],
        "specialization": "Bonsai plants only",
        "available_prompt_modes": {
            "basic": "Simple conversational bonsai advice",
            "structured": "Structured problem-solution format",
            "diagnostic": "Systematic bonsai problem analysis",
            "emergency": "Urgent bonsai care situations"
        },
        "current_mode": current_prompt_template['name'],
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/examples', methods=['GET'])
def example_queries():
    """Get example bonsai questions for testing"""
    examples = [
        {
            "query": "How often should I water my Juniper bonsai?",
            "category": "care_routine",
            "difficulty": "beginner"
        },
        {
            "query": "What is the best soil mix for a Ficus bonsai?",
            "category": "soil_care",
            "difficulty": "beginner"
        },
        {
            "query": "My bonsai's leaves are turning yellow and falling off. What should I do?",
            "category": "problem_diagnosis",
            "difficulty": "intermediate"
        },
        {
            "query": "What does the word 'bonsai' mean?",
            "category": "general_knowledge",
            "difficulty": "beginner"
        },
        {
            "query": "Can I keep my bonsai tree indoors?",
            "category": "care_environment",
            "difficulty": "beginner"
        },
        {
            "query": "What is nebari in bonsai?",
            "category": "techniques",
            "difficulty": "intermediate"
        },
        {
            "query": "How do I wire bonsai branches safely?",
            "category": "styling",
            "difficulty": "advanced"
        },
        {
            "query": "When should I repot my pine bonsai?",
            "category": "care_routine",
            "difficulty": "intermediate"
        }
    ]
    
    return jsonify({
        "example_queries": examples,
        "total_examples": len(examples),
        "instructions": "Send POST request to /chat with {'query': 'your bonsai question'}",
        "note": "BonsAI only answers questions about bonsai plants"
    })

# Initialize the application
def initialize_app():
    """Initialize the BonsAI Flask application"""
    logger.info("🌿 Initializing BonsAI Chat Bot")

    # Before anything else answers a question, so no turn goes untraced.
    setup_tracing()

    # Serve whatever currently holds the @champion alias
    if not load_champion_prompt():
        logger.warning("⚠️ Using fallback configuration")
    
    # This service does not register prompts. Putting Prompt Modes into the registry is
    # the evaluation pipeline's job (src/evaluate_prompts.py), because a mode should only
    # exist in the registry once it has been scored. A service that registered its own
    # prompts would be back to evaluating one thing and serving another.

    # Validate the LLM configuration
    if llm is None:
        logger.warning("⚠️ No GEMINI_API_KEY set — BonsAI cannot answer questions")
        logger.warning("   Put it in aula3_case_study/docker/.env, next to docker-compose.yml")
    else:
        logger.info(f"✅ LLM client ready: {MODEL_NAME} (max_tokens={MAX_TOKENS})")
    
    logger.info("✅ BonsAI Chat Bot initialized successfully")
    logger.info("🌐 Access the chat interface at: http://localhost:3000")

if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=3000, debug=True)
