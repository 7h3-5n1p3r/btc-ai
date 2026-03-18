# =============================================================
# BTR AI Middleware - app.py (Version 3.0)
# =============================================================
# Uses Groq for BOTH answering AND translation
# Supports: English, Hindi, Assamese, Bengali, Bodo and more
# User can ask in any language and request reply in any language
# =============================================================

import os
import time
import hashlib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv

# Make language detection consistent across requests
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# API Keys
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Cache - saves API calls for repeated questions
cache = {}
CACHE_EXPIRY = 3600  # 1 hour

# =============================================================
# Full language name map
# Used to tell Groq which language to reply in
# Key = langdetect code, Value = full language name for Groq
# =============================================================
LANGUAGE_NAMES = {
    "en" : "English",
    "hi" : "Hindi",
    "bn" : "Bengali",
    "as" : "Assamese",
    "ne" : "Nepali",
    "ur" : "Urdu",
    "te" : "Telugu",
    "ta" : "Tamil",
    "ml" : "Malayalam",
    "kn" : "Kannada",
    "gu" : "Gujarati",
    "pa" : "Punjabi",
    "mr" : "Marathi",
    "or" : "Odia",
}

# =============================================================
# Language request keywords
# Detects when user asks for reply in a specific language
# Even if they typed their question in a different language
# =============================================================
LANGUAGE_KEYWORDS = {
    "Bodo"      : [
        "in bodo", "bodo language", "bodo te", "in boro",
        "boro language", "reply in bodo", "answer in bodo",
        "respond in bodo", "translate to bodo", "bodo mwn",
        "बर' भाषा", "बर' ते", "bodo torsino"
    ],
    "Assamese"  : [
        "in assamese", "assamese language", "asomiya",
        "অসমীয়া ত", "reply in assamese", "answer in assamese",
        "assamese te", "asamiya te", "translate to assamese"
    ],
    "Hindi"     : [
        "in hindi", "hindi mein", "hindi me", "हिंदी में",
        "reply in hindi", "answer in hindi", "hindi language",
        "translate to hindi"
    ],
    "Bengali"   : [
        "in bengali", "bengali language", "bangla te",
        "বাংলায়", "reply in bengali", "answer in bengali",
        "translate to bengali"
    ],
    "English"   : [
        "in english", "reply in english", "answer in english",
        "english mein", "english te", "translate to english"
    ],
    "Nepali"    : [
        "in nepali", "nepali ma", "reply in nepali",
        "नेपालीमा", "answer in nepali"
    ],
}

# Knowledge base storage
knowledge_base = []


# =============================================================
# Load knowledge base from /knowledge folder
# =============================================================
def load_knowledge_base():
    global knowledge_base
    knowledge_folder = os.path.join(os.path.dirname(__file__), "knowledge")

    if not os.path.exists(knowledge_folder):
        os.makedirs(knowledge_folder)
        print("[Knowledge base]: Created empty knowledge folder")
        return

    for filename in os.listdir(knowledge_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(knowledge_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                chunk_size = 500
                overlap = 50
                for i in range(0, len(content), chunk_size - overlap):
                    chunk = content[i:i + chunk_size]
                    if chunk.strip():
                        knowledge_base.append({
                            "source"  : filename,
                            "content" : chunk
                        })
            print(f"[Knowledge base]: Loaded {filename}")

    print(f"[Knowledge base]: Total chunks: {len(knowledge_base)}")


# Load on startup
load_knowledge_base()


# =============================================================
# FUNCTION: Search knowledge base
# =============================================================
def search_knowledge_base(query, top_k=3):
    if not knowledge_base:
        return ""

    query_words = query.lower().split()
    scored_chunks = []

    for chunk in knowledge_base:
        content_lower = chunk["content"].lower()
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored_chunks[:top_k]

    if not top_chunks:
        return ""

    result = ""
    for score, chunk in top_chunks:
        result += f"From {chunk['source']}:\n{chunk['content']}\n\n"

    return result


# =============================================================
# FUNCTION: Detect language of message
# =============================================================
def detect_language(text):
    try:
        lang = detect(text)
        print(f"[Language detected]: {lang}")
        return lang
    except Exception as e:
        print(f"[Language detection failed]: {e}")
        return "en"


# =============================================================
# FUNCTION: Detect if user requested a specific reply language
# Returns language name like "Bodo", "Hindi" etc. or None
# Examples:
#   "What is BTR? Reply in Bodo"  → "Bodo"
#   "BTR ki hain? answer in hindi" → "Hindi"
#   "বোডোলেণ্ড কি? in english"     → "English"
# =============================================================
def detect_requested_reply_language(message):
    message_lower = message.lower()

    for language_name, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in message_lower:
                print(f"[Reply language requested]: {language_name}")
                return language_name

    return None


# =============================================================
# FUNCTION: Web search via Tavily (primary)
# =============================================================
def search_tavily(query):
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key"       : TAVILY_API_KEY,
                "query"         : query + " Bodoland BTR Assam India",
                "max_results"   : 5,
                "search_depth"  : "advanced",
                "include_answer": True
            },
            timeout=15
        )

        data = response.json()
        results_text = ""

        if data.get("answer"):
            results_text += f"Summary: {data['answer']}\n\n"

        for r in data.get("results", []):
            results_text += f"Source: {r.get('url', '')}\n"
            results_text += f"Content: {r.get('content', '')}\n\n"

        print(f"[Tavily]: {len(data.get('results', []))} results found")
        return results_text

    except Exception as e:
        print(f"[Tavily error]: {e}")
        return None


# =============================================================
# FUNCTION: Web search via Serper (backup)
# =============================================================
def search_serper(query):
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY"    : SERPER_API_KEY,
                "Content-Type" : "application/json"
            },
            json={
                "q"  : query + " Bodoland BTR Assam",
                "num": 5
            },
            timeout=15
        )

        data = response.json()
        results_text = ""

        for r in data.get("organic", []):
            results_text += f"Source: {r.get('link', '')}\n"
            results_text += f"Title: {r.get('title', '')}\n"
            results_text += f"Content: {r.get('snippet', '')}\n\n"

        print(f"[Serper]: {len(data.get('organic', []))} results found")
        return results_text

    except Exception as e:
        print(f"[Serper error]: {e}")
        return ""


# =============================================================
# FUNCTION: Call Groq API
# Central function used for both answering and translating
# =============================================================
def call_groq(messages, max_tokens=600):
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type" : "application/json"
            },
            json={
                "model"      : "llama-3.1-8b-instant",
                "messages"   : messages,
                "max_tokens" : max_tokens,
                "temperature": 0.7
            },
            timeout=30
        )

        result = response.json()

        if "error" in result:
            print(f"[Groq error]: {result['error']}")
            return None

        answer = result["choices"][0]["message"]["content"].strip()
        return answer

    except Exception as e:
        print(f"[Groq exception]: {e}")
        return None


# =============================================================
# FUNCTION: Get AI answer from Groq in specified language
# This handles BOTH answering AND language output in one step
# =============================================================
def get_ai_answer(question, knowledge_context, web_context, reply_language):
    print(f"[AI]: Getting answer in {reply_language}")

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful AI assistant for Bodoland Territorial Region (BTR), Assam, India.
You help people with information about BTR culture, government, tourism, history and current events.
You are fluent in English, Hindi, Assamese, Bengali, Bodo (Boro) and other Indian languages.
Answer clearly, accurately and respectfully.
Keep answers concise but complete — 3 to 5 sentences is ideal.
If you don't know something, say so honestly.
Always be culturally sensitive and respectful to the Bodo people.

IMPORTANT LANGUAGE INSTRUCTION:
You MUST reply in {reply_language} language only.
Do not reply in English unless {reply_language} is English.
Write your entire response in {reply_language} script and language.
If replying in Bodo, use Devanagari script as used by Bodo people."""
        },
        {
            "role": "user",
            "content": f"""Please answer the following question using the information provided below.
Reply ONLY in {reply_language} language.

KNOWLEDGE BASE INFORMATION:
{knowledge_context if knowledge_context else "No specific knowledge base information found."}

WEB SEARCH RESULTS:
{web_context if web_context else "No web search results available."}

QUESTION: {question}

Remember: Your answer must be written entirely in {reply_language}."""
        }
    ]

    answer = call_groq(messages)

    if answer:
        print(f"[AI answer in {reply_language}]: {answer[:150]}")
        return answer
    else:
        return "I apologize, the AI service is temporarily unavailable. Please try again."


# =============================================================
# MAIN ROUTE: /chat
# =============================================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or not data.get("message"):
        return jsonify({"error": "No message provided"}), 400

    user_message = data.get("message", "").strip()
    session_id   = data.get("session_id", "default")

    print(f"\n{'='*50}")
    print(f"[New message]: {user_message[:100]}")

    # Step 1: Check cache
    cache_key = hashlib.md5(user_message.lower().encode()).hexdigest()
    if cache_key in cache:
        cached = cache[cache_key]
        if time.time() - cached["time"] < CACHE_EXPIRY:
            print(f"[Cache hit]: Returning cached answer")
            return jsonify({
                "answer"  : cached["answer"],
                "language": cached["language"],
                "cached"  : True
            })

    # Step 2: Detect input language
    detected_lang = detect_language(user_message)
    input_language_name = LANGUAGE_NAMES.get(detected_lang, "English")
    print(f"[Input language]: {input_language_name} ({detected_lang})")

    # Step 3: Check if user requested specific reply language
    # e.g. "What is BTR? Reply in Bodo"
    requested_language = detect_requested_reply_language(user_message)

    # Step 4: Determine reply language
    # Priority order:
    # 1. Explicitly requested language ("reply in bodo")
    # 2. User's detected input language (they typed in Hindi → reply Hindi)
    # 3. Default English
    if requested_language:
        reply_language = requested_language
        print(f"[Reply language]: {reply_language} (explicitly requested)")
    elif detected_lang != "en" and input_language_name != "English":
        reply_language = input_language_name
        print(f"[Reply language]: {reply_language} (matched input language)")
    else:
        reply_language = "English"
        print(f"[Reply language]: English (default)")

    # Step 5: Search knowledge base
    # Always search in English for best results
    # If message is not English, use it as-is (Groq understands it)
    knowledge_context = search_knowledge_base(user_message)
    print(f"[Knowledge]: {len(knowledge_context)} chars found")

    # Step 6: Search web
    web_context = search_tavily(user_message)
    if not web_context:
        print("[Falling back to Serper]")
        web_context = search_serper(user_message)

    # Step 7: Get AI answer in the correct language
    # Groq handles both answering AND language in one step
    final_answer = get_ai_answer(
        question          = user_message,
        knowledge_context = knowledge_context,
        web_context       = web_context,
        reply_language    = reply_language
    )

    # Step 8: Cache and return
    cache[cache_key] = {
        "answer"  : final_answer,
        "language": detected_lang,
        "time"    : time.time()
    }

    print(f"[Done]: Replied in {reply_language}")

    return jsonify({
        "answer"         : final_answer,
        "language"       : detected_lang,
        "reply_language" : reply_language,
        "cached"         : False
    })


# =============================================================
# HEALTH CHECK: /health
# =============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"          : "running",
        "service"         : "BTR AI Middleware",
        "version"         : "3.0",
        "knowledge_chunks": len(knowledge_base)
    })


# =============================================================
# Start app
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
