# =============================================================
# BTR AI Middleware - app.py (Version 2.3 - Bodo Fix)
# =============================================================
# Handles: Language detection, Translation, Web Search,
#          Knowledge Base, AI Answering via Groq
# New: Bodo language support, reply-in-language detection
# =============================================================

import os
import time
import hashlib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from langdetect import detect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# API Keys
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY    = os.getenv("SERPER_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")

# Cache setup
cache = {}
CACHE_EXPIRY = 3600

# =============================================================
# Language map for IndicTrans2
# Maps detected language codes to IndicTrans2 codes
# =============================================================
LANGUAGE_MAP = {
    "hi" : "hin_Deva",   # Hindi
    "bn" : "ben_Beng",   # Bengali
    "as" : "asm_Beng",   # Assamese
    "ne" : "npi_Deva",   # Nepali
    "en" : "eng_Latn",   # English
    "brx": "brx_Deva",   # Bodo
}

# =============================================================
# Keywords that indicate user wants response in a specific language
# We check if user's message contains any of these phrases
# =============================================================
LANGUAGE_REQUEST_KEYWORDS = {
    # Bodo language requests
    "brx_Deva": [
        "in bodo", "bodo language", "bodo te", "bodo torsino",
        "boro language", "in boro", "बर' भाषा", "बर'",
        "reply in bodo", "answer in bodo", "respond in bodo",
        "translate to bodo", "bodo mwn"
    ],
    # Assamese language requests
    "asm_Beng": [
        "in assamese", "assamese language", "asomiya",
        "অসমীয়া", "reply in assamese", "answer in assamese",
        "translate to assamese"
    ],
    # Hindi language requests
    "hin_Deva": [
        "in hindi", "hindi language", "hindi mein", "हिंदी में",
        "reply in hindi", "answer in hindi", "translate to hindi"
    ],
    # Bengali language requests
    "ben_Beng": [
        "in bengali", "bengali language", "bangla",
        "বাংলায়", "reply in bengali", "answer in bengali"
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
                            "source": filename,
                            "content": chunk
                        })
            print(f"[Knowledge base]: Loaded {filename}")

    print(f"[Knowledge base]: Total chunks: {len(knowledge_base)}")


# Load on startup
load_knowledge_base()


# =============================================================
# FUNCTION: Check if user requested a specific reply language
# Returns IndicTrans2 language code if found, None otherwise
# Example: "What is BTR? Reply in Bodo" → "brx_Deva"
# =============================================================
def detect_requested_language(message):
    message_lower = message.lower()

    for lang_code, keywords in LANGUAGE_REQUEST_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in message_lower:
                print(f"[Requested language detected]: {lang_code} via keyword '{keyword}'")
                return lang_code

    return None  # No specific language requested


# =============================================================
# FUNCTION: Detect language of incoming message
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
# FUNCTION: Translate using HuggingFace IndicTrans2
# Best model for Indian languages including Bodo
# =============================================================
def translate(text, src_lang, tgt_lang):
    # No translation needed if same language
    if src_lang == tgt_lang:
        return text

    try:
        # Choose model direction
        # indic-en = any Indian language to English
        # en-indic = English to any Indian language
        if tgt_lang == "eng_Latn":
            model = "ai4bharat/indictrans2-indic-en-1B"
        else:
            model = "ai4bharat/indictrans2-en-indic-1B"

        print(f"[Translating]: {src_lang} → {tgt_lang} using {model}")

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={
                "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": text,
                "parameters": {
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang
                }
            },
            timeout=30
        )

        result = response.json()
        print(f"[Translation raw response]: {str(result)[:200]}")

        # Handle model loading
        if isinstance(result, dict) and "error" in result:
            if "loading" in str(result.get("error", "")).lower():
                print("[Translation model loading, waiting 20s]")
                time.sleep(20)
                # Retry once
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers={
                        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": text,
                        "parameters": {
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang
                        }
                    },
                    timeout=30
                )
                result = response.json()

        if isinstance(result, list) and len(result) > 0:
            translated = result[0].get("translation_text", text)
            print(f"[Translation done]: {translated[:100]}")
            return translated
        else:
            print(f"[Translation failed, returning original]: {result}")
            return text

    except Exception as e:
        print(f"[Translation error]: {e}")
        return text  # Return original if translation fails


# =============================================================
# FUNCTION: Search knowledge base for relevant content
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
# FUNCTION: Web search via Tavily (primary)
# =============================================================
def search_tavily(query):
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query + " Bodoland BTR Assam India",
                "max_results": 5,
                "search_depth": "advanced",
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

        print(f"[Tavily]: {len(data.get('results', []))} results")
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
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "q": query + " Bodoland BTR Assam",
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

        print(f"[Serper]: {len(data.get('organic', []))} results")
        return results_text

    except Exception as e:
        print(f"[Serper error]: {e}")
        return ""


# =============================================================
# FUNCTION: Get AI answer from Groq
# Uses Llama 3.1 - fast, free and reliable
# =============================================================
def get_ai_answer(question, knowledge_context, web_context):
    try:
        if not GROQ_API_KEY:
            print("[ERROR]: GROQ_API_KEY not found")
            return "AI service configuration error. Please contact the administrator."

        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant for Bodoland Territorial Region (BTR), Assam, India.
You help people with information about BTR culture, government, tourism, history and current events.
Answer clearly, accurately and respectfully.
Keep answers concise but complete.
If you don't know something, say so honestly rather than making up information.
Always be culturally sensitive and respectful to the Bodo people and their traditions.
IMPORTANT: Always respond in English only. Translation is handled separately."""
            },
            {
                "role": "user",
                "content": f"""Please answer this question about BTR using the information below.

KNOWLEDGE BASE INFORMATION:
{knowledge_context if knowledge_context else "No specific knowledge base information found."}

WEB SEARCH RESULTS:
{web_context if web_context else "No web search results available."}

QUESTION: {question}

Please provide a helpful and accurate answer in English."""
            }
        ]

        print(f"[Groq]: Sending request...")

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7
            },
            timeout=30
        )

        print(f"[Groq status]: {response.status_code}")
        result = response.json()
        print(f"[Groq raw response]: {str(result)[:300]}")

        if "error" in result:
            print(f"[Groq error]: {result['error']}")
            return "I apologize, the AI service is temporarily unavailable. Please try again."

        answer = result["choices"][0]["message"]["content"].strip()
        print(f"[Groq answer]: {answer[:150]}")
        return answer

    except Exception as e:
        print(f"[Groq exception]: {e}")
        return "I apologize, I could not generate an answer right now. Please try again."


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

    print(f"\n[New message]: {user_message[:100]}")

    # Step 1: Check cache
    cache_key = hashlib.md5(user_message.lower().encode()).hexdigest()
    if cache_key in cache:
        cached = cache[cache_key]
        if time.time() - cached["time"] < CACHE_EXPIRY:
            print(f"[Cache hit]")
            return jsonify({
                "answer"  : cached["answer"],
                "language": cached["language"],
                "cached"  : True
            })

    # Step 2: Detect language of input message
    detected_lang = detect_language(user_message)
    src_lang_code = LANGUAGE_MAP.get(detected_lang, "eng_Latn")

    # Step 3: Check if user requested reply in specific language
    # This handles cases like "What is BTR? Reply in Bodo"
    requested_reply_lang = detect_requested_language(user_message)
    print(f"[Requested reply language]: {requested_reply_lang}")

    # Step 4: Translate input to English if needed
    if detected_lang != "en" and src_lang_code != "eng_Latn":
        english_message = translate(user_message, src_lang_code, "eng_Latn")
    else:
        english_message = user_message

    # Step 5: Search knowledge base
    knowledge_context = search_knowledge_base(english_message)
    print(f"[Knowledge]: {len(knowledge_context)} chars found")

    # Step 6: Search web
    web_context = search_tavily(english_message)
    if not web_context:
        print("[Falling back to Serper]")
        web_context = search_serper(english_message)

    # Step 7: Get AI answer (always in English)
    english_answer = get_ai_answer(english_message, knowledge_context, web_context)

    # Step 8: Determine reply language
    # Priority: 1) Explicitly requested language
    #           2) User's input language
    #           3) Default English
    if requested_reply_lang:
        # User asked for specific language (e.g. "reply in Bodo")
        reply_lang_code = requested_reply_lang
        print(f"[Replying in requested language]: {reply_lang_code}")
    elif detected_lang != "en" and src_lang_code != "eng_Latn":
        # User wrote in non-English, reply in same language
        reply_lang_code = src_lang_code
        print(f"[Replying in detected language]: {reply_lang_code}")
    else:
        # Default to English
        reply_lang_code = None

    # Step 9: Translate answer to reply language if needed
    if reply_lang_code and reply_lang_code != "eng_Latn":
        final_answer = translate(english_answer, "eng_Latn", reply_lang_code)
    else:
        final_answer = english_answer

    # Step 10: Cache and return
    cache[cache_key] = {
        "answer"  : final_answer,
        "language": detected_lang,
        "time"    : time.time()
    }

    return jsonify({
        "answer"  : final_answer,
        "language": detected_lang,
        "cached"  : False
    })


# =============================================================
# HEALTH CHECK: /health
# =============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"          : "running",
        "service"         : "BTR AI Middleware",
        "version"         : "2.3",
        "knowledge_chunks": len(knowledge_base)
    })


# =============================================================
# Start app
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
