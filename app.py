# =============================================================
# BTR AI Middleware - app.py (Version 2.1 - Debug Mode)
# =============================================================
# Handles: Language detection, Translation, Web Search,
#          Knowledge Base, AI Answering
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

# Cache setup
cache = {}
CACHE_EXPIRY = 3600

# Language map for IndicTrans2
LANGUAGE_MAP = {
    "hi" : "hin_Deva",
    "bn" : "ben_Beng",
    "as" : "asm_Beng",
    "ne" : "npi_Deva",
    "en" : "eng_Latn",
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
# Search knowledge base for relevant content
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
# Detect language of user message
# =============================================================
def detect_language(text):
    try:
        lang = detect(text)
        print(f"[Language]: {lang}")
        return lang
    except Exception as e:
        print(f"[Language detection failed]: {e}")
        return "en"


# =============================================================
# Translate text using HuggingFace IndicTrans2
# =============================================================
def translate(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text

    try:
        if tgt_lang == "eng_Latn":
            model = "ai4bharat/indictrans2-indic-en-1B"
        else:
            model = "ai4bharat/indictrans2-en-indic-1B"

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
            print(f"[Translation done]: {translated[:50]}")
            return translated
        else:
            print(f"[Translation failed]: {result}")
            return text

    except Exception as e:
        print(f"[Translation error]: {e}")
        return text


# =============================================================
# Web search via Tavily (primary)
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
# Web search via Serper (backup)
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
# Get AI answer from HuggingFace Zephyr
# Retries up to 3 times if model is loading
# =============================================================
def get_ai_answer(question, knowledge_context, web_context):
    try:
        # Build prompt
        prompt = f"""<|system|>
You are a helpful AI assistant for Bodoland Territorial Region (BTR), Assam, India.
Answer clearly, accurately and respectfully based on the information provided.
If you don't know something, say so honestly.
</s>
<|user|>
KNOWLEDGE BASE:
{knowledge_context if knowledge_context else "No specific knowledge base information found."}

WEB SEARCH RESULTS:
{web_context if web_context else "No web search results available."}

QUESTION: {question}
</s>
<|assistant|>"""

        # Try up to 3 times if model is loading
        for attempt in range(3):

            print(f"[AI attempt {attempt + 1}/3]")

            response = requests.post(
                "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
                headers={
                    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.7,
                        "return_full_text": False
                    }
                },
                timeout=60
            )

            # Log full raw response for debugging
            print(f"[HF status code]: {response.status_code}")
            result = response.json()
            print(f"[HF raw response]: {str(result)[:300]}")

            # Model is still loading - wait and retry
            if isinstance(result, dict) and "error" in result:
                error_msg = str(result["error"]).lower()
                print(f"[HF error]: {result['error']}")

                if "loading" in error_msg:
                    wait_time = result.get("estimated_time", 20)
                    print(f"[Model loading, waiting {wait_time}s]")
                    time.sleep(float(wait_time))
                    continue

                elif "quota" in error_msg or "rate" in error_msg:
                    # Rate limited - return friendly message
                    return "I am receiving too many requests right now. Please try again in a minute."

                elif "authorization" in error_msg or "token" in error_msg:
                    # Bad API key
                    print("[ERROR]: HuggingFace token is invalid or missing")
                    return "AI service configuration error. Please contact the administrator."

                else:
                    # Unknown error
                    print(f"[Unknown HF error]: {result}")
                    return "I apologize, the AI service is temporarily unavailable. Please try again."

            # Got a valid response
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get("generated_text", "").strip()
                if answer:
                    print(f"[AI answer]: {answer[:150]}")
                    return answer

            # Empty response
            print(f"[Empty response from HF]: {result}")
            return "I could not generate an answer. Please try again."

        # All 3 attempts failed
        return "I apologize, the AI service is temporarily busy. Please try again in a moment."

    except Exception as e:
        print(f"[AI exception]: {e}")
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

    # Step 2: Detect language
    detected_lang = detect_language(user_message)
    src_lang_code = LANGUAGE_MAP.get(detected_lang, "eng_Latn")

    # Step 3: Translate to English if needed
    if detected_lang != "en":
        english_message = translate(user_message, src_lang_code, "eng_Latn")
    else:
        english_message = user_message

    # Step 4: Search knowledge base
    knowledge_context = search_knowledge_base(english_message)
    print(f"[Knowledge]: {len(knowledge_context)} chars found")

    # Step 5: Search web
    web_context = search_tavily(english_message)
    if not web_context:
        print("[Falling back to Serper]")
        web_context = search_serper(english_message)

    # Step 6: Get AI answer
    english_answer = get_ai_answer(english_message, knowledge_context, web_context)

    # Step 7: Translate answer back if needed
    if detected_lang != "en" and src_lang_code != "eng_Latn":
        final_answer = translate(english_answer, "eng_Latn", src_lang_code)
    else:
        final_answer = english_answer

    # Step 8: Cache result
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
        "version"         : "2.1",
        "knowledge_chunks": len(knowledge_base)
    })


# =============================================================
# Start app
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
