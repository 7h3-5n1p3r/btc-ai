# =============================================================
# BTR AI Middleware - app.py (Updated - No Flowise needed)
# =============================================================
# This file handles EVERYTHING:
#   1. Language detection
#   2. Translation (Bodo, Assamese, Hindi, English)
#   3. Web search (Tavily + Serper)
#   4. Knowledge base (your BTR text files)
#   5. AI answering (HuggingFace Mistral)
# =============================================================

import os
import json
import time
import hashlib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from langdetect import detect
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Allow requests from any domain
CORS(app)

# =============================================================
# Load API keys
# =============================================================
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY    = os.getenv("SERPER_API_KEY")

# =============================================================
# Cache setup - saves API calls for repeated questions
# =============================================================
cache = {}
CACHE_EXPIRY = 3600  # 1 hour

# =============================================================
# Language code mapping for IndicTrans2
# =============================================================
LANGUAGE_MAP = {
    "hi"  : "hin_Deva",  # Hindi
    "bn"  : "ben_Beng",  # Bengali
    "as"  : "asm_Beng",  # Assamese
    "ne"  : "npi_Deva",  # Nepali
    "en"  : "eng_Latn",  # English
}

# =============================================================
# Knowledge base - loads your BTR text files into memory
# Place your .txt files in a folder called 'knowledge' in repo
# =============================================================
knowledge_base = []

def load_knowledge_base():
    """
    Reads all .txt files from the 'knowledge' folder
    and stores them in memory as a list of text chunks.
    Call this once when the app starts.
    """
    global knowledge_base
    knowledge_folder = os.path.join(os.path.dirname(__file__), "knowledge")

    # Create folder if it doesn't exist
    if not os.path.exists(knowledge_folder):
        os.makedirs(knowledge_folder)
        print("[Knowledge base]: No knowledge folder found, created empty one")
        return

    # Read all .txt files in the knowledge folder
    for filename in os.listdir(knowledge_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(knowledge_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # Split into chunks of 500 characters with 50 char overlap
                # Smaller chunks = more precise answers
                chunk_size = 500
                overlap = 50
                for i in range(0, len(content), chunk_size - overlap):
                    chunk = content[i:i + chunk_size]
                    if chunk.strip():  # Skip empty chunks
                        knowledge_base.append({
                            "source": filename,
                            "content": chunk
                        })
                print(f"[Knowledge base]: Loaded {filename}")

    print(f"[Knowledge base]: Total chunks loaded: {len(knowledge_base)}")

# Load knowledge base when app starts
load_knowledge_base()


# =============================================================
# FUNCTION: Search knowledge base for relevant chunks
# Simple keyword matching - finds chunks containing query words
# =============================================================
def search_knowledge_base(query, top_k=3):
    """
    Searches the knowledge base for chunks relevant to the query.
    Returns top_k most relevant chunks as a single string.
    """
    if not knowledge_base:
        return ""

    # Score each chunk based on how many query words it contains
    query_words = query.lower().split()
    scored_chunks = []

    for chunk in knowledge_base:
        content_lower = chunk["content"].lower()
        # Count how many query words appear in this chunk
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            scored_chunks.append((score, chunk))

    # Sort by score (highest first) and take top_k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored_chunks[:top_k]

    if not top_chunks:
        return ""

    # Format results
    result = ""
    for score, chunk in top_chunks:
        result += f"From {chunk['source']}:\n{chunk['content']}\n\n"

    return result


# =============================================================
# FUNCTION: Detect language
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
# =============================================================
def translate(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text

    try:
        # Choose correct model direction
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
# FUNCTION: Get AI answer from HuggingFace Mistral
# Combines knowledge base + web search results
# =============================================================
def get_ai_answer(question, knowledge_context, web_context):
    try:
        # Build the full prompt for Mistral
        # We give it the knowledge base AND web results
        # so it can answer from both sources
        prompt = f"""You are a helpful AI assistant for Bodoland Territorial Region (BTR), Assam, India.
You help people with information about BTR culture, government, tourism, history and current events.
Answer clearly, accurately and respectfully. Keep answers concise but complete.

KNOWLEDGE BASE INFORMATION:
{knowledge_context if knowledge_context else "No specific knowledge base information found."}

WEB SEARCH RESULTS:
{web_context if web_context else "No web search results available."}

USER QUESTION: {question}

Please provide a helpful and accurate answer based on the above information.
If you don't know something, say so honestly rather than making up information.
Answer:"""

        # Call Mistral on HuggingFace
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers={
                "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,   # Max length of answer
                    "temperature": 0.7,       # 0=factual, 1=creative
                    "return_full_text": False  # Only return the answer, not the prompt
                }
            },
            timeout=60  # Mistral can take up to 60 seconds
        )

        result = response.json()

        # Extract the generated text
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get("generated_text", "").strip()
            print(f"[AI answer]: {answer[:100]}")
            return answer
        else:
            print(f"[AI error]: {result}")
            return "I apologize, the AI service is temporarily busy. Please try again in a moment."

    except Exception as e:
        print(f"[AI error]: {e}")
        return "I apologize, I could not generate an answer right now. Please try again."


# =============================================================
# MAIN ROUTE: /chat
# Your website widget calls this endpoint
# =============================================================
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data or not data.get("message"):
        return jsonify({"error": "No message provided"}), 400

    user_message = data.get("message", "").strip()
    session_id   = data.get("session_id", "default")

    print(f"\n[New message]: {user_message[:100]}")

    # STEP 1: Check cache
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

    # STEP 2: Detect language
    detected_lang = detect_language(user_message)
    src_lang_code = LANGUAGE_MAP.get(detected_lang, "eng_Latn")

    # STEP 3: Translate to English if needed
    if detected_lang != "en":
        english_message = translate(user_message, src_lang_code, "eng_Latn")
    else:
        english_message = user_message

    # STEP 4: Search knowledge base
    knowledge_context = search_knowledge_base(english_message)
    print(f"[Knowledge base]: Found {len(knowledge_context)} chars of context")

    # STEP 5: Search web (Tavily first, Serper as backup)
    web_context = search_tavily(english_message)
    if not web_context:
        print("[Falling back to Serper]")
        web_context = search_serper(english_message)

    # STEP 6: Get AI answer
    english_answer = get_ai_answer(english_message, knowledge_context, web_context)

    # STEP 7: Translate answer back to user's language
    if detected_lang != "en" and src_lang_code != "eng_Latn":
        final_answer = translate(english_answer, "eng_Latn", src_lang_code)
    else:
        final_answer = english_answer

    # STEP 8: Cache and return
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
        "version"         : "2.0",
        "knowledge_chunks": len(knowledge_base)
    })


# =============================================================
# Start app
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
