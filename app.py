# =============================================================
# BTR AI Middleware - app.py
# =============================================================
# This file is the bridge between:
#   1. Your website's chat widget (frontend)
#   2. The translation system (IndicTrans2 via HuggingFace)
#   3. The web search (Tavily + Serper)
#   4. The AI agent (Flowise)
#
# Flow: User message → detect language → translate to English
#       → search web → ask AI → translate answer back → send to user
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

# Allow requests from any domain (needed so your cPanel site can call this API)
CORS(app)

# =============================================================
# Load API keys from environment variables
# Never hardcode keys directly in code - always use env vars
# =============================================================
HUGGINGFACE_TOKEN  = os.getenv("HUGGINGFACE_TOKEN")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY     = os.getenv("SERPER_API_KEY")
FLOWISE_API_URL    = os.getenv("FLOWISE_API_URL")
FLOWISE_CHATFLOW_ID = os.getenv("FLOWISE_CHATFLOW_ID")

# =============================================================
# Simple in-memory cache
# Stores recent answers so we don't call APIs for repeated questions
# Key = MD5 hash of the question, Value = {answer, timestamp}
# Cache expires after 1 hour (3600 seconds)
# =============================================================
cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

# =============================================================
# Language code mapping
# Maps langdetect codes to HuggingFace IndicTrans2 language codes
# Add more languages here if needed later
# =============================================================
LANGUAGE_MAP = {
    "hi"  : "hin_Deva",   # Hindi
    "bn"  : "ben_Beng",   # Bengali
    "as"  : "asm_Beng",   # Assamese
    "ne"  : "npi_Deva",   # Nepali
    "en"  : "eng_Latn",   # English
    "brx" : "brx_Deva",   # Bodo (we handle this specially below)
}


# =============================================================
# FUNCTION: Detect language of incoming message
# Returns a language code like 'en', 'hi', 'as' etc.
# Defaults to 'en' if detection fails
# =============================================================
def detect_language(text):
    try:
        lang = detect(text)
        print(f"[Language detected]: {lang}")
        return lang
    except Exception as e:
        print(f"[Language detection failed]: {e}")
        return "en"  # Default to English if detection fails


# =============================================================
# FUNCTION: Translate text using HuggingFace IndicTrans2
# src_lang and tgt_lang use codes from LANGUAGE_MAP above
# Example: translate("नमस्ते", "hin_Deva", "eng_Latn")
# =============================================================
def translate(text, src_lang, tgt_lang):

    # If source and target are the same, no translation needed
    if src_lang == tgt_lang:
        return text

    try:
        # IndicTrans2 model - best for Indian languages including Bodo
        # ai4bharat/indictrans2-indic-en-1B = any Indian language → English
        # ai4bharat/indictrans2-en-indic-1B = English → any Indian language
        if tgt_lang == "eng_Latn":
            model = "ai4bharat/indictrans2-indic-en-1B"
        else:
            model = "ai4bharat/indictrans2-en-indic-1B"

        # Call HuggingFace Inference API
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
            timeout=30  # Wait max 30 seconds for response
        )

        result = response.json()

        # HuggingFace returns a list with translation_text
        if isinstance(result, list) and len(result) > 0:
            translated = result[0].get("translation_text", text)
            print(f"[Translation]: {text[:50]} → {translated[:50]}")
            return translated
        else:
            print(f"[Translation failed, returning original]: {result}")
            return text  # Return original if translation fails

    except Exception as e:
        print(f"[Translation error]: {e}")
        return text  # Return original text if anything goes wrong


# =============================================================
# FUNCTION: Search the web using Tavily (primary)
# Returns formatted search results as a string for the AI to read
# =============================================================
def search_tavily(query):
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query + " Bodoland BTR Assam India",  # Bias results to BTR
                "max_results": 5,                              # Get top 5 results
                "search_depth": "advanced",                    # Fetch full page content
                "include_answer": True                         # Get AI summary too
            },
            timeout=15
        )

        data = response.json()
        results_text = ""

        # Add Tavily's own AI answer if available
        if data.get("answer"):
            results_text += f"Summary: {data['answer']}\n\n"

        # Add individual search results
        for r in data.get("results", []):
            results_text += f"Source: {r.get('url', '')}\n"
            results_text += f"Content: {r.get('content', '')}\n\n"

        print(f"[Tavily search done]: {len(data.get('results', []))} results")
        return results_text

    except Exception as e:
        print(f"[Tavily error]: {e}")
        return None  # Return None so we can fallback to Serper


# =============================================================
# FUNCTION: Search the web using Serper (backup/Google results)
# Called only if Tavily fails or returns no results
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
                "num": 5  # Top 5 Google results
            },
            timeout=15
        )

        data = response.json()
        results_text = ""

        # Extract organic search results
        for r in data.get("organic", []):
            results_text += f"Source: {r.get('link', '')}\n"
            results_text += f"Title: {r.get('title', '')}\n"
            results_text += f"Content: {r.get('snippet', '')}\n\n"

        print(f"[Serper search done]: {len(data.get('organic', []))} results")
        return results_text

    except Exception as e:
        print(f"[Serper error]: {e}")
        return ""  # Return empty string if both searches fail


# =============================================================
# FUNCTION: Get answer from Flowise AI agent
# Sends the question + web search results to your Flowise chatbot
# =============================================================
def ask_flowise(question, search_context, session_id):
    try:
        # Build enriched prompt combining user question + web results
        enriched_question = f"""
You are a helpful AI assistant for Bodoland Territorial Region (BTR), Assam, India.
Answer the following question using the web search results provided below.
If the web results don't have enough info, use your own knowledge about BTR.
Always be accurate, helpful and culturally respectful.

User Question: {question}

Web Search Results:
{search_context}

Please give a clear, accurate and helpful answer.
"""

        # Call Flowise API
        response = requests.post(
            f"{FLOWISE_API_URL}/api/v1/prediction/{FLOWISE_CHATFLOW_ID}",
            json={
                "question": enriched_question,
                "sessionId": session_id  # Keeps conversation history per user
            },
            timeout=60  # AI can take up to 60 seconds to respond
        )

        data = response.json()
        answer = data.get("text", "Sorry, I could not find an answer.")
        print(f"[Flowise answer received]: {answer[:100]}")
        return answer

    except Exception as e:
        print(f"[Flowise error]: {e}")
        return "Sorry, the AI service is temporarily unavailable. Please try again."


# =============================================================
# MAIN ROUTE: /chat
# This is the endpoint your website widget calls
# Method: POST
# Input JSON: { "message": "user question", "session_id": "abc123" }
# Output JSON: { "answer": "AI response", "language": "en" }
# =============================================================
@app.route("/chat", methods=["POST"])
def chat():

    # Get data from the incoming request
    data = request.get_json()

    # Validate that message exists
    if not data or not data.get("message"):
        return jsonify({"error": "No message provided"}), 400

    user_message = data.get("message", "").strip()
    session_id   = data.get("session_id", "default")  # Track conversation per user

    print(f"\n[New message]: {user_message[:100]}")

    # ----------------------------------------------------------
    # STEP 1: Check cache first
    # If same question was asked recently, return cached answer
    # This saves API calls and makes responses faster
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # STEP 2: Detect the language of the user's message
    # ----------------------------------------------------------
    detected_lang = detect_language(user_message)

    # Get the IndicTrans2 language code (default to English if unknown)
    src_lang_code = LANGUAGE_MAP.get(detected_lang, "eng_Latn")

    # ----------------------------------------------------------
    # STEP 3: Translate message to English if it's not English
    # All AI processing happens in English for best accuracy
    # ----------------------------------------------------------
    if detected_lang != "en":
        english_message = translate(user_message, src_lang_code, "eng_Latn")
    else:
        english_message = user_message

    # ----------------------------------------------------------
    # STEP 4: Search the web for relevant information
    # Try Tavily first, fall back to Serper if it fails
    # ----------------------------------------------------------
    search_results = search_tavily(english_message)

    if not search_results:
        print("[Falling back to Serper search]")
        search_results = search_serper(english_message)

    # ----------------------------------------------------------
    # STEP 5: Ask the Flowise AI agent for an answer
    # Passes both the question and web search results
    # ----------------------------------------------------------
    english_answer = ask_flowise(english_message, search_results, session_id)

    # ----------------------------------------------------------
    # STEP 6: Translate the answer back to user's original language
    # ----------------------------------------------------------
    if detected_lang != "en" and src_lang_code != "eng_Latn":
        final_answer = translate(english_answer, "eng_Latn", src_lang_code)
    else:
        final_answer = english_answer

    # ----------------------------------------------------------
    # STEP 7: Save to cache and return answer to user
    # ----------------------------------------------------------
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
# HEALTH CHECK ROUTE: /health
# Used by Render to check if your app is running fine
# Visit yourapp.onrender.com/health to confirm it's alive
# =============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status" : "running",
        "service": "BTR AI Middleware",
        "version": "1.0"
    })


# =============================================================
# Start the Flask app
# Port 10000 is what Render expects by default
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
```

- Click **"Commit new file"**

---

## Your Repo is Now Ready

Your GitHub repo should look like this:
```
btr-ai/
├── app.py              ✅
├── requirements.txt    ✅
├── .env.example        ✅
└── README.md           ✅
