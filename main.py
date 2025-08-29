import os
import uuid
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, List

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kuber-clone")

# Optional TTS (Coqui) 
try:
    from TTS.api import TTS
except Exception as e:
    TTS = None
    log.warning("Coqui TTS import failed. Install TTS or skip voice playback. Err: %s", e)

#  Config (env first, else defaults you gave)
GROQ_KEY = os.getenv("GROQ_KEY", "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")
CHATANYWHERE_KEY = os.getenv("CHATANYWHERE_KEY", "sk-dsaE8XgYOulfcyPeHuCPsYCh71N3f9aNT4hJO8cRfSlU7HUH")
GOLDAPI_KEY = os.getenv("GOLDAPI_KEY", "goldapi-rrsmev9d8gj-io")

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

# TTS model (lightweight VITS)
TTS_MODEL_NAME = os.getenv("TTS_MODEL", "tts_models/en/vctk/vits")
_tts = None

def get_tts():
    global _tts
    if _tts:
        return _tts
    if TTS is None:
        raise RuntimeError("TTS library not available. Install Coqui TTS or skip TTS.")
    _tts = TTS(TTS_MODEL_NAME)
    return _tts

# ---------- Gold detection ----------
GOLD_KEYWORDS = ["gold", "sona", "सोना", "digital gold", "dgold", "digital-gold", "digitalgold", "gold price"]

def looks_like_gold_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in GOLD_KEYWORDS)

def nudge_snippet(locale: str = "en"):
    if locale.startswith("hi"):
        return ("(BTW — agar aap sona mein ruchi rakhte hain, toh Digital Gold jaise options dekh sakte hain. "
                "Agar chaho main aur details bata sakta hoon.)")
    return ("(Also — if you're exploring gold, consider checking Digital Gold as an easy, flexible option. "
            "I can explain more if you want.)")

# ---------- Simple in-memory chat history per user ----------
# { user_id: [ {"role":"user"/"assistant", "content": "..."} ] }
CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}
MAX_TURNS = 8  # last 8 turns kept

def add_to_history(user_id: str, role: str, content: str):
    CHAT_HISTORY.setdefault(user_id, [])
    CHAT_HISTORY[user_id].append({"role": role, "content": content})
    # trim
    if len(CHAT_HISTORY[user_id]) > MAX_TURNS:
        CHAT_HISTORY[user_id] = CHAT_HISTORY[user_id][-MAX_TURNS:]

def get_history(user_id: str) -> List[Dict[str, str]]:
    return CHAT_HISTORY.get(user_id, [])

# ---------- FastAPI app ----------
app = FastAPI(title="Kuber.AI Clone")

# CORS (same-origin serve kar rahe hain; fir bhi open rakha for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from ./public (index.html)
PUBLIC_DIR = BASE_DIR / "public"
PUBLIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")

# Utils: STT / Prices / LLM / TTS 
def transcribe_with_groq(audio_path: str) -> str:
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_KEY}"}
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "application/octet-stream")}
        data = {"model": "whisper-large-v3-turbo"}
        try:
            r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
            r.raise_for_status()
            j = r.json()
            return j.get("text") or j.get("transcript") or ""
        except Exception as e:
            log.error("Groq STT error: %s", e)
            raise

def fetch_gold_price_inr() -> Optional[float]:
    try:
        url = "https://www.goldapi.io/api/XAU/INR"
        headers = {"x-access-token": GOLDAPI_KEY, "Content-Type": "application/json"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
        price = data.get("price")
        return float(price) if price else None
    except Exception as e:
        log.warning("GoldAPI fetch failed: %s", e)
        return None

def ask_gpt5_small(user_id: str, prompt: str, system: Optional[str] = None) -> str:
    url = "https://api.chatanywhere.org/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CHATANYWHERE_KEY}", "Content-Type": "application/json"}

    system_msg = system or (
        "You are a friendly Indian personal finance assistant. "
        "Be concise. End with: 'This is general information, not financial advice.'"
    )

# Include short history for conversational feel
    history = get_history(user_id)
    messages = [{"role": "system", "content": system_msg}] + history + [{"role": "user", "content": prompt}]

    payload = {
        "model": "gpt-5-small",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 500,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        log.error("LLM error: %s", e)
        raise

def tts_to_file(text: str, out_path: str):
    tts = get_tts()
    tts.tts_to_file(text=text, file_path=out_path)

# Health 
@app.get("/api/health")
def health():
    return {"ok": True}

# Text chat 
@app.post("/api/chat_text")
async def chat_text(
    user_id: str = Form(...),
    message: str = Form(...),
    topic: str = Form("General"),
    locale: str = Form("en"),
):
    if not message:
        raise HTTPException(status_code=400, detail="message required")

    base_prompt = f"Topic: {topic}\nUser: {message}"
    is_gold = looks_like_gold_question(message)
    gold_line = ""
    if is_gold:
        price = fetch_gold_price_inr()
        gold_line = f"Current gold price ≈ ₹{price:.2f} per gram." if price else "Could not fetch live gold price right now."
        base_prompt += f"\n\n{gold_line}\n{nudge_snippet(locale)}"

    # history add user msg
    add_to_history(user_id, "user", message)

    try:
        reply_text = ask_gpt5_small(user_id, base_prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {e}")

    # history add bot msg
    add_to_history(user_id, "assistant", reply_text)

    audio_url = None
    try:
        audio_filename = f"resp_{uuid.uuid4().hex}.wav"
        out_path = AUDIO_DIR / audio_filename
        tts_to_file(reply_text, str(out_path))
        audio_url = f"/api/audio/{audio_filename}"
    except Exception as e:
        log.warning("TTS failed: %s", e)

    return {
        "user_text": message,
        "response_text": reply_text,
        "audio_url": audio_url,
        "is_gold_nudge": is_gold,
        "gold_info": gold_line,
    }

# Voice chat
@app.post("/api/chat_voice")
async def chat_voice(
    audio: UploadFile = File(...),
    user_id: str = Form("guest"),
    topic: str = Form("General"),
    locale: str = Form("en"),
):
    suffix = Path(audio.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        user_text = transcribe_with_groq(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    base_prompt = f"Topic: {topic}\nUser: {user_text}"
    is_gold = looks_like_gold_question(user_text)
    gold_line = ""
    if is_gold:
        price = fetch_gold_price_inr()
        gold_line = f"Current gold price ≈ ₹{price:.2f} per gram." if price else "Could not fetch live gold price right now."
        base_prompt += f"\n\n{gold_line}\n{nudge_snippet(locale)}"

    add_to_history(user_id, "user", user_text)

    try:
        reply_text = ask_gpt5_small(user_id, base_prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {e}")

    add_to_history(user_id, "assistant", reply_text)

    audio_url = None
    try:
        audio_filename = f"resp_{uuid.uuid4().hex}.wav"
        out_path = AUDIO_DIR / audio_filename
        tts_to_file(reply_text, str(out_path))
        audio_url = f"/api/audio/{audio_filename}"
    except Exception as e:
        log.warning("TTS failed: %s", e)

    return {
        "user_text": user_text,
        "response_text": reply_text,
        "audio_url": audio_url,
        "is_gold_nudge": is_gold,
        "gold_info": gold_line,
    }

# Serve audio 
@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    p = AUDIO_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(str(p), media_type="audio/wav")
