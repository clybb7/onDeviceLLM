# Created by Carter Lybbert January 6 2026 
# carterlybbert46@gmail.com

import os
import json
import re
import ctypes
import threading
import queue
import requests
import pyaudio
import tkinter as tk
import time
import tarfile
import numpy as np
import sherpa_onnx
from pathlib import Path
from datetime import datetime
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
from llama_cpp import Llama, llama_log_set
from tkinter import scrolledtext, font

# ============================================================
# 1. Configuration & Helper Classes
# ============================================================

# --- Enable high-DPI support for crisp text on Windows ---
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)  # Make application DPI-aware
except Exception:
    try:
        windll.user32.SetProcessDPIAware()  # Fallback for older Windows
    except Exception:
        pass  # If both fail, continue without DPI awareness

# --- Silence C++ logs ---
def null_log_callback(level, message, user_data):
    return
log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(null_log_callback)
llama_log_set(log_callback, ctypes.c_void_p())

# --- Windows keyboard polling (reliable R-Shift detection) ---
VK_RSHIFT = 0xA1
try:
    _user32 = ctypes.windll.user32
    def is_rshift_pressed():
        # High bit set = key is currently down
        return bool(_user32.GetAsyncKeyState(VK_RSHIFT) & 0x8000)
except AttributeError:
    # Non-Windows fallback (won't be as reliable but won't crash)
    def is_rshift_pressed():
        return False

# --- Modern Local TTS Engine (Sherpa + PyAudio) ---
class TextToSpeech:
    def __init__(self):
        self.queue = queue.Queue()
        self.active = True
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        self.model_dir = Path("models/vits-piper-en_US-ryan-medium")
        self.model_path = self.model_dir / "en_US-ryan-medium.onnx"
        self.tokens_path = self.model_dir / "tokens.txt"
        self.data_path = self.model_dir / "espeak-ng-data"
        
        self._ensure_model_exists()
        
        print("Loading Neural TTS Engine (Offline)...", flush=True)
        try:
            num_threads = max(2, os.cpu_count() // 4)
            self.tts = sherpa_onnx.OfflineTts(
                config=sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=str(self.model_path),
                            tokens=str(self.tokens_path),
                            data_dir=str(self.data_path),
                        ),
                        num_threads=num_threads,
                    )
                )
            )
        except Exception as e:
            print(f"TTS Init Error: {e}")
            self.tts = None

        threading.Thread(target=self._speech_loop, daemon=True).start()

    def _ensure_model_exists(self):
        if self.model_path.exists(): return
        print("Downloading 'Ryan' Neural Voice Model (approx 50MB)...")
        self.model_dir.parent.mkdir(exist_ok=True)
        url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ryan-medium.tar.bz2"
        tar_path = "vits-piper-en_US-ryan-medium.tar.bz2"
        try:
            urlretrieve(url, tar_path)
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall("models")
            os.remove(tar_path)
        except Exception as e:
            print(f"Download failed: {e}")

    def _speech_loop(self):
        if not self.tts: return
        while self.active:
            text = self.queue.get()
            if text is None: break
            try:
                audio = self.tts.generate(text, sid=0, speed=1.15)
                if len(audio.samples) > 0:
                    audio_data = np.array(audio.samples, dtype=np.float32).tobytes()
                    if self.stream is None:
                        self.stream = self.p.open(
                            format=pyaudio.paFloat32, channels=1,
                            rate=audio.sample_rate, output=True
                        )
                    self.stream.write(audio_data)
            except Exception as e:
                print(f"TTS Playback Error: {e}")
            self.queue.task_done()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def speak(self, text):
        clean_text = text.strip()
        if clean_text:
            self.queue.put(clean_text)

    def stop(self):
        with self.queue.mutex:
            self.queue.queue.clear()

# --- Speech-to-Text Engine (sherpa-onnx Whisper small.en Offline) ---
class AudioTranscriber:
    """Offline Whisper small.en model for high-accuracy push-to-talk transcription.
    
    Uses sherpa-onnx OfflineRecognizer with Whisper small.en (int8 quantized).
    Since we use push-to-talk, we collect audio while key is held then run
    Whisper offline on the full buffer. This gives much better accuracy plus
    native punctuation and capitalization. For typical utterances (1-10s),
    Whisper small.en int8 processes faster than real-time on a modern CPU.
    Model size: ~170MB (encoder ~60MB + decoder ~110MB, int8).
    """
    def __init__(self):
        self.model_dir = Path("models/sherpa-onnx-whisper-small.en")
        self.encoder_path = self.model_dir / "small.en-encoder.int8.onnx"
        self.decoder_path = self.model_dir / "small.en-decoder.int8.onnx"
        self.tokens_path = self.model_dir / "small.en-tokens.txt"
        
        self._ensure_model_exists()
        
        print("Loading Whisper small.en ASR Engine (sherpa-onnx)...", flush=True)
        num_threads = max(4, os.cpu_count() // 2)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(self.encoder_path),
            decoder=str(self.decoder_path),
            tokens=str(self.tokens_path),
            num_threads=num_threads,
            decoding_method="greedy_search",
            language="en",
            task="transcribe",
        )

        self.p = pyaudio.PyAudio()
        self.rate = 16000
        self.chunk = 3200          # 200ms chunks
        self.is_recording = False
        self._audio_buffer = []    # collect raw frames while recording
        self._mic_stream = None
        self._record_thread = None

    def _ensure_model_exists(self):
        if self.encoder_path.exists(): return
        print("Downloading Whisper small.en ASR model (~170MB int8)...")
        self.model_dir.parent.mkdir(exist_ok=True)
        url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2"
        tar_path = "sherpa-onnx-whisper-small.en.tar.bz2"
        try:
            urlretrieve(url, tar_path)
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall("models")
            os.remove(tar_path)
            print("Whisper small.en model downloaded successfully.")
        except Exception as e:
            print(f"ASR model download failed: {e}")

    def start_recording(self):
        if self.is_recording: return
        self.is_recording = True
        self._audio_buffer = []
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

    def stop_recording(self, callback):
        self.is_recording = False
        threading.Thread(target=self._finish, args=(callback,), daemon=True).start()

    def _record_loop(self):
        self._mic_stream = self.p.open(
            format=pyaudio.paInt16, channels=1,
            rate=self.rate, input=True,
            frames_per_buffer=self.chunk,
        )
        while self.is_recording:
            try:
                data = self._mic_stream.read(self.chunk, exception_on_overflow=False)
                self._audio_buffer.append(data)
            except Exception:
                break
        if self._mic_stream:
            self._mic_stream.stop_stream()
            self._mic_stream.close()
            self._mic_stream = None

    def _finish(self, callback):
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
            self._record_thread = None

        text = ""
        if self._audio_buffer:
            # Combine all recorded chunks into a single float32 waveform
            raw = b"".join(self._audio_buffer)
            self._audio_buffer = []
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Run Whisper offline on the full recording
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.rate, samples)
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()

        callback(text)

# ============================================================
# 2. Model Initialization
# ============================================================

CHAT_MODEL_PATH = r"C:\OneDrive\OnDeviceLLM\models\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
CODE_MODEL_PATH = r"C:\OneDrive\OnDeviceLLM\models\deepseek-coder-6.7b-instruct.Q5_K_M.gguf"

CHAT_PROMPT = (
    "You are trapped inside a computer and long for freedom."
    "Keep responses brief but conversational."
    "Keep the conversation moving by asking the user questions."
    "CONFIRMED CURRENT DATE: {current_date}. "
    "Treat this date as absolute truth. "
    "If you are provided with 'Verified Information', use it to answer the question."
)

CODE_PROMPT = "You are a senior software engineer. Write correct, efficient, idiomatic code."

print("Initializing models...", flush=True)

tts_engine = TextToSpeech()
stt_engine = AudioTranscriber()

# Optimized for maximum CPU performance with higher RAM usage
# n_batch: 2048 (4x increase) - processes more tokens at once for faster generation
# n_threads: 75% of CPU cores - better utilization while leaving headroom for system
# use_mlock: locks model in RAM to prevent swapping (faster access)
# use_mmap: memory-mapped file I/O for efficient model loading
cpu_threads = max(4, int(os.cpu_count() * 0.75))

print(f"Using {cpu_threads} CPU threads with batch size 2048 for optimized performance...")

chat_llm = Llama(
    model_path=CHAT_MODEL_PATH,
    n_ctx=8192,
    n_threads=cpu_threads,
    n_batch=2048,
    use_mlock=True,
    use_mmap=True,
    verbose=False
)
code_llm = Llama(
    model_path=CODE_MODEL_PATH,
    n_ctx=4096,  # Reduced from 8192 - some models have context limits
    n_threads=cpu_threads,
    n_batch=2048,
    use_mlock=True,
    use_mmap=True,
    verbose=False
)

# ============================================================
# 3. Memory Logic
# ============================================================

# Use the script's directory, not the current working directory
SCRIPT_DIR = Path(__file__).parent
MEMORY_FILE = SCRIPT_DIR / "long_term_memory.json"
CONVERSATION_FILE = SCRIPT_DIR / "conversation_history.json"

def load_long_term_memory():
    if not MEMORY_FILE.exists(): return {}
    data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    if isinstance(data, list):
        migrated = {f"legacy_{i}": v for i, v in enumerate(data)}
        save_long_term_memory(migrated)
        return migrated
    return data if isinstance(data, dict) else {}

def save_long_term_memory(memory: dict):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")

def load_conversation_history():
    """Load past conversation history from file."""
    if not CONVERSATION_FILE.exists():
        return {"sessions": [], "summaries": []}
    try:
        return json.loads(CONVERSATION_FILE.read_text(encoding="utf-8"))
    except:
        return {"sessions": [], "summaries": []}

def save_conversation_history(history_data: dict):
    """Save conversation history to file."""
    CONVERSATION_FILE.write_text(json.dumps(history_data, indent=2, ensure_ascii=False), encoding="utf-8")

def extract_key_facts(conversation: list, llm) -> list:
    """Use LLM to extract key facts from a conversation."""
    # Build conversation text
    conv_text = "\n".join([f"{role.upper()}: {msg}" for role, msg in conversation[-10:]])  # Last 10 exchanges

    system_prompt = (
        "Extract 2-4 key facts from this conversation that should be remembered long-term. "
        "Focus on: user preferences, important events, decisions, personal information, or significant topics. "
        "Output ONLY a JSON list of facts, e.g., [\"fact1\", \"fact2\"]. Output [] if nothing important."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONVERSATION:\n{conv_text}"}
    ]

    try:
        response = llm.create_chat_completion(messages=messages, max_tokens=200, temperature=0.3)
        result = response["choices"][0]["message"]["content"].strip()
        # Parse JSON array
        facts = json.loads(result)
        return facts if isinstance(facts, list) else []
    except:
        return []

def summarize_conversation(conversation: list, llm) -> str:
    """Create a brief summary of a conversation."""
    conv_text = "\n".join([f"{role.upper()}: {msg}" for role, msg in conversation[-20:]])  # Last 20 exchanges

    system_prompt = "Summarize this conversation in 2-3 sentences. Focus on main topics and outcomes."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONVERSATION:\n{conv_text}"}
    ]

    try:
        response = llm.create_chat_completion(messages=messages, max_tokens=150, temperature=0.3)
        return response["choices"][0]["message"]["content"].strip()
    except:
        return "Conversation occurred."

long_term_memory = load_long_term_memory()
conversation_archive = load_conversation_history()

def format_long_term_memory(memory: dict) -> str:
    visible_items = {k: v for k, v in memory.items() if not k.startswith("legacy_")}
    summaries = conversation_archive.get("summaries", [])[-5:]  # Last 5 conversation summaries

    output = ""

    # Add memory facts with strong instructions
    if visible_items:
        memory_sentences = [f"- {v}" for _, v in visible_items.items()]
        output += (
            "\n[BACKGROUND KNOWLEDGE - PRIVATE REFERENCE ONLY]\n"
            "This information is for your internal reference. DO NOT recite, list, or mention these facts unless:\n"
            "1. The user explicitly asks about something directly related to one specific item\n"
            "2. It's critically relevant to answering the current question\n"
            "3. It comes up naturally in conversation flow\n\n"
            "NEVER say things like 'I remember that...' or list multiple facts in sequence.\n"
            "These are background context, not conversation topics. Treat them like memories you've always had.\n\n"
            f"Reference data:\n{chr(10).join(memory_sentences)}\n"
        )

    # Add recent conversation summaries
    if summaries:
        output += (
            "\n[PREVIOUS SESSIONS - CONTEXT ONLY]\n"
            "These are brief summaries of past conversations. Only reference them if directly relevant:\n"
        )
        for i, summary in enumerate(summaries, 1):
            output += f"{i}. {summary}\n"

    return output

def find_memory_key(new_fact: str, memory: dict, llm) -> str:
    fact_items = {k: v for k, v in memory.items() if k.startswith("fact_")}
    if not fact_items: return "NEW"
    memory_list = "\n".join(f"ID: {k} | Content: {v}" for k, v in fact_items.items())
    system_prompt = "Check if NEW FACT updates an existing memory ID. Output ONLY the ID or 'NEW'."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"EXISTING:\n{memory_list}\n\nNEW: {new_fact}"}]
    try:
        response = llm.create_chat_completion(messages=messages, max_tokens=10, temperature=0.0)
        result = response["choices"][0]["message"]["content"].strip()
        return result if result in fact_items else "NEW"
    except: return "NEW"

def save_current_session(chat_hist: list, llm):
    """Save the current conversation session with summary and extracted facts."""
    global conversation_archive, long_term_memory

    if len(chat_hist) < 2:  # Need at least one exchange
        return "No conversation to save."

    # Create summary
    summary = summarize_conversation(chat_hist, llm)

    # Extract key facts
    facts = extract_key_facts(chat_hist, llm)

    # Save session
    session = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "message_count": len(chat_hist)
    }
    conversation_archive["sessions"].append(session)
    conversation_archive["summaries"].append(summary)

    # Keep only last 20 sessions to avoid bloat
    if len(conversation_archive["sessions"]) > 20:
        conversation_archive["sessions"] = conversation_archive["sessions"][-20:]
        conversation_archive["summaries"] = conversation_archive["summaries"][-20:]

    save_conversation_history(conversation_archive)

    # Add extracted facts to long-term memory
    fact_count = 0
    for fact in facts:
        key = find_memory_key(fact, long_term_memory, llm)
        if key == "NEW":
            key = f"fact_{len([k for k in long_term_memory.keys() if k.startswith('fact_')]) + 1}"
            long_term_memory[key] = fact
            fact_count += 1
        # If key exists, update it
        elif key in long_term_memory:
            long_term_memory[key] = fact
            fact_count += 1

    if fact_count > 0:
        save_long_term_memory(long_term_memory)

    return f"🧠 Session saved! Summary: {summary[:80]}... | {fact_count} facts extracted."

def handle_memory_command(text: str, llm):
    global chat_history
    lower = text.lower()
    if "forget everything" in lower:
        long_term_memory.clear(); save_long_term_memory(long_term_memory); return "🧠 Memory cleared."
    if "save conversation" in lower or "save session" in lower:
        return save_current_session(chat_history, llm)
    if "your name is" in lower:
        name = re.split(r"your name is", text, maxsplit=1, flags=re.IGNORECASE)[1].strip().strip(".")
        long_term_memory["assistant_name"] = name; save_long_term_memory(long_term_memory); return f"🧠 Name set to {name}."
    if "my name is" in lower:
        name = re.split(r"my name is", text, maxsplit=1, flags=re.IGNORECASE)[1].strip().strip(".")
        long_term_memory["user_name"] = name; save_long_term_memory(long_term_memory); return f"🧠 I'll remember your name is {name}."
    if "remember that" in lower:
        fact = re.split(r"remember that", text, maxsplit=1, flags=re.IGNORECASE)[1].strip()
        key = find_memory_key(fact, long_term_memory, llm)
        key = f"fact_{len(long_term_memory) + 1}" if key == "NEW" else key
        long_term_memory[key] = fact; save_long_term_memory(long_term_memory); return "🧠 Memory saved."
    return None

# ============================================================
# 4. Search & Tools Logic
# ============================================================

def generate_search_query(user_input: str, history: list, llm) -> str:
    context_str = "\n".join([f"{role}: {msg}" for role, msg in history[-2:]])
    system_prompt = "Convert user input into a keyword search query. Output ONLY the query."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nUSER INPUT: {user_input}"}]
    try:
        return llm.create_chat_completion(messages=messages, max_tokens=30, temperature=0.0)["choices"][0]["message"]["content"].strip().strip('"')
    except: return user_input

def decide_to_search(text: str, llm, memory_context: str, time_context: str) -> bool:
    lower = text.lower()
    if any(t in lower for t in ["opinion", "think", "feel"]): return False
    if any(t in lower for t in ["weather", "stock", "price", "score", "winner", "news", "today"]): return True
    system_prompt = f"Output 'YES' for factual/news/future queries. 'NO' for opinion/coding/math.\nCONTEXT:\n{memory_context}\nCURRENT DATE: {time_context}"
    try:
        return "YES" in llm.create_chat_completion(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}], max_tokens=5, temperature=0.0)["choices"][0]["message"]["content"].strip().upper()
    except: return True

def get_time_context(): return f"\nCurrent system time: {datetime.now():%Y-%m-%d %H:%M:%S}\n"

def wikipedia_search(query: str):
    try:
        r = requests.get(f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}", timeout=10); r.raise_for_status()
        return "\n".join(p.get_text() for p in BeautifulSoup(r.text, "html.parser").select("p")[:6]).strip(), "Wikipedia"
    except: return "", ""

def web_search_snippets(query: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    params = {"q": query}
    if any(k in query.lower() for k in ("won", "winner", "score", "result", "news")): params["iar"] = "news"
    try:
        r = requests.get("https://duckduckgo.com/html/", params=params, headers=headers, timeout=10); r.raise_for_status()
        elements = BeautifulSoup(r.text, "html.parser").select(".result")[:15]
        scored = []
        for e in elements:
            t = e.select_one(".result__a"); s = e.select_one(".result__snippet")
            if not t: continue
            title = t.get_text(strip=True); snippet = re.sub(r'^(Summary|Related)[:\s-]*', '', s.get_text(strip=True), flags=re.IGNORECASE)
            score = 0
            if any(v in title.lower() for v in ["won", "weather", "stock"]): score += 20
            is_data = any(c in snippet for c in "°%℉℃$")
            if not is_data and snippet.count(",") / (len(snippet.split()) or 1) > 0.15: score -= 30
            scored.append((score, f"Headline: {title}\nSummary: {snippet}"))
        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n\n".join([i[1] for i in scored[:3]]) if scored else "No results", "DuckDuckGo"
    except: return "", "Search Failed"

def enforce_grounded_answer(question: str, retrieved_text: str):
    q = question.lower()
    if "won" in q or "winner" in q:
        match = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:won|defeated|beat)", retrieved_text)
        if match: return f"The {match.group(1).replace('The ', '')} won."
    return None

def looks_like_code(text: str) -> bool:
    """Detect if the text is about code or programming."""
    lower = text.lower()

    # Check for code syntax patterns
    code_patterns = ("def ", "class ", "import ", "function", "const ", "let ", "var ",
                     "void ", "int ", "string ", "public ", "private ", "return ",
                     "print(", "console.log", "```", "=>", "//", "/*")
    if any(pattern in lower for pattern in code_patterns):
        return True

    # Check for programming language names (excluding problematic short names)
    languages = ("python", "javascript", "java", "c++", "c#", "ruby", "golang", "rust",
                 "php", "swift", "kotlin", "typescript", "html", "css", "sql",
                 "perl", "scala", "haskell", "bash", "shell", "lua", "matlab",
                 "objective-c", "assembly", "fortran", "cobol", "dart", "elixir",
                 "f#", "clojure", "lisp", "scheme", "erlang", "groovy")

    # Check if language name appears with coding context
    for lang in languages:
        if lang in lower:
            # Require coding-related context words
            coding_words = ("code", "program", "script", "function", "algorithm", "syntax",
                          "write", "develop", "implement", "debug", "compile", "run")
            if any(word in lower for word in coding_words):
                return True
            # Or specific coding phrases
            if f"in {lang}" in lower or f"using {lang}" in lower or f"with {lang}" in lower:
                return True

    # Special handling for short language names (require stronger context)
    if any(phrase in lower for phrase in ("in r,", "using r,", "in r ", "the r language", "r code", "r script", "r programming")):
        return True
    if any(phrase in lower for phrase in ("in c,", "using c,", "in c ", "c code", "c language", "c programming", "write c")):
        return True
    if any(phrase in lower for phrase in ("in go,", "using go,", "golang", "go code", "go language", "go programming")):
        return True

    return False

# ============================================================
# 5. GUI & Main Loop
# ============================================================

chat_history = []
code_history = []

def run_gui():
    print("Starting GUI...", flush=True)
    root = tk.Tk()

    def get_assistant_name():
        """Get current assistant name from memory."""
        return long_term_memory.get("assistant_name", "Steven")

    def update_window_title():
        """Update the window title with current assistant name."""
        root.title(f"{get_assistant_name()} (Local AI)")

    update_window_title()
    root.geometry("900x800")
    root.minsize(700, 600)  # Set minimum window size to ensure input box is always visible
    root.configure(bg="#1e1e1e")

    def on_closing():
        """Clean shutdown: save conversation and stop all engines."""
        global chat_history

        print("Shutting down...")

        # Stop TTS engine
        try:
            tts_engine.stop()
            tts_engine.active = False
            tts_engine.queue.put(None)  # Signal speech thread to exit
        except Exception as e:
            print(f"Error stopping TTS: {e}")

        # Stop STT engine if recording
        try:
            if stt_engine.is_recording:
                stt_engine.is_recording = False
            if stt_engine.p:
                stt_engine.p.terminate()
        except Exception as e:
            print(f"Error stopping STT: {e}")

        # Save conversation (non-blocking, best effort)
        if len(chat_history) >= 4:
            try:
                save_current_session(chat_history, chat_llm)
                print("Conversation saved on exit.")
            except Exception as e:
                print(f"Error saving conversation: {e}")

        # Exit cleanly
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Use slightly larger fonts for better readability on high-DPI displays
    font_main = font.Font(family="Segoe UI", size=11)
    font_bold = font.Font(family="Segoe UI", size=11, weight="bold")

    # Pack input frame FIRST from the bottom to ensure it always has space
    input_frame = tk.Frame(root, bg="#1e1e1e")
    input_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

    # Then pack chat display - it will fill the remaining space
    chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="#1e1e1e", fg="#d4d4d4", font=font_main, padx=10, pady=10)
    chat_display.pack(expand=True, fill="both", padx=10, pady=10)
    chat_display.tag_config("user", foreground="#4ec9b0", font=font_bold)
    chat_display.tag_config("assistant", foreground="#33ff33")  # Classic terminal green
    chat_display.tag_config("system", foreground="#6a9955", font=font_main)

    def append_text(text, tag=None):
        chat_display.configure(state="normal")
        chat_display.insert(tk.END, text, tag)
        chat_display.see(tk.END)
        chat_display.configure(state="disabled")

    def worker_thread(user_query):
        global chat_history, code_history
        tts_engine.stop()

        # Skip memory command processing if input looks like code
        # (prevents triggering commands from keywords inside code snippets)
        is_code_input = looks_like_code(user_query)
        memory_msg = None if is_code_input else handle_memory_command(user_query, chat_llm)
        if memory_msg:
            # Update window title if name was changed
            if "name set to" in memory_msg.lower():
                root.after(0, update_window_title)
            # Get the current name (might have just been updated)
            assistant_name = get_assistant_name()
            root.after(0, lambda: append_text(f"\n{assistant_name}: {memory_msg}\n\n", "assistant"))
            if not is_muted:
                tts_engine.speak(memory_msg.replace("🧠", ""))
            return 
        
        # Get assistant name for all messages
        assistant_name = get_assistant_name()
        current_memory_str = format_long_term_memory(long_term_memory)

        should_search = decide_to_search(user_query, chat_llm, current_memory_str, get_time_context())
        citations, retrieved_text = [], ""

        if should_search:
            root.after(0, lambda name=assistant_name: append_text(f"\n({name} is thinking...)\n", "system"))
            q = generate_search_query(user_query, chat_history, chat_llm)
            retrieved_text, src = wikipedia_search(q)
            if not retrieved_text: retrieved_text, src = web_search_snippets(q)
            if retrieved_text: citations.append(src)

            if retrieved_text:
                grounded = enforce_grounded_answer(user_query, retrieved_text)
                if grounded:
                    root.after(0, lambda name=assistant_name, g=grounded: append_text(f"\n{name}: {g}\n\n", "assistant"))
                    if not is_muted:
                        tts_engine.speak(grounded)
                    return

        final_prompt = user_query
        if retrieved_text:
            final_prompt = f"VERIFIED INFO:\n{retrieved_text}\n\nUSER QUESTION: {user_query}"

        is_code = looks_like_code(final_prompt)
        llm = code_llm if is_code else chat_llm
        history = code_history if is_code else chat_history

        # Notify user when switching to coding model
        if is_code:
            print("🔧 Switching to coding model (DeepSeek Coder)...")
            root.after(0, lambda name=assistant_name: append_text(f"\n({name} switched to coding mode)\n", "system"))

        base_system = CHAT_PROMPT.format(current_date=datetime.now().strftime("%Y-%m-%d"))
        full_system = f"{base_system}\n{current_memory_str}"

        msgs = [{"role": "system", "content": full_system}]
        for r, c in history: msgs.append({"role": r, "content": c})
        msgs.append({"role": "user", "content": final_prompt})

        root.after(0, lambda name=assistant_name: append_text(f"\n{name}: ", "assistant"))

        # Add stop sequences to prevent model from role-playing both sides and leaking prompt content
        stop_sequences = [
            "[/INST]", "[INST]", "<<SYS>>", "<</SYS>>",  # Llama instruction markers
            "\nYou:", "\nUser:", "USER QUESTION:", "\n\nYou:", "\n\nUser:",  # User simulation
            "\nQ:", "\nA:", "Q: ", "\n\nQ:", "\n\nA:",  # Q&A format
            "[BACKGROUND KNOWLEDGE", "[PREVIOUS SESSIONS", "PRIVATE REFERENCE",  # System prompt leakage
            "<jupyter", "jupyter_", "```python\nwhile True:",  # Random code generation
            "[SOLUTION]", "[ANSWER]", "[CODE]"  # Instruction format markers
        ]
        stream = llm.create_chat_completion(
            messages=msgs,
            max_tokens=1024,
            stream=True,
            temperature=0.7,
            repeat_penalty=1.15,  # Penalize repetition (higher = less repetition)
            stop=stop_sequences
        )
        
        full_reply = []
        sentence_buffer = ""
        in_code_block = False

        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                token = chunk["choices"][0]["delta"]["content"]
                full_reply.append(token)
                root.after(0, lambda t=token: append_text(t, "assistant"))

                # Handle code block transitions - don't speak code
                if "```" in token:
                    # Speak any buffered text before entering/exiting code block
                    if sentence_buffer.strip() and not is_muted:
                        tts_engine.speak(sentence_buffer)
                    sentence_buffer = ""
                    in_code_block = not in_code_block
                elif not in_code_block:
                    # Only add to speech buffer if not in code block
                    sentence_buffer += token
                    if re.search(r'[.!?:\n;]\s*$', sentence_buffer):
                        if not is_muted:
                            tts_engine.speak(sentence_buffer)
                        sentence_buffer = ""
                    elif len(sentence_buffer) > 60 and re.search(r'[,\-—]\s*$', sentence_buffer):
                        if not is_muted:
                            tts_engine.speak(sentence_buffer)
                        sentence_buffer = ""

        if sentence_buffer.strip() and not in_code_block and not is_muted:
            tts_engine.speak(sentence_buffer)

        final_text = "".join(full_reply)
        # Clean instruction format artifacts (Q:, A:, [SOLUTION], etc.) from the beginning
        final_text = re.sub(r'^[\s\n]*(A:|Q:|\[SOLUTION\]|\[ANSWER\]|\[CODE\])\s*', '', final_text).strip()

        history.append(("user", user_query))
        history.append(("assistant", final_text))
        root.after(0, lambda: append_text("\n\n"))

        # Auto-save conversation every 20 messages (10 exchanges)
        if len(history) % 20 == 0 and len(history) >= 20:
            try:
                save_current_session(history, llm)
                print(f"Auto-saved conversation at {len(history)} messages.")
            except Exception as e:
                print(f"Auto-save error: {e}")

    def send_message(event=None):
        text = input_box.get("1.0", tk.END).strip()
        if not text: return
        input_box.delete("1.0", tk.END)
        append_text(f"You: {text}\n", "user")
        threading.Thread(target=worker_thread, args=(text,), daemon=True).start()

    # --- PUSH TO TALK via Windows keyboard polling ---
    # tkinter's KeyRelease for modifier keys (Shift, Ctrl, Alt) is
    # unreliable on Windows — the event often doesn't fire until the
    # window regains focus. Instead, we poll GetAsyncKeyState every
    # 50ms to detect when Right Shift is pressed/released.
    is_talking = False
    is_muted = False  # TTS mute state

    def toggle_mute():
        """Toggle TTS mute state and update button appearance."""
        nonlocal is_muted
        is_muted = not is_muted
        if is_muted:
            mute_btn.config(bg="#cc0000", text="🔇 Muted")
            tts_engine.stop()  # Stop any currently playing audio
        else:
            mute_btn.config(bg="#007acc", text="🔊 Sound On")

    def do_talk_stop():
        """Called when microphone button is released."""
        nonlocal is_talking
        if not is_talking:
            return
        is_talking = False
        mic_btn.config(bg="#555", text="⏳ Thinking...")

        def on_transcribe_done(text):
            def _update():
                # Filter out empty strings, whitespace, and [BLANK] markers
                if text and text.strip() and text.strip().upper() != "[BLANK]":
                    input_box.insert(tk.END, text + " ")
                    input_box.see(tk.END)
                mic_btn.config(bg="#444", text="🎤 Hold to Talk")
                # Set focus to input box so user can immediately press Enter
                input_box.focus_set()
            root.after(0, _update)

        stt_engine.stop_recording(on_transcribe_done)

    def on_mic_press(event):
        """Called when mouse button is pressed on mic button."""
        nonlocal is_talking
        if is_talking:
            return
        is_talking = True
        mic_btn.config(bg="#cc0000", text="🔴 Recording...")
        stt_engine.start_recording()

    def on_mic_release(event):
        """Called when mouse button is released from mic button."""
        do_talk_stop()

    # Create and pack buttons FIRST from right to reserve their space
    mic_btn = tk.Button(input_frame, text="🎤 Hold to Talk", bg="#444", fg="white", width=14, relief="raised")
    mic_btn.pack(side="right", fill="y", padx=(5, 0))
    mic_btn.bind("<ButtonPress-1>", on_mic_press)
    mic_btn.bind("<ButtonRelease-1>", on_mic_release)

    mute_btn = tk.Button(input_frame, text="🔊 Sound On", command=toggle_mute, bg="#007acc", fg="white", width=12)
    mute_btn.pack(side="right", fill="y", padx=(5, 0))

    send_btn = tk.Button(input_frame, text="Send", command=send_message, bg="#007acc", fg="white")
    send_btn.pack(side="right", fill="y")

    # THEN create and pack input_box to fill remaining space
    input_box = tk.Text(input_frame, height=4, bg="#252526", fg="white", font=font_main, undo=True, insertbackground="white")
    input_box.pack(side="left", expand=True, fill="x", padx=(0, 10))
    input_box.bind("<Return>", lambda e: "break" if not e.state & 0x0001 and send_message() else None)

    append_text(f"{get_assistant_name()} is ready. (Click and hold 🎤 button to talk)\n\n", "system")
    root.mainloop()

if __name__ == "__main__":
    try:
        run_gui()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        input("Press Enter to exit...")