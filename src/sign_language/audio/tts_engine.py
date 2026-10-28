"""Non-blocking, offline Text-to-Speech (TTS) engine with cross-platform fallbacks."""

from __future__ import annotations
import sys
import queue
import threading
import subprocess
from typing import Optional


class TTSEngine:
    """Thread-safe, non-blocking offline speech synthesizer."""

    def __init__(self, speech_rate: int = 160, volume: float = 1.0):
        self.speech_rate = speech_rate
        self.volume = volume
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._current_proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._worker_thread.start()

    def speak(self, text: str) -> None:
        """Enqueue text for asynchronous, non-blocking audible synthesis."""
        clean = text.strip()
        if clean:
            self._queue.put(clean)

    def _speech_loop(self) -> None:
        """Background worker consuming speech jobs."""
        is_macos = sys.platform == "darwin"
        engine = None

        if not is_macos:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.speech_rate)
                engine.setProperty("volume", self.volume)
            except Exception as e:
                print(f"[TTSEngine] pyttsx3 init error: {e}")
                engine = None

        while not self._stop_event.is_set():
            try:
                phrase = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if phrase is None or self._stop_event.is_set():
                break

            try:
                if is_macos:
                    rate_wpm = str(self.speech_rate)
                    with self._proc_lock:
                        if self._stop_event.is_set():
                            break
                        self._current_proc = subprocess.Popen(["say", "-r", rate_wpm, phrase])
                    self._current_proc.wait()
                    with self._proc_lock:
                        self._current_proc = None
                elif engine is not None:
                    engine.say(phrase)
                    engine.runAndWait()
                else:
                    if sys.platform.startswith("linux"):
                        subprocess.run(["espeak", phrase], check=False)
                    elif sys.platform.startswith("win"):
                        ps_cmd = f"Add-Type -AssemblyName System.speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{phrase}')"
                        subprocess.run(["powershell", "-Command", ps_cmd], check=False)
            except Exception as ex:
                print(f"[TTSEngine] Error speaking phrase '{phrase}': {ex}")
            finally:
                self._queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop background worker thread and kill any running speech process."""
        self._stop_event.set()
        with self._proc_lock:
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
        self._queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
