"""Unit tests for TTSEngine."""

import time
from sign_language.audio import TTSEngine


def test_tts_engine_lifecycle():
    engine = TTSEngine()
    assert engine._worker_thread.is_alive()

    # Enqueue speech job
    engine.speak("Testing voice synthesis")

    time.sleep(0.1)
    engine.stop()
    assert not engine._worker_thread.is_alive()
