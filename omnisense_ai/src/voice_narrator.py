"""
OmniSense AI — Voice Narration Pipeline
========================================
Converts LLM analysis output into natural-sounding speech using:
  - gTTS (Google Text-to-Speech) — zero-cost, offline-ready
  - ElevenLabs API — production-grade voice (optional)
  - Audio post-processing via pydub

Author : Pranay Sai Vanam
Stack  : gTTS · pydub · ElevenLabs API · Python
"""

import io
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceProvider(str, Enum):
    GTTS       = "gtts"        # Free, no API key needed
    ELEVENLABS = "elevenlabs"  # Premium quality, requires API key


class VoiceSpeed(str, Enum):
    SLOW   = "slow"
    NORMAL = "normal"
    FAST   = "fast"


@dataclass
class NarrationResult:
    text          : str
    audio_path    : str
    provider      : VoiceProvider
    duration_secs : float
    file_size_kb  : float
    language      : str


# ─── Text Preprocessor ────────────────────────────────────────────────────────

class NarrationPreprocessor:
    """
    Cleans LLM output before passing to TTS to improve narration quality.
    Removes markdown, shortens overly long outputs, handles special characters.
    """

    MAX_TTS_CHARS = 5000  # gTTS limit per call

    @classmethod
    def clean(cls, text: str) -> str:
        """Strip markdown, normalize whitespace, trim to TTS-safe length."""
        import re

        # Remove markdown formatting
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # **bold**
        text = re.sub(r"\*(.*?)\*",     r"\1", text)   # *italic*
        text = re.sub(r"#+\s",          "",    text)   # # headers
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)  # `code`
        text = re.sub(r"\n{2,}", ". ",         text)   # double newlines → pause
        text = re.sub(r"\n",     " ",          text)   # single newlines → space
        text = re.sub(r"\s{2,}", " ",          text)   # collapse whitespace

        # Trim to safe TTS length
        if len(text) > cls.MAX_TTS_CHARS:
            text = text[:cls.MAX_TTS_CHARS] + "... Analysis truncated for audio playback."

        return text.strip()

    @classmethod
    def add_intro(cls, text: str, mode: str) -> str:
        """Prepend a contextual intro for better listener experience."""
        intros = {
            "describe"      : "Here is my visual analysis. ",
            "compare"       : "Comparing image against text claim. ",
            "extract"       : "Extracted information from the image. ",
            "sentiment"     : "Emotional and tonal analysis follows. ",
            "accessibility" : "Image description for accessibility. ",
            "summarize"     : "Summary of the document. ",
        }
        return intros.get(mode, "Analysis result. ") + text


# ─── gTTS Voice Engine ────────────────────────────────────────────────────────

class GTTSNarrator:
    """
    Free text-to-speech using Google Text-to-Speech (gTTS).
    No API key required. Supports 40+ languages.
    """

    LANGUAGE_MAP = {
        "english": "en",
        "hindi"  : "hi",
        "telugu" : "te",
        "spanish": "es",
        "french" : "fr",
        "german" : "de",
        "arabic" : "ar",
    }

    def __init__(
        self,
        language: str = "english",
        output_dir: str = "./outputs/audio",
    ):
        self.lang       = self.LANGUAGE_MAP.get(language.lower(), "en")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def narrate(
        self,
        text      : str,
        filename  : Optional[str] = None,
        slow      : bool          = False,
    ) -> NarrationResult:
        """
        Convert text to speech and save as MP3.

        Args:
            text    : Text to convert (will be auto-cleaned)
            filename: Output filename (auto-generated if None)
            slow    : Slower speech rate for clarity

        Returns:
            NarrationResult with file path and metadata.
        """
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError("Install gTTS: pip install gTTS")

        start  = time.time()
        clean  = NarrationPreprocessor.clean(text)

        if not filename:
            filename = f"narration_{int(time.time())}.mp3"

        output_path = self.output_dir / filename

        tts = gTTS(text=clean, lang=self.lang, slow=slow)
        tts.save(str(output_path))

        elapsed   = time.time() - start
        file_size = output_path.stat().st_size / 1024

        logger.info(f"Audio saved: {output_path} ({file_size:.1f} KB, {elapsed:.2f}s)")

        return NarrationResult(
            text         = clean,
            audio_path   = str(output_path),
            provider     = VoiceProvider.GTTS,
            duration_secs= elapsed,
            file_size_kb = round(file_size, 2),
            language     = self.lang,
        )

    def narrate_to_bytes(self, text: str, slow: bool = False) -> bytes:
        """Return raw MP3 bytes (for API streaming without file I/O)."""
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError("Install gTTS: pip install gTTS")

        clean  = NarrationPreprocessor.clean(text)
        tts    = gTTS(text=clean, lang=self.lang, slow=slow)
        buf    = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()


# ─── ElevenLabs Voice Engine (Premium) ───────────────────────────────────────

class ElevenLabsNarrator:
    """
    Premium text-to-speech using ElevenLabs API.
    Ultra-realistic voices — requires ELEVENLABS_API_KEY.
    """

    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel — clear, professional

    def __init__(
        self,
        voice_id  : str = DEFAULT_VOICE_ID,
        output_dir: str = "./outputs/audio",
    ):
        self.api_key    = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Set ELEVENLABS_API_KEY environment variable.")

        self.voice_id   = voice_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def narrate(self, text: str, filename: Optional[str] = None) -> NarrationResult:
        """Send text to ElevenLabs API and save the resulting MP3."""
        import requests

        start = time.time()
        clean = NarrationPreprocessor.clean(text)

        url  = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "Accept"      : "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key"  : self.api_key,
        }
        payload = {
            "text"           : clean,
            "model_id"       : "eleven_monolingual_v1",
            "voice_settings" : {"stability": 0.5, "similarity_boost": 0.75},
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        if not filename:
            filename = f"el_narration_{int(time.time())}.mp3"

        output_path = self.output_dir / filename
        with open(output_path, "wb") as f:
            f.write(response.content)

        elapsed   = time.time() - start
        file_size = output_path.stat().st_size / 1024

        return NarrationResult(
            text         = clean,
            audio_path   = str(output_path),
            provider     = VoiceProvider.ELEVENLABS,
            duration_secs= elapsed,
            file_size_kb = round(file_size, 2),
            language     = "en",
        )


# ─── Unified Narrator (auto-selects provider) ─────────────────────────────────

class OmniNarrator:
    """
    Smart narrator that selects the best available TTS provider automatically.
    Falls back from ElevenLabs → gTTS gracefully.
    """

    def __init__(self, preferred: VoiceProvider = VoiceProvider.GTTS, **kwargs):
        self._narrator = None
        self.provider  = preferred

        if preferred == VoiceProvider.ELEVENLABS and os.getenv("ELEVENLABS_API_KEY"):
            try:
                self._narrator = ElevenLabsNarrator(**kwargs)
                logger.info("Using ElevenLabs narrator.")
            except Exception as e:
                logger.warning(f"ElevenLabs init failed: {e}. Falling back to gTTS.")

        if self._narrator is None:
            self._narrator = GTTSNarrator(**kwargs)
            self.provider  = VoiceProvider.GTTS
            logger.info("Using gTTS narrator.")

    def narrate(self, text: str, mode: str = "describe", **kwargs) -> NarrationResult:
        """Narrate analysis with optional intro prefix."""
        enriched = NarrationPreprocessor.add_intro(text, mode)
        return self._narrator.narrate(enriched, **kwargs)

    def narrate_to_bytes(self, text: str) -> bytes:
        """Return raw audio bytes for API streaming."""
        if hasattr(self._narrator, "narrate_to_bytes"):
            return self._narrator.narrate_to_bytes(text)
        result = self._narrator.narrate(text)
        with open(result.audio_path, "rb") as f:
            return f.read()
