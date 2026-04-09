"""
OmniSense AI — Test Suite
==========================
Unit and integration tests for the multimodal pipeline.
Run: pytest tests/ -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from src.multimodal_engine import (
    AnalysisMode, AnalysisResult, LLMProvider,
    MultimodalEngine, NarrationPreprocessor, encode_image_base64,
)
from src.voice_narrator import NarrationPreprocessor as VoicePreprocessor


# ─── Preprocessor Tests ───────────────────────────────────────────────────────

class TestNarrationPreprocessor:

    def test_strips_markdown_bold(self):
        result = VoicePreprocessor.clean("This is **bold** text")
        assert "**" not in result
        assert "bold" in result

    def test_strips_headers(self):
        result = VoicePreprocessor.clean("## My Header\nSome content")
        assert "##" not in result
        assert "My Header" in result

    def test_collapses_whitespace(self):
        result = VoicePreprocessor.clean("Hello   world")
        assert "  " not in result

    def test_trims_long_text(self):
        long_text = "A" * 6000
        result    = VoicePreprocessor.clean(long_text)
        assert len(result) <= VoicePreprocessor.MAX_TTS_CHARS + 60

    def test_adds_mode_intro(self):
        result = VoicePreprocessor.add_intro("Analysis here.", "describe")
        assert "visual analysis" in result.lower()


# ─── Confidence Heuristic Tests ───────────────────────────────────────────────

class TestConfidenceEstimation:

    def test_high_confidence(self):
        text = "The image clearly shows a bar chart."
        assert MultimodalEngine._estimate_confidence(text) == "high"

    def test_low_confidence(self):
        text = "It is unclear what the image might show."
        assert MultimodalEngine._estimate_confidence(text) == "low"

    def test_medium_confidence(self):
        text = "The image shows a person in a park."
        assert MultimodalEngine._estimate_confidence(text) == "medium"


# ─── Image Encoding Tests ─────────────────────────────────────────────────────

class TestImageEncoding:

    def test_encodes_png(self, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        b64, media_type = encode_image_base64(str(img_file))
        assert media_type == "image/png"
        assert len(b64) > 0

    def test_encodes_jpg(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        b64, media_type = encode_image_base64(str(img_file))
        assert media_type == "image/jpeg"

    def test_unknown_extension_defaults_jpeg(self, tmp_path):
        img_file = tmp_path / "test.bmp"
        img_file.write_bytes(b"\x00" * 100)
        _, media_type = encode_image_base64(str(img_file))
        assert media_type == "image/jpeg"


# ─── Engine Mock Tests ────────────────────────────────────────────────────────

class TestMultimodalEngine:

    @patch("src.multimodal_engine.anthropic.Anthropic")
    def test_engine_initializes_anthropic(self, mock_client):
        engine = MultimodalEngine(provider=LLMProvider.ANTHROPIC)
        assert engine.provider == LLMProvider.ANTHROPIC
        assert "claude" in engine.model.lower()

    @patch("src.multimodal_engine.openai.OpenAI")
    def test_engine_initializes_openai(self, mock_client):
        engine = MultimodalEngine(provider=LLMProvider.OPENAI)
        assert engine.provider == LLMProvider.OPENAI
        assert "gpt" in engine.model.lower()

    def test_all_analysis_modes_have_prompts(self):
        from src.multimodal_engine import SYSTEM_PROMPTS
        for mode in AnalysisMode:
            assert mode in SYSTEM_PROMPTS
            assert len(SYSTEM_PROMPTS[mode]) > 20


# ─── API Tests ────────────────────────────────────────────────────────────────

class TestAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_modes_endpoint(self, client):
        resp = client.get("/modes")
        assert resp.status_code == 200
        modes = resp.json()
        assert len(modes) == len(AnalysisMode)
        mode_names = [m["mode"] for m in modes]
        assert "describe" in mode_names
        assert "extract"  in mode_names
