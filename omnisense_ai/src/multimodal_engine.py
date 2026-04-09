"""
OmniSense AI — Multimodal Analysis Engine
==========================================
Processes text + image inputs simultaneously using LLMs with vision capability.
Supports Claude Vision, GPT-4 Vision, and text-only fallback modes.

Author  : Pranay Sai Vanam
Stack   : Python · Anthropic Claude API · OpenAI API · LangChain · Base64 Vision
"""

import base64
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import anthropic
import openai
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ─── Enums & Config ───────────────────────────────────────────────────────────

class AnalysisMode(str, Enum):
    DESCRIBE      = "describe"        # General image + text description
    COMPARE       = "compare"         # Compare image against text claim
    EXTRACT       = "extract"         # Extract structured data from image
    SENTIMENT     = "sentiment"       # Emotional/contextual tone analysis
    ACCESSIBILITY = "accessibility"   # Generate alt-text for accessibility
    SUMMARIZE     = "summarize"       # Summarise document/chart images


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI    = "openai"


@dataclass
class AnalysisResult:
    mode         : AnalysisMode
    provider     : LLMProvider
    text_input   : str
    image_path   : Optional[str]
    analysis     : str
    confidence   : str                  # "high" | "medium" | "low"
    tokens_used  : int
    latency_ms   : float
    metadata     : dict = field(default_factory=dict)


# ─── Prompt Library ───────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    AnalysisMode.DESCRIBE: """You are an expert multimodal AI analyst.
When given an image and optional text context, provide a rich, structured description.
Cover: main subjects, visual elements, inferred context, and notable details.
Be precise and professional.""",

    AnalysisMode.COMPARE: """You are a fact-checking AI assistant with vision capabilities.
Compare the provided image against the text claim.
Respond with: SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE
Then explain your reasoning with specific visual evidence.""",

    AnalysisMode.EXTRACT: """You are a data extraction AI.
Extract ALL structured information from the image (tables, charts, forms, text).
Return clean, well-formatted data. Use JSON where appropriate.""",

    AnalysisMode.SENTIMENT: """You are an emotional intelligence AI with visual understanding.
Analyse the mood, tone, and emotional context conveyed by both the image and text.
Identify: primary emotion, secondary cues, cultural context, and confidence level.""",

    AnalysisMode.ACCESSIBILITY: """You are an accessibility AI specialist.
Generate concise, descriptive alt-text for the image following WCAG 2.1 guidelines.
Include: subject, action, context, and any text visible in the image. Max 125 words.""",

    AnalysisMode.SUMMARIZE: """You are a document intelligence AI.
Summarise the key information from the image (document, chart, diagram, or screenshot).
Structure: Key Points → Details → Actionable Insights.""",
}

USER_PROMPT_TEMPLATE = """User context / question:
{text_input}

Please analyse the provided image in the context of the above."""


# ─── Image Utilities ──────────────────────────────────────────────────────────

def encode_image_base64(image_path: str) -> tuple[str, str]:
    """
    Encode an image file to base64 and detect its MIME type.
    Returns (base64_string, media_type).
    """
    path = Path(image_path)
    ext  = path.suffix.lower()

    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    media_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    logger.debug(f"Encoded image: {path.name} ({media_type})")
    return b64, media_type


# ─── Multimodal Engine ────────────────────────────────────────────────────────

class MultimodalEngine:
    """
    Unified multimodal analysis engine supporting Claude Vision and GPT-4 Vision.
    Handles text-only, image-only, and combined text+image inputs.
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.ANTHROPIC,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ):
        self.provider    = provider
        self.temperature = temperature
        self.max_tokens  = max_tokens

        if provider == LLMProvider.ANTHROPIC:
            self.model  = model or "claude-opus-4-5"
            self.client = anthropic.Anthropic()
        else:
            self.model  = model or "gpt-4o"
            self.client = openai.OpenAI()

        logger.info(f"MultimodalEngine initialized: {provider} / {self.model}")

    # ── Core Analysis ────────────────────────────────────────────────────────

    def analyse(
        self,
        text_input   : str,
        image_path   : Optional[str] = None,
        mode         : AnalysisMode  = AnalysisMode.DESCRIBE,
        extra_context: str           = "",
    ) -> AnalysisResult:
        """
        Run a multimodal analysis combining text and/or image.

        Args:
            text_input   : User's question or context text
            image_path   : Optional path to an image file
            mode         : Analysis mode (describe, compare, extract, etc.)
            extra_context: Additional context injected into the system prompt

        Returns:
            AnalysisResult with the generated analysis and metadata.
        """
        start = time.time()
        system = SYSTEM_PROMPTS[mode]
        if extra_context:
            system += f"\n\nAdditional context: {extra_context}"

        user_text = USER_PROMPT_TEMPLATE.format(text_input=text_input)

        if self.provider == LLMProvider.ANTHROPIC:
            result_text, tokens = self._call_claude(system, user_text, image_path)
        else:
            result_text, tokens = self._call_openai(system, user_text, image_path)

        latency = (time.time() - start) * 1000

        return AnalysisResult(
            mode       = mode,
            provider   = self.provider,
            text_input = text_input,
            image_path = image_path,
            analysis   = result_text,
            confidence = self._estimate_confidence(result_text),
            tokens_used= tokens,
            latency_ms = round(latency, 2),
            metadata   = {"model": self.model, "temperature": self.temperature},
        )

    # ── Provider: Anthropic Claude ───────────────────────────────────────────

    def _call_claude(
        self,
        system    : str,
        user_text : str,
        image_path: Optional[str],
    ) -> tuple[str, int]:
        """Send a vision-enabled request to Claude."""
        content: list = []

        if image_path:
            b64, media_type = encode_image_base64(image_path)
            content.append({
                "type"  : "image",
                "source": {
                    "type"      : "base64",
                    "media_type": media_type,
                    "data"      : b64,
                },
            })

        content.append({"type": "text", "text": user_text})

        response = self.client.messages.create(
            model     = self.model,
            max_tokens= self.max_tokens,
            system    = system,
            messages  = [{"role": "user", "content": content}],
        )

        text   = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens

    # ── Provider: OpenAI GPT-4 Vision ────────────────────────────────────────

    def _call_openai(
        self,
        system    : str,
        user_text : str,
        image_path: Optional[str],
    ) -> tuple[str, int]:
        """Send a vision-enabled request to GPT-4o."""
        user_content: list = []

        if image_path:
            b64, media_type = encode_image_base64(image_path)
            user_content.append({
                "type"     : "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })

        user_content.append({"type": "text", "text": user_text})

        response = self.client.chat.completions.create(
            model      = self.model,
            temperature= self.temperature,
            max_tokens = self.max_tokens,
            messages   = [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user_content},
            ],
        )

        text   = response.choices[0].message.content
        tokens = response.usage.total_tokens
        return text, tokens

    # ── Confidence Heuristic ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_confidence(text: str) -> str:
        """Simple heuristic to estimate output confidence from wording."""
        low_signals  = ["unclear", "uncertain", "possibly", "might", "hard to tell"]
        high_signals = ["clearly", "definitively", "evidently", "confirms", "shows"]
        text_lower   = text.lower()

        if any(s in text_lower for s in low_signals):
            return "low"
        if any(s in text_lower for s in high_signals):
            return "high"
        return "medium"

    # ── Batch Processing ─────────────────────────────────────────────────────

    def batch_analyse(
        self,
        requests: list[dict],
    ) -> list[AnalysisResult]:
        """
        Process multiple analysis requests sequentially.

        Each request dict: {"text_input": str, "image_path": str, "mode": AnalysisMode}
        """
        results = []
        for i, req in enumerate(requests):
            logger.info(f"Batch [{i+1}/{len(requests)}] mode={req.get('mode', 'describe')}")
            results.append(self.analyse(**req))
        return results


# ─── LangChain Wrapper ────────────────────────────────────────────────────────

class LangChainMultimodalChain:
    """
    LangChain-native multimodal chain using LCEL (LangChain Expression Language).
    Enables easy chaining with memory, tools, and RAG pipelines.
    """

    def __init__(self, model: str = "claude-opus-4-5", temperature: float = 0.3):
        self.llm = ChatAnthropic(model=model, temperature=temperature)

    def run(self, text: str, image_path: Optional[str] = None) -> str:
        """Run the chain with optional image attachment."""
        content: list = []

        if image_path:
            b64, media_type = encode_image_base64(image_path)
            content.append({
                "type"  : "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })

        content.append({"type": "text", "text": text})

        message = HumanMessage(content=content)
        response = self.llm.invoke([message])
        return response.content
