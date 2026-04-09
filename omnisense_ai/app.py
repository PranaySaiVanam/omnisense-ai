"""
OmniSense AI — Interactive Streamlit Demo
==========================================
A live web UI for the multimodal AI pipeline.
Upload an image, type a question, choose analysis mode → get AI analysis + audio.

Run:
  streamlit run app.py

Author : Pranay Sai Vanam
"""

import os
import tempfile
import time

import streamlit as st

from src.multimodal_engine import AnalysisMode, LLMProvider, MultimodalEngine
from src.voice_narrator import OmniNarrator, VoiceProvider

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "OmniSense AI",
    page_icon  = "🧠",
    layout     = "wide",
)

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
# 🧠 OmniSense AI
### Multimodal Intelligence — Text × Image × Voice
""")
st.markdown("---")

# ─── Sidebar Configuration ────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["anthropic", "openai"],
        index=0,
        help="Claude Vision (Anthropic) or GPT-4o (OpenAI)",
    )

    mode = st.selectbox(
        "Analysis Mode",
        options=[m.value for m in AnalysisMode],
        index=0,
        help="Select what kind of analysis to perform",
    )

    language = st.selectbox(
        "Voice Language",
        options=["english", "hindi", "telugu", "spanish", "french"],
        index=0,
    )

    enable_voice = st.toggle("🔊 Enable Voice Narration", value=True)
    slow_speech  = st.toggle("🐢 Slow Speech", value=False)

    st.markdown("---")
    st.markdown("**Built by** Pranay Sai Vanam")
    st.markdown("[GitHub](https://github.com/PranaySaiVanam)")

# ─── Main UI ─────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Input")

    uploaded_image = st.file_uploader(
        "Upload Image (optional)",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        help="The AI will analyse this image together with your question",
    )

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

    text_input = st.text_area(
        "Your Question / Context",
        placeholder="e.g. What is happening in this image? / Describe the chart data / Is this document an invoice?",
        height=130,
    )

    extra_context = st.text_input(
        "Extra Context (optional)",
        placeholder="e.g. This is a medical image / This is from a financial report",
    )

    run_btn = st.button("🚀 Analyse", type="primary", use_container_width=True)

# ─── Analysis ─────────────────────────────────────────────────────────────────

with col2:
    st.subheader("📊 Results")

    if run_btn:
        if not text_input.strip():
            st.warning("Please enter a question or context.")
            st.stop()

        # Save uploaded image to temp file
        image_path = None
        if uploaded_image:
            suffix = "." + uploaded_image.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_image.read())
                image_path = tmp.name

        # Run analysis
        with st.spinner("🧠 Analysing with AI..."):
            try:
                engine = MultimodalEngine(
                    provider    = LLMProvider(provider),
                    temperature = 0.3,
                )
                t0     = time.time()
                result = engine.analyse(
                    text_input    = text_input,
                    image_path    = image_path,
                    mode          = AnalysisMode(mode),
                    extra_context = extra_context,
                )
                elapsed = time.time() - t0

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # Results display
        confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(result.confidence, "⚪")

        col_m, col_c, col_t, col_l = st.columns(4)
        col_m.metric("Mode",       result.mode.capitalize())
        col_c.metric("Confidence", f"{confidence_color} {result.confidence.capitalize()}")
        col_t.metric("Tokens",     result.tokens_used)
        col_l.metric("Latency",    f"{result.latency_ms:.0f} ms")

        st.markdown("### Analysis")
        st.markdown(result.analysis)

        # Voice narration
        if enable_voice:
            with st.spinner("🔊 Generating voice narration..."):
                try:
                    narrator = OmniNarrator(
                        preferred  = VoiceProvider.GTTS,
                        language   = language,
                        output_dir = "./outputs/audio",
                    )
                    audio_bytes = narrator.narrate_to_bytes(result.analysis)
                    st.audio(audio_bytes, format="audio/mp3")
                    st.caption("🎙️ AI Voice Narration")
                except Exception as e:
                    st.warning(f"Voice narration unavailable: {e}")

        # Clean up temp file
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

    else:
        st.info("Upload an image and/or type a question, then click Analyse.")

        st.markdown("#### 🔍 What OmniSense AI can do:")
        features = [
            ("🖼️ **Describe**",       "Explain any image in rich detail"),
            ("✅ **Compare**",        "Check if an image matches a text claim"),
            ("📊 **Extract**",        "Pull structured data from charts/docs"),
            ("💬 **Sentiment**",      "Detect emotion and tone in visual content"),
            ("♿ **Accessibility**",  "Generate WCAG alt-text for any image"),
            ("📝 **Summarize**",      "Condense documents and presentations"),
        ]
        for icon, desc in features:
            st.markdown(f"- {icon}: {desc}")
