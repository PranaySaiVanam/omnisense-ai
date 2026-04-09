# 🧠 OmniSense AI — Multimodal Intelligence Platform

> **Text × Image × Voice** — A production-grade Generative AI system that sees, thinks, and speaks.

**Author:** Pranay Sai Vanam  
**Stack:** Python · Claude Vision API · GPT-4o · LangChain · FastAPI · Streamlit · gTTS  
**GitHub:** [github.com/PranaySaiVanam/omnisense-ai](https://github.com/PranaySaiVanam/omnisense-ai)

---

## What This Does

OmniSense AI is a **multimodal GenAI pipeline** that combines three modalities in a single unified system:

1. **👁️ Vision** — Understands images using Claude Vision / GPT-4o
2. **🧠 Language** — Answers questions, extracts data, compares claims using LLMs
3. **🔊 Voice** — Narrates analysis results as natural-sounding MP3 audio

Real use cases: document intelligence, accessibility tooling, visual QA systems, content moderation, chart analysis, medical image description.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   OmniSense AI                       │
│                                                      │
│  ┌──────────┐   ┌──────────────────┐   ┌─────────┐  │
│  │  Image   │──▶│  Multimodal      │──▶│  Voice  │  │
│  │  Upload  │   │  Engine          │   │ Narrator│  │
│  └──────────┘   │                  │   └─────────┘  │
│                 │  Claude Vision   │        │        │
│  ┌──────────┐──▶│  GPT-4o Vision   │        │        │
│  │  Text    │   │  LangChain LCEL  │        ▼        │
│  │  Input   │   └──────────────────┘   MP3 Audio    │
│  └──────────┘            │                           │
│                          ▼                           │
│                   JSON Analysis                      │
│               (mode · confidence ·                   │
│                tokens · latency)                     │
└─────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| 6 Analysis Modes | Describe · Compare · Extract · Sentiment · Accessibility · Summarize |
| Dual LLM Support | Claude Opus (Anthropic) or GPT-4o (OpenAI) — swap with one env var |
| Voice Output | gTTS (free) or ElevenLabs (premium) — auto-selected |
| Multi-language | English, Hindi, Telugu, Spanish, French, German, Arabic |
| LangChain LCEL | Fully chainable with RAG, memory, and tool-use pipelines |
| FastAPI Backend | Async, production-ready, 4-worker deployment |
| Streamlit UI | Live demo app for non-technical stakeholders |
| Batch API | Process up to 10 requests in a single call |
| Test Suite | pytest with mocked LLM calls for CI/CD |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/PranaySaiVanam/omnisense-ai
cd omnisense-ai
pip install -r requirements.txt

# 2. Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"        # optional
export LLM_PROVIDER="anthropic"          # or "openai"

# 3a. Start the API
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# 3b. OR launch the Streamlit demo
streamlit run app.py
```

---

## API Examples

### Analyse an image
```bash
curl -X POST http://localhost:8000/analyse \
  -F "text_input=What does this chart show?" \
  -F "mode=summarize" \
  -F "image=@chart.png"
```

### Get voice narration
```bash
curl -X POST http://localhost:8000/analyse/voice \
  -F "text_input=Describe this image" \
  -F "mode=describe" \
  -F "language=english" \
  -F "image=@photo.jpg" \
  --output analysis.mp3
```

### Batch (text-only)
```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"text_input": "Explain RAG", "mode": "describe"},
    {"text_input": "What is LangChain?", "mode": "summarize"}
  ]'
```

---

## Analysis Modes

| Mode | What it does |
|------|-------------|
| `describe` | Rich description of image + text context |
| `compare` | SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE verdict |
| `extract` | Pulls tables, numbers, text from images as structured data |
| `sentiment` | Emotional + tonal analysis of visual + textual content |
| `accessibility` | WCAG 2.1 alt-text generation |
| `summarize` | Condenses documents, slides, and charts |

---

## Project Structure

```
omnisense-ai/
├── src/
│   ├── multimodal_engine.py   # Core vision + LLM pipeline
│   └── voice_narrator.py      # gTTS + ElevenLabs TTS pipeline
├── tests/
│   └── test_omnisense.py      # pytest test suite
├── outputs/
│   └── audio/                 # Generated MP3 files
├── api.py                     # FastAPI production server
├── app.py                     # Streamlit demo UI
├── requirements.txt
└── README.md
```

---

## Skills Demonstrated

- **Multimodal LLM integration** (Claude Vision, GPT-4o) with base64 image encoding
- **Prompt engineering** — 6 distinct system prompts tuned for specific analysis tasks
- **LangChain LCEL** — composable chain for RAG and tool-use extension
- **Text-to-Speech pipeline** — gTTS + ElevenLabs with auto-fallback
- **FastAPI** — async endpoints, file upload, streaming audio response
- **Streamlit** — production demo UI for stakeholder presentations
- **Clean Python** — dataclasses, enums, type hints, logging throughout
- **Test-driven** — pytest with mocked providers, fixture-based API tests

---

## Roadmap

- [ ] RAG integration — answer questions grounded in a document corpus
- [ ] Video frame analysis — extract keyframes and analyse sequences  
- [ ] Real-time streaming — stream LLM tokens to frontend via WebSocket
- [ ] Fine-tuning pipeline — build domain-specific vision model adapters
- [ ] Docker deployment — containerised with nginx reverse proxy

---

*Built to demonstrate production-grade GenAI engineering. Open to full-time, contract & remote opportunities.*
