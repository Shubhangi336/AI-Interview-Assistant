# 🎯 Interview Trainer Agent

> **AI-powered interview preparation platform built with IBM Granite (watsonx.ai) and Retrieval-Augmented Generation (RAG)**

[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-0f62fe?style=flat-square)](https://www.ibm.com/watsonx)
[![IBM Granite](https://img.shields.io/badge/IBM-Granite--13b--chat--v2-8a3ffc?style=flat-square)](https://www.ibm.com/granite)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange?style=flat-square)](https://langchain.com)

---

## 📋 Overview

The **Interview Trainer Agent** prepares job seekers for interviews by generating tailored question sets and preparation strategies based on their profile, experience level, and target role.

### Core Features

| Feature | Description |
|---|---|
| **Question Set Generation** | Technical, behavioral (STAR), and HR questions tailored to role + experience level with category filters |
| **Interactive Mock Simulator** | Timed simulation studio with Speech-to-Text (voice answers) & instant AI evaluation scorecards (0-10 score, strengths, improvements) |
| **Model Answer Generator** | Expert model answers with key points, mistakes to avoid, and Text-to-Speech (TTS) audio narration |
| **STAR Story Builder** | Structured 4-phase wizard (Situation, Task, Action, Result) with AI critique, action verbs, and metric suggestions |
| **Resume & Gap Analyzer** | Direct file upload (.pdf, .docx, .txt) or paste; yields strengths, skill gaps, predicted questions, and 30-day roadmap |
| **Saved Study Deck** | Bookmark tricky questions to your personal study deck for last-minute flashcard review and export |
| **Conversational Chat Coach** | Free-form coaching with IBM Granite, voice input, persistent history, and one-click transcript export |
| **RAG Knowledge Base & Search** | Upload custom guides (.pdf, .txt, .json), view chunk breakdown, and test semantic keyword retrieval live |
| **Modern Sleek UI & Dark Mode** | Clean IBM Carbon-inspired theme with light/dark toggle, responsive mobile drawer, and status telemetry |
| **Smart Fallback Engine** | Operates in Live IBM Granite mode or intelligent offline RAG simulation mode out of the box |

### Technology Stack

| Layer | Technology |
|---|---|
| **LLM** | IBM Granite 13B Chat v2 via **IBM watsonx.ai** |
| **RAG** | LangChain + ChromaDB (vector store) + Sentence Transformers (embeddings) |
| **Backend** | Python 3.11+ · FastAPI · Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS · Responsive forest-green and warm-beige interface |
| **Document Loaders** | PyPDF · python-docx · JSON structured ingestion |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Interview Trainer Agent                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Frontend (static/index.html)                          │
│      │                                                  │
│      ▼                                                  │
│   FastAPI Backend (app/main.py + app/routes.py)         │
│      │                                                  │
│      ├─► Agent (app/agent.py)                           │
│      │        │                                         │
│      │        ├─► RAG Pipeline (app/rag.py)             │
│      │        │        │                                │
│      │        │        ├─► ChromaDB Vector Store        │
│      │        │        └─► HuggingFace Embeddings       │
│      │        │             (all-MiniLM-L6-v2)          │
│      │        │                                         │
│      │        └─► IBM Granite LLM (app/llm.py)          │
│      │                 └─► IBM watsonx.ai API            │
│      │                                                  │
│      └─► Pydantic Schemas (app/schemas.py)              │
│                                                         │
│   Knowledge Base (data/documents/)                      │
│      ├── interview_qa_seed.json                         │
│      ├── interview_preparation_guide.txt                │
│      └── [user-uploaded PDFs/TXTs/JSONs]                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- IBM Cloud account (free Lite tier works)
- IBM watsonx.ai project

### 2. Get IBM watsonx.ai Credentials

1. Sign up at [IBM Cloud](https://cloud.ibm.com) (Lite tier is free)
2. Create a **watsonx.ai** project at [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
3. Generate an **API key** at [cloud.ibm.com/iam/apikeys](https://cloud.ibm.com/iam/apikeys)
4. Copy your **Project ID** from the watsonx.ai project settings

### 3. Installation

```bash
# Clone the repository
git clone <repo-url>
cd Interview-Trainer-Agent

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
```

Your `.env` file:
```env
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-13b-chat-v2
```

Watsonx credentials are read by the backend from environment variables or `.env`; the browser does not accept or receive them. When credentials are unavailable, generation uses the built-in simulation fallback. Set `USE_SIMULATION_MODE=true` to force simulation mode.

### 5. Run the Application

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **http://localhost:8000** 🎉

---

## 📁 Project Structure

```
Interview-Trainer-Agent/
├── app/
│   ├── __init__.py         # Package init
│   ├── main.py             # FastAPI app + lifespan startup
│   ├── routes.py           # All API endpoints
│   ├── agent.py            # Core agent logic (orchestration)
│   ├── llm.py              # IBM Granite / watsonx.ai wrapper
│   ├── rag.py              # RAG pipeline (ingestion + retrieval)
│   ├── prompts.py          # Prompt templates for IBM Granite
│   ├── schemas.py          # Pydantic request/response models
│   └── config.py           # Configuration from environment
│
├── static/
│   └── index.html          # Full-featured frontend SPA
│
├── data/
│   ├── documents/          # Seed + uploaded documents
│   │   ├── interview_qa_seed.json
│   │   └── interview_preparation_guide.txt
│   └── vector_store/       # ChromaDB persistent vector store (auto-created)
│
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---


## 📚 Knowledge Base

The RAG pipeline uses two seed documents:

1. **`interview_qa_seed.json`** — 12+ structured Q&A entries covering:
   - Software Engineer (technical + behavioral)
   - Data Scientist (ML concepts)
   - Product Manager (prioritization, metrics)
   - DevOps Engineer (IaC)
   - Frontend / Backend Developer
   - General HR questions (tell me about yourself, weaknesses, 5-year plan)

2. **`interview_preparation_guide.txt`** — Comprehensive guide covering:
   - STAR method framework
   - Experience level expectations (Fresher → Lead)
   - Interview formats (phone screen, coding, system design, behavioral)
   - Salary negotiation scripts
   - Questions to ask the interviewer
   - Soft skills assessment criteria

### Adding Custom Content

Upload additional documents via the **Knowledge Base** tab in the UI or via the API:

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@your_interview_guide.pdf"

# Rebuild the index after bulk uploads
curl -X POST http://localhost:8000/api/v1/ingest/rebuild
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `WATSONX_API_KEY` | — | **Required** · IBM Cloud API key |
| `WATSONX_PROJECT_ID` | — | **Required** · watsonx.ai project ID |
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` | watsonx.ai regional endpoint |
| `GRANITE_MODEL_ID` | `ibm/granite-13b-chat-v2` | IBM Granite model to use |
| `APP_HOST` | `0.0.0.0` | Server bind host |
| `APP_PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug logging + hot reload |
| `VECTOR_STORE_PATH` | `./data/vector_store` | ChromaDB persistence path |
| `DOCUMENTS_PATH` | `./data/documents` | Source documents directory |

---

## 🧠 IBM Granite Models Available on Lite Tier

| Model ID | Description |
|---|---|
| `ibm/granite-13b-chat-v2` | General-purpose chat (recommended) |
| `ibm/granite-13b-instruct-v2` | Instruction-following |
| `ibm/granite-3-8b-instruct` | Faster, lighter model |
| `ibm/granite-20b-multilingual` | Multilingual support |

Update `GRANITE_MODEL_ID` in `.env` to switch models.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using IBM Granite · watsonx.ai · LangChain · FastAPI*
