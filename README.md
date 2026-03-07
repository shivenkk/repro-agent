# ReproAgent

An AI-powered system that analyzes ML paper reproducibility. Give it any arXiv URL — it extracts the full methodology with confidence scores, generates a runnable training script, and produces a reproducibility report.

## What It Does

1. **Parses** any arXiv paper (downloads PDF, extracts structured text)
2. **Extracts** methodology — model architecture, hyperparameters, dataset, training procedure — each with a confidence score showing how explicitly the paper states it
3. **Generates** a complete, self-contained PyTorch training script that attempts to reproduce the core experiment
4. **Scores** reproducibility (0–100) based on how completely the paper documents its methodology

## Why This Matters

ML papers are notoriously hard to reproduce. Missing hyperparameters, unclear methodology, no code provided. ReproAgent automates the tedious first step: figuring out exactly what the paper did and what it left out.

## Tested On

| Paper | Score | Key Finding |
|-------|-------|-------------|
| Deep Residual Learning (ResNet) | 94.3 | All hyperparameters explicitly stated |
| Attention Is All You Need (Transformer) | 89.1 | Complex lr schedule correctly extracted |
| BERT | 77.1 | Fine-tuning params found, optimizer details inferred |

## Architecture

```
arXiv URL
    │
    ▼
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐
│  Parser  │ ──▶ │ Extractor │ ──▶ │ CodeGen  │ ──▶ │  Scoring  │
│          │     │           │     │          │     │           │
│ PDF text │     │ Method +  │     │ PyTorch  │     │ Score +   │
│ sections │     │ confidence│     │ script   │     │ report    │
└──────────┘     └───────────┘     └──────────┘     └───────────┘
                      │
                 Llama 3.3 70B
                  (via Groq)
```

## Tech Stack

- **Backend**: Python, FastAPI, Pydantic
- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **LLM**: Llama 3.3 70B via Groq (free tier)
- **PDF Parsing**: PyMuPDF
- **Paper Metadata**: arXiv API

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Groq API key ([console.groq.com](https://console.groq.com))

### Backend

```bash
cd repro-agent
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn groq arxiv pymupdf python-dotenv pydantic httpx

echo 'GROQ_API_KEY=your_key_here' > .env
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 -m backend.main
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), paste an arXiv URL, and hit Analyze.

### CLI Usage

```bash
python3 -m backend.orchestrator https://arxiv.org/abs/1512.03385
```

## How Scoring Works

| Component | Weight | Criteria |
|-----------|--------|----------|
| Hyperparameter completeness | 40% | % of params with confidence ≥ 0.7 |
| Dataset identified | 15% | Named and recognized as standard |
| Model identified | 15% | Architecture described with high confidence |
| Code generated | 20% | Valid training script produced |
| Claimed results found | 10% | Paper reports quantitative results |

## Project Structure

```
repro-agent/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── orchestrator.py      # Pipeline coordinator
│   ├── agents/
│   │   ├── parser.py        # arXiv PDF → structured text
│   │   ├── extractor.py     # Text → methodology with confidence
│   │   └── codegen.py       # Methodology → PyTorch script
│   ├── services/
│   │   └── llm.py           # Groq/Llama API wrapper
│   └── models/
│       └── schemas.py       # Pydantic data models
├── frontend/
│   └── app/
│       └── page.tsx         # React UI
├── requirements.txt
└── README.md
```

## Key Design Decisions

**Confidence scoring on every extracted field.** Instead of just outputting "lr=0.001", the system reports "lr=0.001, confidence: 1.0, found in Section 4.1." This surfaces what the paper actually says vs. what's inferred.

**Smart text chunking.** ML papers are long but hyperparameters are concentrated in specific sections. The extractor scores text chunks by keyword density (learning rate, batch size, optimizer, etc.) and prioritizes the most information-dense sections, staying within LLM token limits.

**Deterministic code generation.** Generated scripts use fixed seeds, config dicts for all hyperparameters, and standardized output formats so results are comparable across runs.

## Limitations

- Text-only extraction (no figure/table parsing yet)
- Code generation targets PyTorch; papers using TensorFlow/JAX may get approximate translations
- Groq free tier has rate limits (30 RPM, 100k tokens/day)
- Papers with methodology split across main text and appendix may miss details in the appendix