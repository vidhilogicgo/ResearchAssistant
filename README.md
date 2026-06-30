# AI Multi-Agent Research Assistant

An end-to-end AI research automation system built with LangGraph, LangChain, Groq Llama models, DuckDuckGo Search, document loaders, HuggingFace Embeddings, ChromaDB, MongoDB Atlas, Streamlit, and FastAPI.

The app accepts a user query, creates a research plan, gathers web/document evidence, performs RAG retrieval, generates a compact final answer, evaluates quality, and optionally stores the session in MongoDB Atlas.

## Features

- Multi-agent workflow using LangGraph
- Planner, Researcher, Writer, and Evaluator agents
- Groq API with Llama 3.3 model support
- DuckDuckGo web research
- PDF, TXT, Markdown, DOCX, and HTML document loading
- RAG pipeline using HuggingFace Embeddings and ChromaDB
- Optional MongoDB Atlas session storage
- Streamlit user interface
- FastAPI backend endpoint
- Render deployment configuration

## Architecture

```text
User Query
  -> Streamlit UI / FastAPI API
  -> LangGraph Workflow
  -> Planner Agent
  -> Researcher Agent
  -> DuckDuckGo Search + Document Processing + RAG
  -> ChromaDB Retrieval
  -> Writer Agent
  -> Evaluator Agent
  -> Final Answer
```

The implementation is intentionally compressed into `app.py`. Supporting files are included only for setup, deployment, documentation, and GitHub readiness.

## Project Structure

```text
.
+-- app.py                  # Streamlit UI, FastAPI API, LangGraph workflow, agents, RAG, storage
+-- requirements.txt        # Python dependencies
+-- .env.example            # Example environment variables
+-- render.yaml             # Render deployment config for FastAPI
+-- CODE_DOCUMENTATION.md   # Function-by-function code documentation
+-- CONTRIBUTING.md         # Contribution guidelines
+-- LICENSE                 # MIT license
+-- .gitignore              # Files excluded from GitHub
+-- README.md               # Project overview and setup
```

## Requirements

- Python 3.11 recommended
- Groq API key
- Optional MongoDB Atlas connection string

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
copy .env.example .env
```

Edit `.env` and set at least:

```env
GROQ_API_KEY=your_groq_api_key
```

MongoDB Atlas is optional. If `MONGODB_URI` is not set, the app still runs but skips database persistence.

## Run Streamlit UI

```bash
streamlit run app.py
```

## Run FastAPI Backend

```bash
uvicorn app:api_app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Research request:

```bash
curl -X POST http://localhost:8000/research ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What are AI agents?\",\"use_web\":true,\"use_rag\":false,\"max_results\":5}"
```

## Final Answer Format

The Writer Agent is designed to produce a short final answer, not a long essay or formal report.

Example:

```markdown
**Final Answer**
- AI agents use an LLM to reason, choose actions, and complete tasks.
- They can call tools such as search engines, APIs, databases, and retrievers.
- LangGraph coordinates the workflow between multiple agents.
- ChromaDB stores document embeddings for semantic retrieval.
- The Evaluator Agent checks relevance, clarity, and completeness.
```

## Deployment

### Streamlit Community Cloud

Use:

```bash
streamlit run app.py
```

Add these secrets in Streamlit Cloud:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

### Render FastAPI

The included `render.yaml` can be used as a starting point.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:api_app --host 0.0.0.0 --port $PORT
```

Add `GROQ_API_KEY` as a Render environment variable.

## Documentation

See `CODE_DOCUMENTATION.md` for a function-by-function explanation of how `app.py` produces the final answer.

## GitHub Push Commands

Run these commands yourself from the project folder:

```bash
git init
git add .
git commit -m "Initial commit: AI multi-agent research assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If the repository already has a remote, use:

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## Important Before Pushing

- Do not commit `.env`.
- Do not commit `.venv/`.
- Do not commit `chroma_db/`.
- Keep real API keys only in local environment variables or deployment secrets.

## License

This project is licensed under the MIT License.
