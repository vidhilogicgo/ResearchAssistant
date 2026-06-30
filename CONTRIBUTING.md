# Contributing

Thanks for improving this project.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

## Development Checks

Before opening a pull request, run:

```bash
python -m py_compile app.py
```

If dependencies are installed, also run the app locally:

```bash
streamlit run app.py
```

## Guidelines

- Keep the main implementation in `app.py` unless a new file is clearly necessary.
- Do not commit `.env`, API keys, ChromaDB data, caches, or virtual environments.
- Keep prompts concise and aligned with the final-answer-only output style.
- Update `CODE_DOCUMENTATION.md` when changing workflow logic.
