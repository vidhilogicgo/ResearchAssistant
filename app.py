import asyncio
import json
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from pymongo import MongoClient

load_dotenv()

APP_TITLE = "AI Multi-Agent Research Assistant"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class ResearchState(TypedDict, total=False):
    session_id: str
    query: str
    documents: List[Document]
    plan: str
    search_results: List[str]
    rag_context: str
    insights: str
    report: str
    evaluation: str
    progress: List[str]
    errors: List[str]
    use_web: bool
    use_rag: bool
    max_results: int


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    use_web: bool = True
    use_rag: bool = True
    max_results: int = Field(default=5, ge=1, le=10)


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    plan: str
    insights: str
    report: str
    evaluation: str
    progress: List[str]


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.2) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your environment or .env file.")
    return ChatGroq(model=model, temperature=temperature, groq_api_key=os.getenv("GROQ_API_KEY"))


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_mongo_collection():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None
    client = MongoClient(uri)
    db = client[os.getenv("MONGODB_DB", "research_assistant")]
    return db[os.getenv("MONGODB_COLLECTION", "sessions")]


def save_session(state: ResearchState) -> None:
    collection = get_mongo_collection()
    if collection is None:
        return
    payload = {
        "session_id": state["session_id"],
        "query": state.get("query", ""),
        "plan": state.get("plan", ""),
        "insights": state.get("insights", ""),
        "report": state.get("report", ""),
        "evaluation": state.get("evaluation", ""),
        "progress": state.get("progress", []),
        "errors": state.get("errors", []),
        "updated_at": datetime.now(timezone.utc),
    }
    collection.update_one({"session_id": state["session_id"]}, {"$set": payload}, upsert=True)


def append_progress(state: ResearchState, message: str) -> ResearchState:
    state.setdefault("progress", []).append(message)
    return state


def load_uploaded_documents(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    if not uploaded_files:
        return docs

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix in {".txt", ".md"}:
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                loader = UnstructuredFileLoader(tmp_path)
            docs.extend(loader.load())
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return docs


def build_vectorstore(session_id: str, documents: List[Document]) -> Optional[Chroma]:
    if not documents:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=180)
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["session_id"] = session_id
        chunk.metadata["chunk_id"] = i
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(Path(CHROMA_DIR) / session_id),
        collection_name=f"research_{session_id.replace('-', '_')}",
    )


def retrieve_context(session_id: str, query: str, documents: List[Document], use_rag: bool) -> str:
    if not use_rag or not documents:
        return ""
    vectorstore = build_vectorstore(session_id, documents)
    if vectorstore is None:
        return ""
    item_count = vectorstore._collection.count()
    if item_count == 0:
        return ""
    matches = vectorstore.similarity_search(query, k=min(5, item_count))
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'uploaded document')}\n{doc.page_content}"
        for doc in matches
    )


def planner_agent(state: ResearchState) -> ResearchState:
    append_progress(state, "Planner Agent: creating research strategy.")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior research planner. Create a clear research plan with objectives, "
                "sub-questions, search strategy, evidence criteria, and answer structure.",
            ),
            ("human", "Research query: {query}"),
        ]
    )
    state["plan"] = (prompt | get_llm()).invoke({"query": state["query"]}).content
    return state


def researcher_agent(state: ResearchState) -> ResearchState:
    append_progress(state, "Researcher Agent: gathering web and document evidence.")
    max_results = int(state.get("max_results", 5))
    use_web = bool(state.get("use_web", True))
    use_rag = bool(state.get("use_rag", True))

    search_results: List[str] = []
    if use_web:
        try:
            search = DuckDuckGoSearchResults(max_results=max_results, output_format="list")
            raw_results = search.invoke(state["query"])
            search_results = [
                f"{item.get('title', 'Untitled')}\n{item.get('snippet', '')}\nURL: {item.get('link', '')}"
                for item in raw_results
            ]
        except Exception as exc:
            state.setdefault("errors", []).append(f"DuckDuckGo search failed: {exc}")

    rag_context = retrieve_context(
        state["session_id"],
        state["query"],
        state.get("documents", []),
        use_rag,
    )
    evidence = "\n\n".join(search_results) or "No web search results collected."
    context = rag_context or "No uploaded document context available."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research analyst. Extract only query-relevant facts, themes, "
                "contradictions, gaps, and source-grounded insights. Avoid filler.",
            ),
            (
                "human",
                "Query:\n{query}\n\nPlan:\n{plan}\n\nWeb evidence:\n{evidence}\n\nDocument/RAG context:\n{context}",
            ),
        ]
    )
    state["search_results"] = search_results
    state["rag_context"] = rag_context
    state["insights"] = (prompt | get_llm(temperature=0.1)).invoke(
        {
            "query": state["query"],
            "plan": state["plan"],
            "evidence": evidence,
            "context": context,
        }
    ).content
    return state


def writer_agent(state: ResearchState) -> ResearchState:
    append_progress(state, "Writer Agent: drafting concise bullet response.")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write only the direct response. Do not write a title, heading, executive summary, "
                "methodology, conclusion, or generic filler. The answer may be a short paragraph, "
                "a compact bullet list, or a mix of both depending on what best fits the query. "
                "Keep it concise and include only relevant information. Add source URLs inline "
                "only when they directly support a specific point. Avoid unnecessary pointwise "
                "formatting when a natural paragraph is clearer.",
            ),
            (
                "human",
                "Query:\n{query}\n\nPlan:\n{plan}\n\nInsights:\n{insights}\n\nSources:\n{sources}\n\nAdditional writing instruction:\n{rewrite_instruction}",
            ),
        ]
    )
    state["report"] = (prompt | get_llm(temperature=float(state.get("writer_temperature", 0.2)))).invoke(
        {
            "query": state["query"],
            "plan": state["plan"],
            "insights": state["insights"],
            "sources": "\n\n".join(state.get("search_results", [])),
            "rewrite_instruction": state.get("rewrite_instruction", "Write the best concise answer."),
        }
    ).content
    return state


def evaluator_agent(state: ResearchState) -> ResearchState:
    append_progress(state, "Evaluator Agent: checking completeness and quality.")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict research evaluator. Briefly score the answer from 1-10 on "
                "relevance, evidence use, clarity, and completeness. Mention only important gaps.",
            ),
            (
                "human",
                "Original query:\n{query}\n\nResearch plan:\n{plan}\n\nAnswer:\n{report}",
            ),
        ]
    )
    state["evaluation"] = (prompt | get_llm(temperature=0)).invoke(
        {"query": state["query"], "plan": state["plan"], "report": state["report"]}
    ).content
    append_progress(state, "Workflow complete.")
    save_session(state)
    return state


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("evaluator", evaluator_agent)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "evaluator")
    graph.add_edge("evaluator", END)
    return graph.compile()



def regenerate_answer(state: ResearchState) -> ResearchState:
    """Regenerate only the user-facing answer from existing research context."""
    state = dict(state)
    state["writer_temperature"] = 0.55
    state["rewrite_instruction"] = (
        "Regenerate the answer using the same research evidence. Phrase it freshly, "
        "keep it concise, and do not add unsupported facts."
    )
    append_progress(state, "Writer Agent: regenerating answer from existing research.")
    state = writer_agent(state)
    state = evaluator_agent(state)
    return state
def run_research(
    query: str,
    documents: Optional[List[Document]] = None,
    use_web: bool = True,
    use_rag: bool = True,
    max_results: int = 5,
) -> ResearchState:
    initial_state: ResearchState = {
        "session_id": str(uuid.uuid4()),
        "query": query.strip(),
        "documents": documents or [],
        "use_web": use_web,
        "use_rag": use_rag,
        "max_results": max_results,
        "progress": ["Session created."],
        "errors": [],
    }
    return build_graph().invoke(initial_state)


api_app = FastAPI(title=APP_TITLE, version="1.0.0")


@api_app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": APP_TITLE}


@api_app.post("/research", response_model=ResearchResponse)
async def research_endpoint(payload: ResearchRequest) -> ResearchResponse:
    try:
        state = await asyncio.to_thread(
            run_research,
            payload.query,
            None,
            payload.use_web,
            payload.use_rag,
            payload.max_results,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ResearchResponse(
        session_id=state["session_id"],
        query=state["query"],
        plan=state.get("plan", ""),
        insights=state.get("insights", ""),
        report=state.get("report", ""),
        evaluation=state.get("evaluation", ""),
        progress=state.get("progress", []),
    )


def render_streamlit_app() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("LangGraph + LangChain + Groq + DuckDuckGo + RAG + ChromaDB + MongoDB Atlas")

    with st.sidebar:
        st.header("Configuration")
        model = st.text_input("Groq model", value=DEFAULT_MODEL)
        os.environ["GROQ_MODEL"] = model
        use_web = st.toggle("Use DuckDuckGo web search", value=True)
        use_rag = st.toggle("Use uploaded document RAG", value=True)
        max_results = st.slider("Search results", min_value=1, max_value=10, value=5)
        uploaded_files = st.file_uploader(
            "Upload research documents",
            type=["pdf", "txt", "md", "docx", "html"],
            accept_multiple_files=True,
        )

    query = st.text_area(
        "Research topic or question",
        placeholder="Example: Analyze recent trends in agentic RAG systems for enterprise knowledge management.",
        height=120,
    )
    run_button = st.button("Run Multi-Agent Research", type="primary", use_container_width=True)

    if run_button:
        if not query.strip():
            st.warning("Enter a research query first.")
            return
        if not os.getenv("GROQ_API_KEY"):
            st.error("Missing GROQ_API_KEY. Add it to your .env file or deployment secrets.")
            return

        progress_box = st.status("Research workflow running...", expanded=True)
        try:
            with progress_box:
                st.write("Loading uploaded documents...")
                docs = load_uploaded_documents(uploaded_files)
                st.write(f"Loaded {len(docs)} document pages/sections.")
                st.write("Starting LangGraph workflow...")
                state = run_research(
                    query=query,
                    documents=docs,
                    use_web=use_web,
                    use_rag=use_rag,
                    max_results=max_results,
                )
                for item in state.get("progress", []):
                    st.write(item)
                progress_box.update(label="Research workflow complete.", state="complete")
        except Exception as exc:
            progress_box.update(label="Research workflow failed.", state="error")
            st.exception(exc)
            return

        st.session_state["last_result"] = state

    state = st.session_state.get("last_result")
    if state:
        if st.button("Regenerate answer", use_container_width=True):
            if not os.getenv("GROQ_API_KEY"):
                st.error("Missing GROQ_API_KEY. Add it to your .env file or deployment secrets.")
            else:
                with st.status("Regenerating answer from existing research...", expanded=True) as regen_status:
                    try:
                        state = regenerate_answer(state)
                        st.session_state["last_result"] = state
                        regen_status.update(label="Answer regenerated.", state="complete")
                    except Exception as exc:
                        regen_status.update(label="Regeneration failed.", state="error")
                        st.exception(exc)
        st.markdown(state.get("report", ""))


if __name__ == "__main__":
    render_streamlit_app()
