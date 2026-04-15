"""
Main Streamlit application for Agentic RAG System
"""
import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DOCUMENTS_DIR, CHUNKS_DIR, VECTORDB_PATH, LLM_PROVIDER,
    GROQ_API_KEY, OPENAI_API_KEY, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, MAX_ITERATIONS, USE_QUERY_REWRITING, USE_RE_RANKING,
    EMBEDDING_MODEL
)
from src.document_loader import DocumentLoader
from src.retrieval.vector_db import VectorDatabase
from src.agents.rag_agent import RAGAgent


def initialize_session_state():
    """Initialize Streamlit session state"""
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "db_loaded" not in st.session_state:
        st.session_state.db_loaded = False


def load_or_build_vector_db():
    """Load or build the vector database"""
    with st.spinner("Loading vector database..."):
        try:
            vector_db = VectorDatabase(
                embedding_model=EMBEDDING_MODEL,
                vectordb_path=VECTORDB_PATH
            )

            if vector_db.get_stats()["total_documents"] == 0:
                return None

            st.success(f"Vector DB loaded with {vector_db.get_stats()['total_documents']} documents")
            return vector_db
        except Exception as e:
            st.error(f"Error loading vector database: {str(e)}")
            return None


def load_documents_to_db(vector_db: VectorDatabase):
    """Load documents from disk to vector database"""
    if not DOCUMENTS_DIR.exists() or not list(DOCUMENTS_DIR.glob("*")):
        st.warning("No documents found in the documents directory")
        return

    with st.spinner("Processing documents..."):
        try:
            loader = DocumentLoader(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )

            chunks = loader.process_documents(str(DOCUMENTS_DIR))

            if chunks:
                vector_db.add_documents(chunks)
                loader.save_chunks(chunks, str(CHUNKS_DIR))
                st.success(f"Loaded {len(chunks)} document chunks into vector database")
            else:
                st.warning("No documents were processed")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")


def initialize_agent(vector_db: VectorDatabase, top_k: int, query_rewrite: bool, use_reranking: bool) -> RAGAgent:
    """Initialize the RAG agent"""
    try:
        agent = RAGAgent(
            vector_db=vector_db,
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL,
            max_iterations=MAX_ITERATIONS,
            top_k=top_k,
            use_query_rewriting=query_rewrite,
            use_reranking=use_reranking
        )
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None


@st.dialog("Document Preview", width="large")
def preview_document_dialog():
    chunk = st.session_state.get("preview_chunk")
    if not chunk:
        return

    # Header row
    col_title, col_close = st.columns([5, 1])
    with col_title:
        st.markdown(f"### {chunk['source']}")
    with col_close:
        if st.button("Close", use_container_width=True):
            del st.session_state["preview_chunk"]
            st.rerun()

    # Metadata strip
    meta_cols = st.columns(3)
    meta_cols[0].metric("Relevance", f"{chunk['relevance']:.2%}")
    meta_cols[1].metric("Chunk", f"{chunk['chunk_index'] + 1} / {chunk['total_chunks']}")
    if chunk.get("metadata", {}).get("chunk_size"):
        meta_cols[2].metric("Chunk size", f"{chunk['metadata']['chunk_size']} chars")

    st.divider()

    # Chunk content
    st.markdown("**Retrieved passage:**")
    ext = Path(chunk["source"]).suffix.lower()
    if ext == ".md":
        st.markdown(chunk["content"])
    else:
        st.text_area(
            label="",
            value=chunk["content"],
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )

    # Download original file if it still exists on disk
    file_path = DOCUMENTS_DIR / chunk["source"]
    if file_path.exists():
        st.divider()
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".md": "text/markdown",
        }
        mime = mime_map.get(ext, "application/octet-stream")
        st.download_button(
            label=f"Download {chunk['source']}",
            data=file_bytes,
            file_name=chunk["source"],
            mime=mime,
            use_container_width=True,
        )


def display_sources(citations, retrieved_documents=None):
    """Display source citations with preview buttons"""
    if not citations:
        return

    # Build lookup from chunk_id → Document for full content access
    doc_lookup = {}
    if retrieved_documents:
        for doc, _ in retrieved_documents:
            doc_lookup[doc.id] = doc

    st.divider()
    st.subheader("Sources")

    for i, citation in enumerate(citations, 1):
        col_info, col_btn = st.columns([5, 1])
        with col_info:
            st.markdown(
                f"**{i}. {citation['source']}** &nbsp;·&nbsp; "
                f"{citation['relevance']:.2%} relevance",
                unsafe_allow_html=True
            )
        with col_btn:
            doc = doc_lookup.get(citation["chunk_id"])
            if doc is not None:
                if st.button("Preview", key=f"preview_{i}_{citation['chunk_id']}"):
                    st.session_state["preview_chunk"] = {
                        "source": citation["source"],
                        "relevance": citation["relevance"],
                        "content": doc.content,
                        "chunk_index": doc.chunk_index,
                        "total_chunks": doc.total_chunks,
                        "metadata": doc.metadata or {},
                    }
                    st.rerun()


def display_reasoning(state):
    """Display agent reasoning history"""
    if not state.reasoning_history:
        return

    with st.expander("Agent Reasoning (Debug Info)"):
        for step in state.reasoning_history:
            st.write(f"-> {step}")


def handle_uploaded_files(uploaded_files):
    """Save uploaded files to documents dir and index them"""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded_file in uploaded_files:
        dest = DOCUMENTS_DIR / uploaded_file.name
        dest.write_bytes(uploaded_file.getvalue())
        saved.append(uploaded_file.name)

    if st.session_state.vector_db is None:
        st.session_state.vector_db = VectorDatabase(
            embedding_model=EMBEDDING_MODEL,
            vectordb_path=VECTORDB_PATH
        )

    load_documents_to_db(st.session_state.vector_db)
    st.session_state.agent = None
    st.session_state.db_loaded = True
    return saved


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Agentic RAG System",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Trigger document preview dialog if a chunk was selected
    if "preview_chunk" in st.session_state:
        preview_document_dialog()

    st.title("Agentic RAG System")
    st.markdown("Ask questions about your documents with AI-powered reasoning, retrieval, and answer generation.")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Retrieval Settings")
        top_k_display = st.slider("Top K Results", min_value=1, max_value=20, value=TOP_K)

        st.subheader("Agent Settings")
        query_rewrite = st.checkbox("Enable Query Rewriting", value=USE_QUERY_REWRITING)
        use_reranking = st.checkbox("Enable Re-ranking", value=USE_RE_RANKING)

        # Re-init agent if settings changed
        settings_changed = (
            st.session_state.get("active_top_k") != top_k_display
            or st.session_state.get("active_query_rewrite") != query_rewrite
            or st.session_state.get("active_use_reranking") != use_reranking
        )
        if settings_changed and st.session_state.get("agent") is not None:
            st.session_state.agent = None

        st.session_state.active_top_k = top_k_display
        st.session_state.active_query_rewrite = query_rewrite
        st.session_state.active_use_reranking = use_reranking

        st.divider()

        st.subheader("LLM Provider")
        st.write(f"Provider: `{LLM_PROVIDER.upper()}`")
        st.write(f"Model: `{LLM_MODEL}`")

        if LLM_PROVIDER.lower() == "groq":
            if not GROQ_API_KEY:
                st.warning("GROQ_API_KEY not set. Set it in .env file.")
        else:
            if not OPENAI_API_KEY:
                st.warning("OPENAI_API_KEY not set. Set it in .env file.")

        st.divider()

        st.subheader("Database Stats")
        if st.session_state.vector_db:
            stats = st.session_state.vector_db.get_stats()
            st.write(f"Text Chunks: {stats['total_documents']}")
            st.write(f"Embedding Dim: {stats['embedding_dimension']}")

    # Main content
    col1, col2 = st.columns([3, 1])

    with col1:
        # Initialize vector DB
        if st.session_state.vector_db is None:
            st.session_state.vector_db = load_or_build_vector_db()

        has_documents = (
            st.session_state.vector_db is not None
            and st.session_state.vector_db.get_stats()["total_documents"] > 0
        )

        if not has_documents:
            # Show upload area prominently when no documents are loaded
            st.subheader("Upload Documents")
            st.markdown("Drag and drop your files below, or click to browse.")

            uploaded_files = st.file_uploader(
                "Supported formats: PDF, DOCX, TXT, MD",
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                label_visibility="visible"
            )

            if uploaded_files:
                if st.button("Load into Knowledge Base", use_container_width=True, type="primary"):
                    with st.spinner("Saving and indexing documents..."):
                        saved = handle_uploaded_files(uploaded_files)
                    st.success(f"Indexed {len(saved)} file(s): {', '.join(saved)}")
                    st.rerun()

            st.divider()
            st.markdown("**Or load documents already in the folder:**")
            if st.button("Load from `data/documents/` folder", use_container_width=True):
                st.session_state.vector_db = VectorDatabase(
                    embedding_model=EMBEDDING_MODEL,
                    vectordb_path=VECTORDB_PATH
                )
                load_documents_to_db(st.session_state.vector_db)
                st.session_state.db_loaded = True
                st.rerun()

        else:
            # Documents loaded — show Q&A interface
            # Upload more documents expander
            with st.expander("Upload more documents"):
                extra_files = st.file_uploader(
                    "Drag and drop files here",
                    type=["pdf", "docx", "txt", "md"],
                    accept_multiple_files=True,
                    key="extra_uploader"
                )
                if extra_files:
                    if st.button("Add to Knowledge Base", use_container_width=True):
                        with st.spinner("Saving and indexing..."):
                            saved = handle_uploaded_files(extra_files)
                        st.success(f"Added {len(saved)} file(s)")
                        st.rerun()

            # Initialize agent
            if st.session_state.agent is None:
                st.session_state.agent = initialize_agent(
                    st.session_state.vector_db,
                    top_k=top_k_display,
                    query_rewrite=query_rewrite,
                    use_reranking=use_reranking
                )

            if st.session_state.agent:
                st.subheader("Ask Your Question")

                query = st.text_input(
                    "Enter your question about the documents:",
                    placeholder="e.g., What are the key features of ASP.NET Core 8?"
                )

                if query:
                    if st.button("Search", use_container_width=True, type="primary"):
                        # Pass last 3 Q&A turns as memory context
                        recent_history = st.session_state.chat_history[-3:]
                        state_ref = {}

                        try:
                            st.subheader("Answer")
                            answer = st.write_stream(
                                st.session_state.agent.run_with_streaming(
                                    query,
                                    chat_history=recent_history,
                                    state_ref=state_ref
                                )
                            )

                            state = state_ref.get("state")

                            st.session_state.chat_history.append({
                                "question": query,
                                "answer": answer,
                                "citations": state.citations if state else []
                            })

                            if state:
                                display_sources(state.citations, state.retrieved_documents)
                                display_reasoning(state)

                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")

            if st.session_state.chat_history:
                st.divider()
                st.subheader("Chat History")

                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    q = chat['question']
                    label = f"Q: {q[:50]}{'...' if len(q) > 50 else ''}"
                    with st.expander(label):
                        st.write(chat['answer'])

    with col2:
        st.subheader("About")
        st.info(
            """
            **Agentic RAG System**

            This system uses:
            - Vector embeddings for semantic search
            - Agent orchestration for intelligent retrieval
            - Multiple document formats
            - Re-ranking and query rewriting
            - Source citation tracking
            """
        )


if __name__ == "__main__":
    main()
