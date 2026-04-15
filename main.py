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


def initialize_agent(vector_db: VectorDatabase) -> RAGAgent:
    """Initialize the RAG agent"""
    try:
        agent = RAGAgent(
            vector_db=vector_db,
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL,
            max_iterations=MAX_ITERATIONS,
            top_k=TOP_K,
            use_query_rewriting=USE_QUERY_REWRITING,
            use_reranking=USE_RE_RANKING
        )
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None


def display_sources(citations):
    """Display source citations"""
    if not citations:
        return

    st.divider()
    st.subheader("Sources")

    for i, citation in enumerate(citations, 1):
        with st.expander(f"Source {i}: {citation['source']}"):
            st.write(f"**Relevance Score:** {citation['relevance']:.2%}")
            st.write(f"**Chunk ID:** {citation['chunk_id']}")
            if citation['timestamp']:
                st.write(f"**Timestamp:** {citation['timestamp']}")


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
                st.session_state.agent = initialize_agent(st.session_state.vector_db)

            if st.session_state.agent:
                st.subheader("Ask Your Question")

                query = st.text_input(
                    "Enter your question about the documents:",
                    placeholder="e.g., What are the key features of ASP.NET Core 8?"
                )

                if query:
                    if st.button("Search", use_container_width=True, type="primary"):
                        with st.spinner("Thinking and retrieving..."):
                            try:
                                answer, state = st.session_state.agent.run(query)

                                st.session_state.chat_history.append({
                                    "question": query,
                                    "answer": answer,
                                    "citations": state.citations
                                })

                                st.subheader("Answer")
                                st.markdown(answer)

                                display_sources(state.citations)
                                display_reasoning(state)

                            except Exception as e:
                                st.error(f"Error processing query: {str(e)}")

            if st.session_state.chat_history:
                st.divider()
                st.subheader("Chat History")

                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['question'][:50]}..."):
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
