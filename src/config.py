"""
Configuration settings for Agentic RAG System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORDB_DIR = DATA_DIR / "vectordb"

# Create directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq, openai
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Document Processing Configuration
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # character overlap between chunks

# Retrieval Configuration
TOP_K = 5  # Number of top results to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score

# Vector DB Configuration
VECTORDB_TYPE = "faiss"  # faiss or chroma
VECTORDB_PATH = str(VECTORDB_DIR / "rag_index")

# Agent Configuration
MAX_ITERATIONS = 5
AGENT_TIMEOUT = 60  # seconds
USE_QUERY_REWRITING = True
USE_RE_RANKING = True

# Temperature and other LLM params
TEMPERATURE = 0.3
MAX_TOKENS = 2000
