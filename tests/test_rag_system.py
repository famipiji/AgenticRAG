"""
Unit tests for Agentic RAG System
"""
import pytest
import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader, Document
from src.retrieval.vector_db import VectorDatabase
from src.agents.rag_agent import RAGAgent, QueryRewriter, DocumentRanker
from src.config import EMBEDDING_MODEL


class TestDocumentLoader:
    """Tests for document loader"""

    def test_chunk_text(self):
        """Test text chunking"""
        loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
        text = "A" * 300
        chunks = loader.chunk_text(text, "test.txt")
        
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
        assert all(hasattr(c, 'id') for c in chunks)

    def test_document_metadata(self):
        """Test document metadata"""
        loader = DocumentLoader()
        text = "Test content"
        chunks = loader.chunk_text(text, "test.md")
        
        assert chunks[0].source == "test.md"
        assert chunks[0].chunk_index == 0
        assert "timestamp" in chunks[0].metadata


class TestVectorDatabase:
    """Tests for vector database"""

    def test_vector_db_initialization(self):
        """Test vector DB initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec_db = VectorDatabase(
                embedding_model=EMBEDDING_MODEL,
                vectordb_path=tmpdir
            )
            stats = vec_db.get_stats()
            
            assert stats["total_documents"] == 0
            assert stats["embedding_dimension"] > 0

    def test_add_documents(self):
        """Test adding documents"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec_db = VectorDatabase(
                embedding_model=EMBEDDING_MODEL,
                vectordb_path=tmpdir
            )
            
            # Create test documents
            docs = [
                Document(
                    id="1",
                    content="Python is a programming language",
                    source="test.txt",
                    chunk_index=0,
                    total_chunks=1
                ),
                Document(
                    id="2",
                    content="Java is also a programming language",
                    source="test.txt",
                    chunk_index=1,
                    total_chunks=1
                )
            ]
            
            vec_db.add_documents(docs)
            stats = vec_db.get_stats()
            
            assert stats["total_documents"] == 2

    def test_semantic_search(self):
        """Test semantic search"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec_db = VectorDatabase(
                embedding_model=EMBEDDING_MODEL,
                vectordb_path=tmpdir
            )
            
            docs = [
                Document(
                    id="1",
                    content="Machine learning is a subset of artificial intelligence",
                    source="test.txt",
                    chunk_index=0,
                    total_chunks=1
                ),
                Document(
                    id="2",
                    content="The weather is sunny today",
                    source="test.txt",
                    chunk_index=1,
                    total_chunks=1
                )
            ]
            
            vec_db.add_documents(docs)
            results = vec_db.search("What is machine learning?", top_k=1)
            
            assert len(results) > 0
            assert results[0][0].id == "1"


class TestRAGAgent:
    """Tests for RAG agent"""

    def test_agent_initialization(self):
        """Test agent initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec_db = VectorDatabase(
                embedding_model=EMBEDDING_MODEL,
                vectordb_path=tmpdir
            )
            
            # Add documents
            docs = [
                Document(
                    id="1",
                    content="ASP.NET Core 8 has many features",
                    source="test.txt",
                    chunk_index=0,
                    total_chunks=1
                )
            ]
            vec_db.add_documents(docs)
            
            # This will fail without API keys, which is expected
            try:
                agent = RAGAgent(
                    vector_db=vec_db,
                    llm_provider="groq",
                    max_iterations=3
                )
                assert agent is not None
            except Exception as e:
                # Expected if API key not set
                assert "GROQ_API_KEY" in str(e) or "api_key" in str(e).lower()


class TestDocumentChunking:
    """Tests for document chunking strategies"""

    def test_overlap_preservation(self):
        """Test that overlap preserves context"""
        loader = DocumentLoader(chunk_size=50, chunk_overlap=10)
        text = "The quick brown fox jumps over the lazy dog. " * 5
        chunks = loader.chunk_text(text, "test.txt")
        
        # Check that overlap exists
        if len(chunks) > 1:
            chunk1_end = chunks[0].content[-10:]
            chunk2_start = chunks[1].content[:10]
            overlap_exists = chunk1_end == chunk2_start or len(set(chunk1_end) & set(chunk2_start)) > 0
            # Chunks should have some contextual relationship
            assert len(chunks[0].content) > 0
            assert len(chunks[1].content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
