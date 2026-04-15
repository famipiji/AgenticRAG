#!/usr/bin/env python3
"""
Demo script for Agentic RAG System
Shows how to use the system programmatically
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DOCUMENTS_DIR, CHUNKS_DIR, VECTORDB_PATH, LLM_PROVIDER,
    LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)
from src.document_loader import DocumentLoader
from src.retrieval.vector_db import VectorDatabase
from src.agents.rag_agent import RAGAgent


def demo_document_loading():
    """Demo: Load and chunk documents"""
    print("\n" + "="*60)
    print("DEMO 1: Document Loading and Chunking")
    print("="*60)

    loader = DocumentLoader(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Process documents
    print("\nProcessing documents...")
    chunks = loader.process_documents(str(DOCUMENTS_DIR))

    print(f"\nSuccessfully loaded {len(chunks)} chunks")

    # Show sample chunks
    if chunks:
        print(f"\nSample chunk from '{chunks[0].source}':")
        print(f"   Chunk ID: {chunks[0].id}")
        print(f"   Size: {len(chunks[0].content)} characters")
        print(f"   Preview: {chunks[0].content[:200]}...")

    return chunks


def demo_vector_database(chunks):
    """Demo: Vector database operations"""
    print("\n" + "="*60)
    print("DEMO 2: Vector Database Operations")
    print("="*60)

    # Initialize vector DB
    print("\nInitializing vector database...")
    vector_db = VectorDatabase(vectordb_path=VECTORDB_PATH)

    # Add documents
    print("\nAdding documents to vector database...")
    vector_db.add_documents(chunks)

    # Get stats
    stats = vector_db.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Index type: {stats['index_type']}")

    # Semantic search demo
    print("\nPerforming semantic search...")
    query = "What features does ASP.NET Core have?"
    results = vector_db.search(query, top_k=3)

    print(f"\n   Query: '{query}'")
    print(f"   Found {len(results)} relevant documents:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"   {i}. [{doc.source}] (similarity: {score:.2f})")
        print(f"      {doc.content[:100]}...")

    # Hybrid search demo
    print("\nPerforming hybrid search...")
    hybrid_results = vector_db.hybrid_search(query, top_k=3)

    print(f"   Found {len(hybrid_results)} results with re-ranking:")
    for i, (doc, score, metadata) in enumerate(hybrid_results, 1):
        print(f"   {i}. [{doc.source}] (semantic: {score:.2f}, combined: {metadata['combined_score']:.2f})")

    return vector_db


def demo_agent_orchestration(vector_db):
    """Demo: Agent orchestration and query answering"""
    print("\n" + "="*60)
    print("DEMO 3: Agent Orchestration and Query Answering")
    print("="*60)

    # Initialize agent
    print("\nInitializing RAG Agent...")
    agent = RAGAgent(
        vector_db=vector_db,
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
        max_iterations=3,
        top_k=TOP_K,
        use_query_rewriting=True,
        use_reranking=True
    )

    # Example queries
    queries = [
        "What are the key features of ASP.NET Core 8?",
        "How should I design REST API endpoints?",
        "What are the types of machine learning?"
    ]

    for query in queries:
        print(f"\n{'─'*60}")
        print(f"Question: {query}")
        print(f"{'─'*60}")

        # Run agent
        print("\nAgent thinking...")
        answer, state = agent.run(query)

        # Display results
        print(f"\nAnswer:")
        print(f"{answer[:500]}..." if len(answer) > 500 else f"{answer}")

        # Show reasoning
        print(f"\nAgent Reasoning:")
        for step in state.reasoning_history:
            print(f"   -> {step}")

        # Show citations
        if state.citations:
            print(f"\nSources:")
            for citation in state.citations:
                print(f"   - {citation['source']} (relevance: {citation['relevance']:.2%})")


def main():
    """Run all demos"""
    print("\nStarting Agentic RAG System Demo")
    print("=" * 60)

    # Check if documents exist
    if not list(DOCUMENTS_DIR.glob("*")):
        print(f"\nNo documents found in {DOCUMENTS_DIR}")
        print("   Please add documents to the documents folder first.")
        return

    try:
        # Demo 1: Document Loading
        chunks = demo_document_loading()

        # Demo 2: Vector Database
        vector_db = demo_vector_database(chunks)

        # Demo 3: Agent Orchestration
        demo_agent_orchestration(vector_db)

        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("   1. Try the web interface: streamlit run main.py")
        print("   2. Add your own documents to data/documents/")
        print("   3. Customize agent settings in src/config.py")

    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
