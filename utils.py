#!/usr/bin/env python3
"""
Utility script for managing Agentic RAG system
Handles vector database operations, document management, etc.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DOCUMENTS_DIR, CHUNKS_DIR, VECTORDB_PATH
from src.document_loader import DocumentLoader
from src.retrieval.vector_db import VectorDatabase


def build_database():
    """Build the vector database from documents"""
    print("\n🔨 Building Vector Database...")
    print("=" * 60)
    
    # Load documents
    loader = DocumentLoader()
    print("📄 Loading documents...")
    chunks = loader.process_documents(str(DOCUMENTS_DIR))
    
    if not chunks:
        print("❌ No documents found to process!")
        return False
    
    # Create vector DB
    print(f"\n🗄️  Creating vector database with {len(chunks)} chunks...")
    vector_db = VectorDatabase(vectordb_path=VECTORDB_PATH)
    vector_db.add_documents(chunks)
    
    # Save chunks reference
    loader.save_chunks(chunks, str(CHUNKS_DIR))
    
    print("\n✅ Vector database built successfully!")
    print(f"   Documents: {len(chunks)}")
    
    stats = vector_db.get_stats()
    print(f"   Total indexed: {stats['total_documents']}")
    
    return True


def rebuild_database(force=False):
    """Rebuild the vector database from scratch"""
    print("\n🔄 Rebuilding Vector Database...")
    
    if not force:
        confirm = input("⚠️  This will delete the current database. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("❌ Cancelled")
            return False
    
    # Remove old database
    import shutil
    vectordb_dir = Path(VECTORDB_PATH)
    if vectordb_dir.parent.exists():
        shutil.rmtree(vectordb_dir.parent, ignore_errors=True)
    
    # Build new
    return build_database()


def show_stats():
    """Show vector database statistics"""
    print("\n📊 Vector Database Statistics")
    print("=" * 60)
    
    try:
        vector_db = VectorDatabase(vectordb_path=VECTORDB_PATH)
        stats = vector_db.get_stats()
        
        print(f"Total documents: {stats['total_documents']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Index type: {stats['index_type']}")
        
        if stats['total_documents'] == 0:
            print("\n⚠️  Vector database is empty. Run 'build' command to populate.")
        else:
            print(f"\n✅ Database is ready with {stats['total_documents']} documents")
    except Exception as e:
        print(f"❌ Error loading database: {str(e)}")


def list_documents():
    """List all documents in the documents directory"""
    print("\n📚 Documents in Database")
    print("=" * 60)
    
    try:
        vector_db = VectorDatabase(vectordb_path=VECTORDB_PATH)
        
        if not vector_db.documents:
            print("❌ Vector database is empty")
            return
        
        # Group by source
        sources = {}
        for doc in vector_db.documents:
            source = doc['source']
            sources[source] = sources.get(source, 0) + 1
        
        print(f"\nTotal chunks: {len(vector_db.documents)}\n")
        for source, count in sorted(sources.items()):
            print(f"  📄 {source}: {count} chunks")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def search_documents(query):
    """Search the vector database"""
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 60)
    
    try:
        vector_db = VectorDatabase(vectordb_path=VECTORDB_PATH)
        
        if not vector_db.documents:
            print("❌ Vector database is empty")
            return
        
        results = vector_db.search(query, top_k=5)
        
        print(f"\nFound {len(results)} results:\n")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. [{doc.source}] (relevance: {score:.2%})")
            print(f"   {doc.content[:150]}...")
            print()
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Agentic RAG System Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils.py build              # Build vector database from documents
  python utils.py rebuild --force    # Rebuild database from scratch
  python utils.py stats              # Show database statistics
  python utils.py list               # List all documents
  python utils.py search "query"     # Search database
        """
    )
    
    parser.add_argument(
        "command",
        choices=["build", "rebuild", "stats", "list", "search"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query (for 'search' command)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force operation without confirmation"
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == "build":
        build_database()
    elif args.command == "rebuild":
        rebuild_database(force=args.force)
    elif args.command == "stats":
        show_stats()
    elif args.command == "list":
        list_documents()
    elif args.command == "search":
        if not args.query:
            print("❌ Search query required")
            parser.print_help()
        else:
            search_documents(args.query)


if __name__ == "__main__":
    main()
