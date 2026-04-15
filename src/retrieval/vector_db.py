"""
Vector Database module for storing and retrieving embeddings
"""
from typing import List, Optional, Dict, Tuple
import numpy as np
from pathlib import Path
import json
import faiss

from sentence_transformers import SentenceTransformer
from ..document_loader import Document


class VectorDatabase:
    """FAISS-based vector database for document embeddings"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vectordb_path: str = "data/vectordb/rag_index"
    ):
        self.embedding_model_name = embedding_model
        self.vectordb_path = Path(vectordb_path)
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.doc_mapping = {}  # Map from index to document
        
        self._load_index_if_exists()

    def _load_index_if_exists(self):
        """Load existing FAISS index if it exists"""
        index_file = self.vectordb_path / "faiss.index"
        docs_file = self.vectordb_path / "documents.json"
        mapping_file = self.vectordb_path / "mapping.json"
        
        if index_file.exists() and docs_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(docs_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                with open(mapping_file, "r", encoding="utf-8") as f:
                    self.doc_mapping = json.load(f)
                print(f"Loaded existing FAISS index with {len(self.documents)} documents")
            except Exception as e:
                print(f"Error loading existing index: {str(e)}")
                self._create_index()
        else:
            self._create_index()

    def _create_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
        self.doc_mapping = {}

    def get_indexed_sources(self) -> set:
        """Return the set of source filenames already in the index"""
        return {doc["source"] for doc in self.documents}

    def add_documents(self, documents: List[Document]) -> Dict:
        """Add documents to the vector database, skipping already-indexed sources.

        Returns a dict with 'added' and 'skipped' source name sets.
        """
        if not documents:
            return {"added": set(), "skipped": set()}

        indexed_sources = self.get_indexed_sources()
        new_docs = [doc for doc in documents if doc.source not in indexed_sources]
        skipped_sources = {doc.source for doc in documents if doc.source in indexed_sources}

        if skipped_sources:
            print(f"Skipping already-indexed sources: {skipped_sources}")

        if not new_docs:
            return {"added": set(), "skipped": skipped_sources}

        texts = [doc.content for doc in new_docs]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)

        start_idx = len(self.documents)
        self.index.add(embeddings)

        for i, doc in enumerate(new_docs):
            doc_dict = doc.to_dict() if hasattr(doc, "to_dict") else {
                "id": doc.id,
                "content": doc.content,
                "source": doc.source,
                "chunk_index": doc.chunk_index,
                "metadata": doc.metadata
            }
            self.documents.append(doc_dict)
            self.doc_mapping[str(start_idx + i)] = doc.id

        self._save_index()
        added_sources = {doc.source for doc in new_docs}
        print(f"Added {len(new_docs)} chunks from {len(added_sources)} source(s)")
        return {"added": added_sources, "skipped": skipped_sources}

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (L2 distance, lower is better)
        
        Returns:
            List of (Document, similarity_score) tuples
        """
        if len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            idx = int(idx)
            if idx < 0:
                continue
            if distance <= similarity_threshold or similarity_threshold == 0.0:
                doc_dict = self.documents[idx]
                doc = Document(**doc_dict)
                # Convert L2 distance to similarity score (0-1, higher is better)
                similarity = float(1 / (1 + distance))
                results.append((doc, similarity))
        
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[Document, float, Dict]]:
        """
        Hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            top_k: Number of top results
        
        Returns:
            List of (Document, similarity_score, metadata) tuples
        """
        results = self.search(query, top_k=top_k * 2)
        
        # Re-rank based on keyword presence
        ranked_results = []
        query_keywords = set(query.lower().split())
        
        for doc, similarity in results:
            # Count keyword matches
            content_words = set(doc.content.lower().split())
            keyword_matches = len(query_keywords & content_words)
            keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0
            
            # Combined score
            combined_score = 0.7 * similarity + 0.3 * keyword_score
            ranked_results.append((doc, similarity, {"keyword_score": keyword_score, "combined_score": combined_score}))
        
        # Sort by combined score
        ranked_results.sort(key=lambda x: x[2]["combined_score"], reverse=True)
        
        return ranked_results[:top_k]

    def _save_index(self) -> None:
        """Save FAISS index to disk"""
        index_file = self.vectordb_path / "faiss.index"
        docs_file = self.vectordb_path / "documents.json"
        mapping_file = self.vectordb_path / "mapping.json"
        
        faiss.write_index(self.index, str(index_file))
        
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.doc_mapping, f, indent=2)

    def get_stats(self) -> Dict:
        """Get statistics about the vector database"""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "index_type": "FAISS-L2",
            "embedding_model": self.embedding_model_name
        }
