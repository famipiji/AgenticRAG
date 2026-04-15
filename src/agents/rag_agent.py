"""
Agent Orchestration module for Agentic RAG
Manages query rewriting, retrieval, re-ranking, and answer generation
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ..retrieval.vector_db import VectorDatabase
from ..llm.llm_integration import LLMProvider, get_llm_provider
from ..document_loader import Document


class ActionType(Enum):
    """Agent action types"""
    SEARCH = "search"
    REFINE_QUERY = "refine_query"
    RERANK = "rerank"
    GENERATE_ANSWER = "generate_answer"
    RETRIEVE_MORE = "retrieve_more"
    STOP = "stop"


@dataclass
class AgentState:
    """State tracking for agent decisions"""
    original_query: str
    refined_queries: List[str] = field(default_factory=list)
    iterations: int = 0
    retrieved_documents: List[Tuple[Document, float]] = field(default_factory=list)
    reasoning_history: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    citations: List[Dict] = field(default_factory=list)


class QueryRewriter:
    """Rewrites and refines user queries for better retrieval"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def rewrite_query(self, original_query: str) -> List[str]:
        """Generate multiple query variations for better retrieval"""
        prompt = f"""Rewrite the following query in 3 different ways to improve retrieval:
        
Original query: "{original_query}"

Generate exactly 3 alternative phrasings that would help find relevant documents.
Response format: Return only the queries, one per line, without numbering."""

        response = self.llm.generate(prompt)
        queries = [q.strip() for q in response.split("\n") if q.strip()]
        return queries[:3]  # Ensure we get exactly 3


class DocumentRanker:
    """Re-ranks retrieved documents based on relevance"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def rank_documents(
        self,
        query: str,
        documents: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Re-rank documents using LLM judgment"""
        if len(documents) <= 1:
            return documents
        
        # Prepare document summaries
        doc_summaries = "\n".join([
            f"{i+1}. [{doc.source}] {doc.content[:200]}..."
            for i, (doc, _) in enumerate(documents)
        ])
        
        prompt = f"""Given this query: "{query}"
        
Rank the following documents by relevance (most relevant first):

{doc_summaries}

Return only the ranking order (e.g., "3, 1, 2") without explanation."""

        try:
            response = self.llm.generate(prompt)
            ranking = [int(x.strip()) - 1 for x in response.split(",")]
            
            # Reorder documents
            ranked = []
            for idx in ranking:
                if 0 <= idx < len(documents):
                    ranked.append(documents[idx])
            
            return ranked
        except:
            # If ranking fails, return original order
            return documents


class RAGAgent:
    """Main agent orchestrating the RAG workflow"""

    def __init__(
        self,
        vector_db: VectorDatabase,
        llm_provider: str = "groq",
        llm_model: Optional[str] = None,
        max_iterations: int = 5,
        top_k: int = 5,
        use_query_rewriting: bool = True,
        use_reranking: bool = True,
    ):
        self.vector_db = vector_db
        self.llm = get_llm_provider(provider=llm_provider, model=llm_model)
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.use_query_rewriting = use_query_rewriting
        self.use_reranking = use_reranking
        
        self.query_rewriter = QueryRewriter(self.llm)
        self.document_ranker = DocumentRanker(self.llm)

    def decide_action(self, state: AgentState) -> ActionType:
        """Decide next action based on current state"""
        # If we've hit max iterations, generate answer
        if state.iterations >= self.max_iterations:
            return ActionType.GENERATE_ANSWER
        
        # If we have no documents yet, search
        if not state.retrieved_documents:
            if state.iterations == 0:
                return ActionType.SEARCH
            elif self.use_query_rewriting and len(state.refined_queries) < 3:
                return ActionType.REFINE_QUERY
            else:
                return ActionType.SEARCH
        
        # If we have documents and haven't tried reranking, do it
        if self.use_reranking and state.iterations == 1:
            return ActionType.RERANK
        
        # Otherwise, generate answer
        return ActionType.GENERATE_ANSWER

    def search_documents(self, query: str, state: AgentState) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        state.reasoning_history.append(f"Searching for: '{query}'")
        
        # Use hybrid search for better results
        results = self.vector_db.hybrid_search(query, top_k=self.top_k)
        
        # Convert results to (Document, score) tuples
        return [(doc, score) for doc, score, _ in results]

    def refine_query(self, state: AgentState) -> str:
        """Refine the original query"""
        if not state.refined_queries:
            state.refined_queries = self.query_rewriter.rewrite_query(state.original_query)
            state.reasoning_history.append(
                f"Generated alternative queries: {state.refined_queries}"
            )
        
        # Return next refined query
        if state.iterations < len(state.refined_queries) + 1:
            next_query = state.refined_queries[state.iterations - 1]
            state.reasoning_history.append(f"Using refined query: '{next_query}'")
            return next_query
        
        return state.original_query

    def rerank_documents(self, state: AgentState):
        """Re-rank retrieved documents"""
        if not state.retrieved_documents:
            return
        
        state.reasoning_history.append("Re-ranking documents...")
        state.retrieved_documents = self.document_ranker.rank_documents(
            state.original_query,
            state.retrieved_documents
        )

    def generate_answer(self, state: AgentState) -> str:
        """Generate final answer with citations"""
        if not state.retrieved_documents:
            return "No relevant documents found to answer your question."
        
        # Prepare context from top documents
        context = self._prepare_context(state.retrieved_documents)
        
        # Generate answer prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

User Question: {state.original_query}

Relevant Documents:
{context}

Please answer the question based on the provided documents. 
Include specific references like [Source: document_name] for any claims.
If information is not in the documents, say so."""

        state.reasoning_history.append("Generating answer from retrieved documents...")
        answer = self.llm.generate(prompt)
        
        # Extract citations from answer and documents
        state.citations = self._extract_citations(state.retrieved_documents)
        
        return answer

    def _prepare_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Prepare context string from documents"""
        context_parts = []
        for i, (doc, score) in enumerate(documents, 1):
            context_parts.append(
                f"Document {i} [{doc.source}] (relevance: {score:.2f}):\n{doc.content[:500]}..."
            )
        return "\n".join(context_parts)

    def _extract_citations(self, documents: List[Tuple[Document, float]]) -> List[Dict]:
        """Extract citation information from documents"""
        citations = []
        for doc, score in documents:
            citations.append({
                "source": doc.source,
                "chunk_id": doc.id,
                "relevance": float(score),
                "timestamp": doc.metadata.get("timestamp") if doc.metadata else None
            })
        return citations

    def run(self, query: str) -> Tuple[str, AgentState]:
        """Run the agent on a query"""
        state = AgentState(original_query=query)
        
        while state.iterations < self.max_iterations:
            state.iterations += 1
            
            # Decide action
            action = self.decide_action(state)
            
            # Execute action
            if action == ActionType.SEARCH:
                search_query = query if state.iterations == 1 else self.refine_query(state)
                state.retrieved_documents = self.search_documents(search_query, state)
                
            elif action == ActionType.REFINE_QUERY:
                if not state.refined_queries and self.use_query_rewriting:
                    self.refine_query(state)
                    
            elif action == ActionType.RERANK:
                self.rerank_documents(state)
                
            elif action == ActionType.GENERATE_ANSWER:
                answer = self.generate_answer(state)
                state.final_answer = answer
                break
        
        # Ensure we have an answer
        if state.final_answer is None:
            state.final_answer = self.generate_answer(state)
        
        return state.final_answer, state

    def run_with_streaming(self, query: str):
        """Run the agent with streaming response"""
        state = AgentState(original_query=query)
        
        while state.iterations < self.max_iterations:
            state.iterations += 1
            action = self.decide_action(state)
            
            if action == ActionType.SEARCH:
                search_query = query if state.iterations == 1 else self.refine_query(state)
                state.retrieved_documents = self.search_documents(search_query, state)
            elif action == ActionType.REFINE_QUERY:
                if not state.refined_queries and self.use_query_rewriting:
                    self.refine_query(state)
            elif action == ActionType.RERANK:
                self.rerank_documents(state)
            elif action == ActionType.GENERATE_ANSWER:
                break
        
        # Generate context
        context = self._prepare_context(state.retrieved_documents) if state.retrieved_documents else ""
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

User Question: {state.original_query}

Relevant Documents:
{context}

Please answer the question based on the provided documents. Include citations."""

        for chunk in self.llm.generate_with_streaming(prompt):
            yield chunk
        
        # Extract citations
        state.citations = self._extract_citations(state.retrieved_documents)
