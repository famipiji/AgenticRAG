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
    chat_history: List[Dict] = field(default_factory=list)
    refined_queries: List[str] = field(default_factory=list)
    iterations: int = 0
    retrieved_documents: List[Tuple[Document, float]] = field(default_factory=list)
    reasoning_history: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    citations: List[Dict] = field(default_factory=list)
    has_searched: bool = False
    has_reranked: bool = False


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
        if state.iterations >= self.max_iterations:
            return ActionType.GENERATE_ANSWER

        # Always do the initial search first
        if not state.has_searched:
            return ActionType.SEARCH

        # No results — try query rewriting then retry search
        if not state.retrieved_documents:
            if self.use_query_rewriting and len(state.refined_queries) < 3:
                return ActionType.REFINE_QUERY
            return ActionType.SEARCH

        # Have results — rerank if enabled and not done yet
        if self.use_reranking and not state.has_reranked:
            return ActionType.RERANK

        return ActionType.GENERATE_ANSWER

    def search_documents(self, query: str, state: AgentState) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        state.reasoning_history.append(f"Searching for: '{query}'")
        state.has_searched = True

        results = self.vector_db.hybrid_search(query, top_k=self.top_k)
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
        state.has_reranked = True
        state.retrieved_documents = self.document_ranker.rank_documents(
            state.original_query,
            state.retrieved_documents
        )

    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        """Format chat history for inclusion in the prompt"""
        if not chat_history:
            return ""
        lines = ["Previous conversation:"]
        for turn in chat_history:
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer']}")
        return "\n".join(lines) + "\n\n"

    def generate_answer(self, state: AgentState) -> str:
        """Generate final answer with citations"""
        if not state.retrieved_documents:
            return "No relevant documents found to answer your question."

        context = self._prepare_context(state.retrieved_documents)
        history_block = self._format_chat_history(state.chat_history)

        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

{history_block}Relevant Documents:
{context}

User Question: {state.original_query}

Please answer the question based on the provided documents.
Include specific references like [Source: document_name] for any claims.
If information is not in the documents, say so."""

        state.reasoning_history.append("Generating answer from retrieved documents...")
        answer = self.llm.generate(prompt)

        state.citations = self._extract_citations(state.retrieved_documents)

        return answer

    # Reserve ~2000 tokens for the question, history, and LLM response.
    # 1 token ≈ 4 chars; 24 000 chars ≈ 6 000 tokens of context budget.
    MAX_CONTEXT_CHARS = 24_000

    def _prepare_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Prepare context string from documents, capped to MAX_CONTEXT_CHARS total."""
        context_parts = []
        used = 0

        for i, (doc, score) in enumerate(documents, 1):
            header = f"Document {i} [{doc.source}] (relevance: {score:.2f}):\n"
            remaining = self.MAX_CONTEXT_CHARS - used - len(header) - 4  # 4 for "\n\n"
            if remaining <= 0:
                break
            content = doc.content[:remaining]
            part = header + content
            context_parts.append(part)
            used += len(part) + 4  # account for the "\n\n" separator

        return "\n\n".join(context_parts)

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

    def run(self, query: str, chat_history: Optional[List[Dict]] = None) -> Tuple[str, AgentState]:
        """Run the agent on a query"""
        state = AgentState(original_query=query, chat_history=chat_history or [])
        
        while state.iterations < self.max_iterations:
            state.iterations += 1
            
            # Decide action
            action = self.decide_action(state)
            
            # Execute action
            if action == ActionType.SEARCH:
                state.retrieved_documents = self.search_documents(query, state)

            elif action == ActionType.REFINE_QUERY:
                refined_query = self.refine_query(state)
                results = self.search_documents(refined_query, state)
                # Merge with existing results, deduplicating by chunk id
                existing_ids = {doc.id for doc, _ in state.retrieved_documents}
                for doc, score in results:
                    if doc.id not in existing_ids:
                        state.retrieved_documents.append((doc, score))
                        existing_ids.add(doc.id)

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

    def run_with_streaming(self, query: str, chat_history: Optional[List[Dict]] = None, state_ref: Optional[Dict] = None):
        """Run the agent with streaming response.

        Yields text chunks as they arrive. When finished, populates state_ref['state']
        so the caller can access citations and reasoning after the generator is exhausted.
        """
        state = AgentState(original_query=query, chat_history=chat_history or [])

        while state.iterations < self.max_iterations:
            state.iterations += 1
            action = self.decide_action(state)

            if action == ActionType.SEARCH:
                state.retrieved_documents = self.search_documents(query, state)

            elif action == ActionType.REFINE_QUERY:
                refined_query = self.refine_query(state)
                results = self.search_documents(refined_query, state)
                existing_ids = {doc.id for doc, _ in state.retrieved_documents}
                for doc, score in results:
                    if doc.id not in existing_ids:
                        state.retrieved_documents.append((doc, score))
                        existing_ids.add(doc.id)

            elif action == ActionType.RERANK:
                self.rerank_documents(state)

            elif action == ActionType.GENERATE_ANSWER:
                break

        context = self._prepare_context(state.retrieved_documents) if state.retrieved_documents else ""
        history_block = self._format_chat_history(state.chat_history)

        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

{history_block}Relevant Documents:
{context}

User Question: {state.original_query}

Please answer the question based on the provided documents.
Include specific references like [Source: document_name] for any claims.
If information is not in the documents, say so."""

        state.reasoning_history.append("Generating answer from retrieved documents...")
        full_answer = ""
        for chunk in self.llm.generate_with_streaming(prompt):
            full_answer += chunk
            yield chunk

        state.final_answer = full_answer
        state.citations = self._extract_citations(state.retrieved_documents)

        if state_ref is not None:
            state_ref["state"] = state
