# 🏗️ System Architecture

This document explains the architecture and design of the Agentic RAG System.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│              (Streamlit Web App / CLI / Python API)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT ORCHESTRATION LAYER                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ QueryRewriter│  │ DocumentRanker│ │ RAG Agent    │          │
│  │              │  │               │ │              │          │
│  │ - Rewrite    │  │ - Re-rank     │ │ - Route      │          │
│  │   queries    │  │   documents   │ │ - Decide     │          │
│  └──────────────┘  └──────────────┘ │ - Execute    │          │
│                                      └──────────────┘          │
└────────────────────────┬───────────────┬──────────────────────┘
                         │               │
                    ┌────▼────┐    ┌────▼─────┐
                    │ Retrieval│    │ LLM      │
                    │ System   │    │ Integration
                    └────┬────┘    └────┬─────┘
                         │              │
         ┌───────────────┬┴──────────────┴───────────┐
         │               │                           │
         ▼               ▼                           ▼
    ┌─────────────┐  ┌──────────────┐    ┌────────────────┐
    │Vector DB    │  │ Document     │    │LLM Providers   │
    │(FAISS)      │  │ Loader       │    │ - Groq         │
    │             │  │              │    │ - OpenAI       │
    │ - Index     │  │ - Chunk      │    │ - Extensible   │
    │ - Search    │  │ - Parse      │    └────────────────┘
    │ - Re-rank   │  │ - Store      │
    └─────────────┘  └──────────────┘
         │               │
    ┌────▼────────────────▼─────┐
    │ Data Layer                │
    │                           │
    │ - Documents (/data)       │
    │ - Vector Index            │
    │ - Chunk Store             │
    │ - Embeddings              │
    └─────────────────────────┘
```

## Component Architecture

### 1. User Interface Layer

**Streamlit Web App** (`main.py`)
- Interactive query interface
- Document management
- Chat history
- Source visualization
- Agent reasoning display

**Command Line** (`demo.py`)
- Programmatic access
- Batch operations
- Integration testing

**Python API**
- Direct component access
- Custom workflows
- Production integration

### 2. Agent Orchestration Layer

```python
RAGAgent
├── decide_action()
│   ├── Search
│   ├── Refine Query
│   ├── Re-rank
│   ├── Generate Answer
│   └── Stop
│
├── QueryRewriter
│   └── rewrite_query()    # Generate alternatives
│
├── DocumentRanker
│   └── rank_documents()   # LLM-based re-ranking
│
└── run()                  # Main orchestration loop
    ├── Iterate up to MAX_ITERATIONS
    ├── Execute actions in sequence
    └── Generate final answer with citations
```

**Decision Flow:**
```
Start
  ↓
Documents Retrieved?
  ├─ No → Search (iteration 1) / Refine Query (iteration 2+)
  └─ Yes → Ready to rank?
           ├─ No → Re-rank (if enabled)
           └─ Yes → Generate Answer → Stop
```

### 3. Retrieval System

**Vector Database** (`vector_db.py`)
- FAISS-based semantic search
- Hybrid search (semantic + keyword)
- Document re-ranking
- Citation tracking

**Retrieval Pipeline:**
```
Query
  ↓
Embed Query (sentence-transformers)
  ↓
FAISS Search (L2 distance)
  ↓
Convert to Similarity Scores
  ↓
Keyword Matching
  ↓
Combine Scores (70% semantic + 30% keyword)
  ↓
Top-K Results
```

### 4. Document Processing Layer

**Document Loader** (`document_loader.py`)
- Multi-format support (PDF, DOCX, TXT, MD)
- Intelligent chunking with overlap
- Metadata extraction
- Persistent chunk storage

**Processing Pipeline:**
```
Documents
  ↓
Load by Format
  ├─ PyPDF2 for PDF
  ├─ python-docx for DOCX
  └─ File I/O for TXT/MD
  ↓
Extract Text
  ↓
Chunk with Overlap
  ├─ Chunk Size: 1000 chars
  ├─ Overlap: 200 chars
  └─ Preserve Context
  ↓
Generate Embeddings
  ↓
Store in Vector DB
```

### 5. LLM Integration Layer

**LLM Provider System** (`llm_integration.py`)
- Abstract provider interface
- Multiple implementations
- Consistent API across providers

```python
LLMProvider (Abstract)
├── GroqLLM
│   └── Uses: Mixtral 8x7B
├── OpenAILLM
│   └── Uses: GPT-3.5-turbo / GPT-4
└── Extensible for others
    └── Anthropic, LLaMA, etc.
```

**Query Flow:**
```
System Prompt + Context + Question
  ↓
Send to LLM
  ↓
Temperature: 0.3 (focused answers)
Max Tokens: 2000
  ↓
Stream Response
  ↓
Extract Citations
```

## Data Flow

### Complete Query Processing Flow

```
1. USER INPUT
   Query: "What are key features?" 
   ↓
2. AGENT DECIDES
   → Search (first iteration)
   ↓
3. QUERY REWRITING (if enabled)
   Original: "What are key features?"
   Variants:
   - "List main features"
   - "Describe capabilities"
   - "What are the benefits?"
   ↓
4. MULTI-QUERY SEARCH
   Search #1: Original query
   Search #2: Variant 1
   Search #3: Variant 2
   ↓
5. COMBINE & HYBRID SEARCH
   - Semantic similarity (embedding distance)
   - Keyword matching
   - Combine scores (70/30 weighting)
   ↓
6. RE-RANKING (if enabled)
   LLM judges relevance:
   "Given query X, rank documents by relevance"
   ↓
7. TOP-K SELECTION
   Select top 5 documents (configurable)
   ↓
8. CONTEXT PREPARATION
   Combine documents into context
   Add relevance scores and source info
   ↓
9. ANSWER GENERATION
   Prompt: System + Query + Context
   → LLM generates answer
   → Extracts citations
   ↓
10. OUTPUT
    Answer: Structured response
    Citations: Source documents with scores
    Reasoning: Agent decision history (optional)
```

## Memory and State Management

### Session State (Streamlit)
```python
session_state = {
    "vector_db": VectorDatabase,
    "agent": RAGAgent,
    "chat_history": [
        {"question": str, "answer": str, "citations": [...]},
        ...
    ],
    "db_loaded": bool
}
```

### Agent State
```python
AgentState = {
    "original_query": str,
    "refined_queries": [str],
    "iterations": int,
    "retrieved_documents": [(Document, score)],
    "reasoning_history": [str],
    "final_answer": str,
    "citations": [{"source": str, "relevance": float}]
}
```

## Configuration & Customization

### Key Configuration Points

**In `src/config.py`:**
- Document chunking strategy
- Embedding model selection
- Vector DB parameters
- Agent behavior (query rewriting, re-ranking)
- LLM provider and model selection
- Temperature and token limits

**Runtime Override:**
```python
agent = RAGAgent(
    vector_db=vector_db,
    llm_provider="groq",
    max_iterations=5,           # Control search depth
    top_k=3,                    # Fewer = faster
    use_query_rewriting=True,   # Enable/disable
    use_reranking=True          # Enable/disable
)
```

## Performance Characteristics

### Time Complexity
```
Document Loading:      O(n × m) where n=docs, m=avg_size
Embedding Generation:  O(n × d) where d=embedding_dim
FAISS Search:          O(log n) to O(n) depending on index
Re-ranking:            O(k × query_processing)
Answer Generation:     O(1) LLM call (variable time)

Total per Query:       O(n) for indexing + O(1) for search + LLM time
```

### Space Complexity
```
Documents:             O(n × m)
FAISS Index:           O(n × d)
Metadata:              O(n × m_size)
Session Cache:         O(k) where k=top-k
```

## Extensibility Points

### Add Custom Retrieval Strategy
```python
class CustomRetriever:
    def search(self, query, top_k):
        # Your logic here
        return results
```

### Add New LLM Provider
```python
class CustomLLM(LLMProvider):
    def generate(self, prompt, **kwargs):
        # Call your LLM API
        return response
```

### Add Document Format
```python
def load_custom_format(self, file_path):
    # Your parsing logic
    return text
```

## Security Considerations

### Input Validation
- Query validation in agent
- Document format validation
- API key environment variable isolation

### API Key Management
- Stored in `.env` only
- Never logged or exposed
- Rotatable per deployment

### Data Privacy
- No data sent to external services except LLM
- Local vector database
- Optional encryption of stored data

## Deployment Architecture

### Local Deployment
```
Single Server
├── Streamlit App
├── Vector DB (FAISS)
└── LLM API calls
```

### Production Deployment
```
Load Balancer
├─ Web Server (Streamlit/FastAPI)
├─ Cache Layer (Redis)
├─ Vector DB Server (Weaviate/Milvus)
└─ Queue System (Celery)
    └─ LLM API
```

## Monitoring & Logging

### Key Metrics
- Query processing time
- Retrieval accuracy
- LLM response quality
- Token usage and costs
- Cache hit rate
- Error rates by component

### Logging Integration Points
- Document processing events
- Vector DB operations
- Agent decisions
- LLM API calls
- Error stack traces

---

**This architecture ensures scalability, maintainability, and extensibility.**
