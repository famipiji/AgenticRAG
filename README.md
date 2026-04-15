# 🤖 Agentic RAG System

A powerful, production-ready Retrieval-Augmented Generation (RAG) system that uses intelligent agent orchestration to query technical documents and provide accurate, cited answers.

## 🎯 Overview

This project demonstrates a real-world LLM + RAG implementation with:

- **Intelligent Agent Orchestration**: Agent decides how to retrieve, when to refine queries, and what sources are relevant
- **Multi-Strategy Retrieval**: Semantic search + keyword matching + re-ranking
- **Query Rewriting**: Automatically generates alternative query phrasings for better retrieval
- **Source Citation**: Every answer includes proper citations with relevance scores
- **Support for Multiple Document Formats**: PDF, DOCX, TXT, Markdown
- **Vector Database**: FAISS-based semantic search with embeddings
- **Multiple LLM Providers**: Groq, OpenAI, and extensible for others
- **Beautiful Web Interface**: Streamlit-based UI for interactive querying

## 🏗️ System Architecture

```
User Query
    ↓
[Agent Layer] → Decides: Search? Refine Query? Re-rank?
    ↓
[Retrieval Layer] → Semantic + Keyword Search → Top-K Results
    ↓
[Re-ranking Layer] → LLM-based relevance scoring
    ↓
[Reasoning Layer] → LLM combines context + generates answer
    ↓
[Citation Layer] → Attaches sources with relevance scores
    ↓
Final Answer with Citations
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project
cd AgenticRAG

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Required:
#   - GROQ_API_KEY (if using Groq)
#   - OPENAI_API_KEY (if using OpenAI)
```

### 3. Add Documents

Place your documents in `data/documents/`:

```bash
data/
└── documents/
    ├── aspnet-core-8.md          # Sample documents included
    ├── ml-guide.md
    ├── rest-api-guide.md
    └── your-documents/           # Add your own
        ├── api-docs.pdf
        ├── research-paper.docx
        └── knowledge-base.txt
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.md`

### 4. Run the Web Interface

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`

### 5. Or Use the Python API

```bash
python demo.py
```

## 📊 Project Structure

```
AgenticRAG/
├── src/
│   ├── config.py                 # Configuration settings
│   ├── document_loader.py        # Document loading and chunking
│   ├── retrieval/
│   │   └── vector_db.py          # FAISS vector database
│   ├── llm/
│   │   └── llm_integration.py    # LLM providers (Groq, OpenAI)
│   └── agents/
│       └── rag_agent.py          # Agent orchestration logic
├── data/
│   ├── documents/                # Your documents go here
│   ├── chunks/                   # Processed chunks (auto-generated)
│   └── vectordb/                 # Vector database (auto-generated)
├── tests/                        # Unit tests
├── main.py                       # Streamlit web interface
├── demo.py                       # Command-line demo
├── requirements.txt              # Dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## 🧠 How It Works

### Agent Decision Making

The agent orchestrates RAG through intelligent decisions:

1. **Search**: Queries vector database using semantic search
2. **Refine Query**: Generates alternative phrasings if initial results aren't sufficient
3. **Re-rank**: Uses LLM to judge relevance of retrieved documents
4. **Generate Answer**: Combines top documents and generates response with citations

### Retrieval Strategy

- **Semantic Search**: Uses sentence-transformers to embed queries and documents
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Re-ranking**: LLM-based relevance scoring for better ordering
- **Citation Tracking**: Maintains source information throughout the pipeline

### Query Processing

```
Original Query: "What are the key features of ASP.NET Core 8?"
        ↓
[Query Rewriting]
- "What are the main capabilities of ASP.NET Core 8?"
- "List the features introduced in ASP.NET Core 8"
- "Key improvements in ASP.NET Core 8"
        ↓
[Multi-Query Search]
- Search with each query variant
- Combine results
        ↓
[Re-ranking & Answering]
```

## 🎮 Usage Examples

### Web Interface

1. Click "Load Documents" in sidebar to index documents
2. Type your question in the input field
3. View the answer with citations and source documents
4. Check "Agent Reasoning" for debug information

### Python API

```python
from src.document_loader import DocumentLoader
from src.retrieval.vector_db import VectorDatabase
from src.agents.rag_agent import RAGAgent

# Load and chunk documents
loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
chunks = loader.process_documents("data/documents")

# Create vector database
vector_db = VectorDatabase()
vector_db.add_documents(chunks)

# Initialize agent
agent = RAGAgent(
    vector_db=vector_db,
    llm_provider="groq",
    use_query_rewriting=True,
    use_reranking=True
)

# Query
answer, state = agent.run("What are the key features?")
print(answer)
print(state.citations)  # See sources
```

## 🔌 LLM Providers

### Groq (Recommended for Free Tier)

- Fast inference with MoE models
- Mixtral 8x7B model
- Free tier available

```bash
# Set in .env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
LLM_MODEL=mixtral-8x7b-32768
```

### OpenAI

- Multiple models available
- Production-ready
- Requires paid API key

```bash
# Set in .env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-3.5-turbo
```

## 📈 Performance Features

### Query Optimization

- **Query Rewriting**: Improves retrieval by generating semantically similar queries
- **Re-ranking**: LLM-based relevance scoring reorders results
- **Hybrid Search**: Combines semantic and keyword-based matching

### Database Optimization

- **Vector Indexing**: FAISS flat index with L2 distance
- **Chunk Overlap**: Maintains context between chunks
- **Efficient Storage**: JSON metadata storage with FAISS binaries

## 🎯 Key Differentiators from Simple RAG

| Aspect | Simple RAG | Agentic RAG |
|--------|-----------|-----------|
| Query Processing | Single search | Query rewriting + multi-search |
| Retrieval | Fixed top-k | Adaptive based on quality |
| Document Ranking | Similarity score only | LLM-based re-ranking |
| Answer Generation | Template-based | Reasoning-based |
| Citations | Basic | Rich with relevance scores |
| Intelligence | Lookup-based | Decision-making based |

## 🧪 Testing

```bash
# Run specific demo
python demo.py

# The demo shows:
# 1. Document loading and chunking
# 2. Vector database operations
# 3. Agent orchestration and answering
```

## 📚 Configuration Options

Edit `src/config.py` to customize:

```python
# Document Processing
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# Retrieval
TOP_K = 5                  # Number of results to retrieve
SIMILARITY_THRESHOLD = 0.3 # Minimum similarity score

# Agent
MAX_ITERATIONS = 5         # Max agent iterations
USE_QUERY_REWRITING = True # Enable query rewriting
USE_RE_RANKING = True      # Enable re-ranking

# LLM
TEMPERATURE = 0.3          # Lower = more focused
MAX_TOKENS = 2000          # Max response length
```

## 🔒 Security Considerations

- Store API keys in `.env` file (never in code)
- Use `.env.example` in version control
- Validate and sanitize user inputs
- Implement rate limiting for production
- Use HTTPS for APIs
- Rotate credentials regularly

## 🚀 Production Deployment

For production use:

1. **Database**: Use persistent storage or cloud vector DB
2. **Caching**: Implement Redis for frequent queries
3. **Monitoring**: Add logging and metrics
4. **Scaling**: Deploy with load balancing
5. **Security**: Implement authentication
6. **Rate Limiting**: Restrict API usage
7. **Async Processing**: Use task queues for large documents

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional document formats (Excel, HTML, etc.)
- More LLM providers
- Advanced retrieval strategies
- Fine-tuned embedding models
- Performance optimizations
- Additional vector DB support (Chroma, Weaviate)

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- LangChain for LLM orchestration
- FAISS for efficient vector search
- Sentence Transformers for embeddings
- Streamlit for the web interface
- Groq for fast inference

## 📞 Support

For issues or questions:
1. Check the documentation above
2. Review demo.py for usage examples
3. Examine agent reasoning in the UI
4. Check configuration in src/config.py

---

**Built with ❤️ for intelligent document querying**
