# 🚀 Quick Start Guide

Get up and running with Agentic RAG in 5 minutes!

## Prerequisites

- Python 3.8+
- pip
- API key from Groq or OpenAI

## Step 1: Setup (2 minutes)

```bash
# Navigate to project
cd AgenticRAG

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API key
# For Groq (recommended): https://console.groq.com
#   - Set: GROQ_API_KEY=...
# For OpenAI:
#   - Set: OPENAI_API_KEY=...
#   - Set: LLM_PROVIDER=openai
```

## Step 3: Add Documents (1 minute)

The package includes 3 sample documents:
- ASP.NET Core 8 documentation
- Machine Learning guide
- REST API best practices

**To add your own documents:**

```bash
# Copy your files to data/documents/
# Supported: .pdf, .docx, .txt, .md
cp your-document.pdf data/documents/
```

## Step 4: Launch (1 minute)

### Option A: Web Interface (Recommended)

```bash
streamlit run main.py
```

Then:
1. Click "Load Documents" button
2. Type your question
3. Click "Search"
4. View answer and sources!

### Option B: Command Line

```bash
python demo.py
```

This runs a comprehensive demo with example queries.

### Option C: Python API

```python
from src.document_loader import DocumentLoader
from src.retrieval.vector_db import VectorDatabase
from src.agents.rag_agent import RAGAgent

# Setup
loader = DocumentLoader()
chunks = loader.process_documents("data/documents")
vector_db = VectorDatabase()
vector_db.add_documents(chunks)
agent = RAGAgent(vector_db=vector_db)

# Query
answer, state = agent.run("Your question here?")
print(answer)
```

## Utility Commands

```bash
# Build vector database
python utils.py build

# Rebuild from scratch
python utils.py rebuild --force

# Show database stats
python utils.py stats

# List all documents
python utils.py list

# Search database
python utils.py search "your query"
```

## Try These Example Questions

With the included sample documents:

1. **"What are the key features of ASP.NET Core 8?"**
   - Tests retrieval from documentation

2. **"How should I design REST API endpoints?"**
   - Tests REST best practices retrieval

3. **"What are the types of machine learning and give examples?"**
   - Tests multi-document retrieval with reasoning

## Troubleshooting

### "No documents found"
- Check that `data/documents/` has files
- Supported formats: `.pdf`, `.docx`, `.txt`, `.md`
- Run: `python utils.py list`

### "API key error"
- Verify `.env` file exists
- Check API key is correct
- For Groq: Visit https://console.groq.com
- For OpenAI: Visit https://platform.openai.com

### "FAISS/embedding errors"
- Try: `pip install --upgrade faiss-cpu sentence-transformers`
- On Windows: May need Visual C++ runtime

### "Out of memory"
- Reduce `CHUNK_SIZE` in `src/config.py`
- Use smaller embedding model
- Process fewer documents

## Next Steps

1. **Customize Configuration**
   - Edit `src/config.py` for agent behavior
   - Adjust chunk sizes for your documents
   - Change LLM models and parameters

2. **Production Deployment**
   - Set up proper logging
   - Use persistent vector database storage
   - Implement rate limiting
   - Add authentication

3. **Advanced Features**
   - Fine-tune embedding model
   - Add more document formats
   - Implement advanced retrieval strategies
   - Add document versioning

4. **Performance Optimization**
   - Implement caching for frequent queries
   - Use GPU for embeddings
   - Implement streaming for large answers
   - Profile and optimize bottlenecks

## Tips & Tricks

### 🎯 Better Answers
- Use specific, detailed questions
- Include context from previous answers
- Break complex questions into steps

### ⚡ Faster Processing
- Smaller chunk sizes = faster retrieval but less context
- Disable query rewriting if speed is critical
- Use Groq for faster inference than OpenAI

### 📊 Better Debugging
- Enable reasoning history in UI
- Use `python utils.py search "query"` to test retrieval
- Check `data/chunks/chunks.jsonl` for processed chunks

### 🔧 Custom Setup
```python
from src.agents.rag_agent import RAGAgent

# Customize agent behavior
agent = RAGAgent(
    vector_db=vector_db,
    llm_provider="groq",
    max_iterations=3,      # Decrease for speed
    top_k=3,              # Fewer docs = faster
    use_query_rewriting=True,
    use_reranking=False   # Skip for speed
)
```

## Common Use Cases

### 📖 Internal Knowledge Base
- Add company docs: SOPs, policies, FAQs
- Query with: "How do I...?" questions
- Great for onboarding and support

### 🔬 Research Paper Analysis
- Upload research papers as PDFs
- Ask: "What are the key findings?"
- Extract specific information from papers

### 📚 API Documentation
- Add API docs and guides
- Query: "How do I call endpoint X?"
- Get code examples and best practices

### 🎓 Learning Resource
- Add textbook chapters
- Ask definition and concept questions
- Get explained answers with citations

## Performance Benchmarks

On typical documents (all sample docs indexed):

- Document loading: ~2-5 seconds
- Query processing: ~3-10 seconds (includes LLM inference)
- First query: Slightly slower due to model loading
- Subsequent queries: Faster due to caching

*Times vary based on document size, LLM provider, and hardware*

## Support

For detailed information, see:
- 📋 [README.md](README.md) - Full documentation
- 🧪 [demo.py](demo.py) - Working examples
- ⚙️ [src/config.py](src/config.py) - All configuration options

---

**You're ready to go! 🚀 Happy querying! 🤖**
