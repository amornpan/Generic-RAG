# Generic RAG System 🔍🤖

A powerful **Retrieval-Augmented Generation (RAG)** system built with modern AI technologies, featuring hybrid search capabilities and Thai language support for health-related information.

## 🌟 Features

- **🔍 Hybrid Search**: Combines semantic search (vector) with keyword search (BM25) for optimal results
- **🧠 AI-Powered Q&A**: Uses Large Language Models for intelligent question answering
- **🌐 Web Interface**: User-friendly Streamlit interface with chat-like experience
- **🏥 Health Domain**: Pre-loaded with Thai medical knowledge base
- **⚡ Real-time**: Fast API backend with async processing
- **🐳 Containerized**: Easy deployment with Docker
- **🔧 Extensible**: Modular design for easy customization
- **💾 HuggingFace Embeddings**: Uses BAAI/bge-m3 model for high-quality embeddings

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◄──►│   FastAPI       │◄──►│   OpenSearch    │
│   (Frontend)    │    │   (Backend)     │    │   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                       ┌─────────▼─────────┐
                       │   HuggingFace     │
                       │  Embeddings +     │
                       │     Ollama LLM    │
                       └───────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Docker** & **Docker Compose**
- **Python 3.10+** (via Miniconda/Anaconda)
- **Git**
- **CUDA-compatible GPU** (optional, for faster embeddings)

### 1. Clone Repository

```bash
git clone https://github.com/amornpan/Generic-RAG.git
cd Generic-RAG
```

### 2. Setup Environment

#### Install Miniconda (if not already installed)

**For Windows:**
1. Download installer from: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
2. Install and check "Add Miniconda3 to my PATH environment variable"
3. Open new Command Prompt or PowerShell

**For Linux/macOS:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### Create Environment and Install Dependencies

```bash
# Create conda environment
conda create -n generic_rag_env python=3.10 -y
conda activate generic_rag_env

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Start Services

#### Start Docker (if not running)

Make sure Docker Desktop is running on your system.

#### Start OpenSearch

**For Windows (PowerShell/Command Prompt):**
```cmd
docker run -d --name opensearch-node -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "bootstrap.memory_lock=true" -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" -e "DISABLE_INSTALL_DEMO_CONFIG=true" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:2.11.1
```

**For Linux/macOS:**
```bash
docker run -d --name opensearch-node -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "bootstrap.memory_lock=true" -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" -e "DISABLE_INSTALL_DEMO_CONFIG=true" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:2.11.1
```

#### Setup Hybrid Search Pipeline

Wait for OpenSearch to start (30-60 seconds), then:

**For Windows (PowerShell):**
```powershell
Invoke-RestMethod -Uri "http://localhost:9200/_search/pipeline/hybrid-search-pipeline" `
  -Method PUT `
  -ContentType "application/json" `
  -Body '{
    "description": "Post processor for hybrid search",
    "phase_results_processors": [
      {
        "normalization-processor": {
          "normalization": {"technique": "min_max"},
          "combination": {
            "technique": "arithmetic_mean",
            "parameters": {"weights": [0.3, 0.7]}
          }
        }
      }
    ]
  }'
```

**For Linux/macOS:**
```bash
curl -X PUT "localhost:9200/_search/pipeline/hybrid-search-pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Post processor for hybrid search",
    "phase_results_processors": [
      {
        "normalization-processor": {
          "normalization": {"technique": "min_max"},
          "combination": {
            "technique": "arithmetic_mean",
            "parameters": {"weights": [0.3, 0.7]}
          }
        }
      }
    ]
  }'
```

#### Install and Start Ollama

1. Download and install Ollama from: https://ollama.ai/download
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull required model:
   ```bash
   ollama pull qwen2.5:0.5b
   ```

### 4. Initialize Data

```bash
# Create vector index with HuggingFace embeddings
python embedding.py
```

This will:
- Load markdown documents from `md_corpus/` directory
- Create embeddings using BAAI/bge-m3 model
- Store vectors in OpenSearch
- Save index to `md_index.pkl`

### 5. Run Application

```bash
# Terminal 1: Start API server
python api.py

# Terminal 2: Start Streamlit UI
streamlit run app.py
```

### 6. Access Application

- **Web UI**: http://localhost:8501
- **API Docs**: http://localhost:9000/docs
- **OpenSearch**: http://localhost:9200

The UI will show the status of all services (API, Ollama, OpenSearch) at the top.

## 📁 Project Structure

```
Generic-RAG/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables (optional)
│
├── embedding.py             # Data indexing with HuggingFace embeddings
├── api.py                   # FastAPI backend with HuggingFace embeddings
├── app.py                   # Streamlit frontend
│
├── md_corpus/               # Knowledge base (Markdown files)
│   ├── 1.md                # German measles (หัดเยอรมัน)
│   ├── 2.md                # Cholera (อหิวาตกโรค)
│   ├── 44.md               # Cataract (ต้อกระจก)
│   └── 5555.md             # GERD (กรดไหลย้อน)
│
└── md_index.pkl            # Saved index (created after running embedding.py)
```

## 🛠️ Configuration

### Environment Variables (.env)

Create a `.env` file (optional) to override defaults:

```env
# OpenSearch Configuration
OPENSEARCH_ENDPOINT=http://localhost:9200
OPENSEARCH_INDEX=dg_md_index

# API Configuration
API_HOST=0.0.0.0
API_PORT=9000
```

### Models

| Component | Model | Purpose | Notes |
|-----------|--------|---------|-------|
| **Embeddings** | `BAAI/bge-m3` | Convert text to vectors | Downloaded automatically (~2GB) |
| **LLM** | `qwen2.5:0.5b` | Generate answers | Must be pulled via Ollama |

## 📊 Current Knowledge Base

The system includes Thai medical information covering:

1. **หัดเยอรมัน (German Measles/Rubella)** - Symptoms, causes, treatment
2. **อหิวาตกโรค (Cholera)** - Bacterial infection causing severe diarrhea  
3. **ต้อกระจก (Cataract)** - Eye condition common in elderly
4. **กรดไหลย้อน (GERD)** - Gastroesophageal reflux disease

## 🔧 Customization

### Adding New Documents

1. Place markdown files in `md_corpus/` directory
2. Delete old index:
   ```bash
   curl -X DELETE "localhost:9200/dg_md_index"
   ```
3. Run `python embedding.py` to reindex
4. Restart the API server

### Changing Models

#### For Embeddings (in embedding.py and api.py):
```python
embedding_model_name = 'BAAI/bge-m3'  # Current model
# Can change to other HuggingFace models like:
# embedding_model_name = 'intfloat/multilingual-e5-large'
```

#### For LLM (in app.py):
```python
llm_model = "qwen2.5:0.5b"  # Current model
# Can change to:
# llm_model = "qwen2.5:7b"  # Better quality
# llm_model = "llama2:13b"  # Alternative
```

Remember to pull new Ollama models:
```bash
ollama pull qwen2.5:7b
```

## 🧪 Testing

### API Testing

```bash
# Test API health
curl http://localhost:9000/health

# Test search endpoint
curl -X POST "http://localhost:9000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "อาการของโรคหัดเยอรมัน"}'
```

### OpenSearch Collections Management

```bash
# View all indices
curl -X GET "localhost:9200/_cat/indices?v"

# View index details
curl -X GET "localhost:9200/dg_md_index?pretty"

# Count documents in index
curl -X GET "localhost:9200/dg_md_index/_count?pretty"

# View sample documents
curl -X GET "localhost:9200/dg_md_index/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "size": 5,
  "query": {
    "match_all": {}
  }
}'
```

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'xxx'" Error**
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenSearch connection failed**
   ```bash
   # Check if OpenSearch is running
   docker ps
   # Check OpenSearch health
   curl http://localhost:9200/_cluster/health
   ```

3. **Ollama not responding**
   ```bash
   # Make sure Ollama is running
   ollama serve
   # Check installed models
   ollama list
   ```

4. **No search results**
   ```bash
   # Check document count
   curl -X GET "localhost:9200/dg_md_index/_count"
   # If 0, rerun embedding.py
   python embedding.py
   ```

5. **GPU not detected**
   ```bash
   # Check PyTorch GPU support
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## 📈 Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **RAM** | 8GB | 16GB+ | HuggingFace models need memory |
| **CPU** | 4 cores | 8+ cores | - |
| **Storage** | 10GB | 50GB+ | For models and data |
| **GPU** | None | CUDA 11.8+ | Speeds up embeddings |

### Tips

1. **Use GPU**: Significantly faster for embeddings
2. **Batch Processing**: embedding.py processes in batches automatically
3. **Adjust Chunk Size**: In embedding.py, modify `chunk_size=1024`
4. **Use Larger LLM**: For better answers, use `qwen2.5:7b` or larger

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **LlamaIndex** - RAG framework
- **OpenSearch** - Vector database
- **Ollama** - Local LLM runtime
- **HuggingFace** - Embedding models
- **Streamlit** - Web interface
- **FastAPI** - API framework

---

Made with ❤️ for the AI community