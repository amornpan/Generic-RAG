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
                       │      Ollama       │
                       │   (LLM + Embed)   │
                       └───────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Docker** & **Docker Compose**
- **Miniconda** or **Anaconda**
- **Git**

### 1. Clone Repository

```bash
git clone <repository-url>
cd Generic-RAG
```

### 2. Setup Environment

#### Install Miniconda

**For Linux (Ubuntu/Debian):**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts and restart terminal
source ~/.bashrc
```

**For macOS (Intel):**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
# Follow the prompts and restart terminal
source ~/.zshrc  # or ~/.bash_profile
```

**For macOS (Apple Silicon M1/M2/M3):**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
# Follow the prompts and restart terminal
source ~/.zshrc  # or ~/.bash_profile
```

**For Windows:**
1. Download installer from: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
2. Double-click the downloaded file and follow the installation wizard
3. Check "Add Miniconda3 to my PATH environment variable" during installation
4. Open **Anaconda Prompt** or **Command Prompt** after installation

#### Create Environment and Install Dependencies

```bash
# Create conda environment
conda create -n generic_rag_env python=3.10 -y

# Activate environment
conda activate generic_rag_env

# Install core packages with conda (recommended for better compatibility)
conda install numpy pandas jupyter requests beautifulsoup4 -y

# Install PyTorch (choose based on your system)
# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# For GPU (CUDA 11.8) - if you have NVIDIA GPU
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install specialized packages with pip
pip install -r requirements.txt
```

### 3. Start Services

#### Install and Start Docker

**For Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Logout and login again
```

**For macOS:**
1. Download Docker Desktop from: https://docs.docker.com/desktop/install/mac-install/
2. Install and start Docker Desktop
3. Verify installation: `docker --version`

**For Windows:**
1. Download Docker Desktop from: https://docs.docker.com/desktop/install/windows-install/
2. Install Docker Desktop
3. Start Docker Desktop from Start menu
4. Verify installation: `docker --version`

#### Start OpenSearch

**For Linux/macOS:**
```bash
# Start OpenSearch
docker run -d \
  --name opensearch-node \
  -p 9200:9200 \
  -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "bootstrap.memory_lock=true" \
  -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" \
  -e "DISABLE_INSTALL_DEMO_CONFIG=true" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=65536:65536 \
  opensearch:2.11.1
```

**For Windows (PowerShell or Command Prompt):**
```cmd
docker run -d ^
  --name opensearch-node ^
  -p 9200:9200 ^
  -p 9600:9600 ^
  -e "discovery.type=single-node" ^
  -e "bootstrap.memory_lock=true" ^
  -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" ^
  -e "DISABLE_INSTALL_DEMO_CONFIG=true" ^
  -e "DISABLE_SECURITY_PLUGIN=true" ^
  opensearch:2.11.1
```

#### Setup Hybrid Search Pipeline

**For Linux/macOS:**
```bash
# Wait for OpenSearch to start (30-60 seconds)
sleep 60

# Setup hybrid search pipeline
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

**For Windows (PowerShell):**
```powershell
# Wait for OpenSearch to start
Start-Sleep -Seconds 60

# Setup hybrid search pipeline
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

#### Install and Start Ollama

**For Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

**For macOS:**
1. Download from: https://ollama.ai/download/mac
2. Install the .dmg file
3. Start Ollama from Applications or run: `ollama serve`

**For Windows:**
1. Download from: https://ollama.ai/download/windows
2. Install the .exe file
3. Open Command Prompt or PowerShell and run: `ollama serve`

#### Pull Models

```bash
# Pull models (same for all platforms)
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b

# Verify models are installed
ollama list
```

### 4. Initialize Data

```bash
# Create vector index
python embedding.py
```

### 5. Run Application

```bash
# Terminal 1: Start API server
python api.py

# Terminal 2: Start UI
streamlit run app.py
```

### 6. Access Application

- **Web UI**: http://localhost:8501
- **API Docs**: http://localhost:9000/docs
- **OpenSearch**: http://localhost:9200

## 📁 Project Structure

```
Generic-RAG/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── environment.yml          # Conda environment
├── .env                     # Environment variables
├── .gitignore              # Git ignore file
│
├── api.py                  # FastAPI backend server
├── app.py                  # Streamlit frontend
├── embedding.py            # Data indexing script
│
├── md_corpus/              # Knowledge base (Markdown files)
│   ├── 1.md               # German measles (หัดเยอรมัน)
│   ├── 2.md               # Cholera (อหิวาตกโรค)
│   ├── 44.md              # Cataract (ต้อกระจก)
│   └── 5555.md            # GERD (กรดไหลย้อน)
│
├── docs/                   # Documentation
│   ├── setup.md           # Detailed setup guide
│   ├── api.md             # API documentation
│   └── troubleshooting.md # Common issues
│
└── scripts/               # Utility scripts
    ├── start_services.sh  # Start all services
    ├── stop_services.sh   # Stop all services
    └── reset_index.sh     # Reset OpenSearch index
```

## 🛠️ Configuration

### Environment Variables (.env)

```env
# Conda Environment
CONDA_ENV_NAME=generic_rag_env

# OpenSearch Configuration
OPENSEARCH_ENDPOINT=http://localhost:9200
OPENSEARCH_INDEX=dg_md_index

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=qwen2.5:0.5b

# API Configuration
API_HOST=0.0.0.0
API_PORT=9000
```

### Models

| Component | Model | Purpose |
|-----------|--------|---------|
| **Embeddings** | `nomic-embed-text` | Convert text to vectors |
| **LLM** | `qwen2.5:0.5b` | Generate answers (fast, basic) |
| **LLM** | `qwen2.5:7b` | Generate answers (better quality) |
| **LLM** | `qwen2.5:14b` | Generate answers (best quality) |

## 📊 Current Knowledge Base

The system includes Thai medical information covering:

1. **หัดเยอรมัน (German Measles/Rubella)** - Symptoms, causes, treatment
2. **อหิวาตกโรค (Cholera)** - Bacterial infection causing severe diarrhea  
3. **ต้อกระจก (Cataract)** - Eye condition common in elderly
4. **กรดไหลย้อน (GERD)** - Gastroesophageal reflux disease

## 🔧 Customization

### Adding New Documents

1. Place markdown files in `md_corpus/` directory
2. Run `python embedding.py` to reindex
3. Restart the API server

### Changing Models

```bash
# Pull different models
ollama pull qwen2.5:7b          # Better quality
ollama pull llama2:13b          # Alternative LLM
ollama pull mxbai-embed-large   # Better embeddings

# Update .env file with new model names
```

### Customizing System Prompt

Edit the `system_prompt` variable in `app.py`:

```python
system_prompt = """คุณเป็นผู้เชี่ยวชาญเกี่ยวกับ [YOUR DOMAIN]
ในการตอบคำถาม:
1. ใช้เฉพาะข้อมูลที่ให้มาในผลการค้นหา
2. หากไม่มีข้อมูลเพียงพอ ให้ตอบว่า "ขออภัย ไม่มีข้อมูลเพียงพอ"
..."""
```

## 🧪 Testing

### API Testing

```bash
# Test search endpoint
curl -X POST "http://localhost:9000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "อาการของโรคหัดเยอรมัน"}'

# Test OpenSearch
curl -X GET "localhost:9200/_cluster/health"

# Test Ollama
curl http://localhost:11434/api/tags
```

### Performance Testing

```bash
# Test hybrid search performance
python -c "
import requests
import time
for i in range(10):
    start = time.time()
    r = requests.post('http://localhost:9000/search', 
                     json={'query': 'ต้อกระจกรักษาอย่างไร'})
    print(f'Query {i+1}: {time.time()-start:.2f}s')
"
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenSearch won't start**
   ```bash
   docker logs opensearch-node
   # Usually memory issue - increase Docker memory limit
   ```

2. **Ollama connection failed**
   ```bash
   ollama ps
   # Check if service is running
   ollama serve
   ```

3. **Conda environment issues**
   ```bash
   conda env remove -n generic_rag_env
   conda env create -f environment.yml
   ```

4. **Hybrid search not working**
   ```bash
   # Recreate search pipeline
   curl -X DELETE "localhost:9200/_search/pipeline/hybrid-search-pipeline"
   # Run setup script again
   ```

### Logs Location

- **OpenSearch**: `docker logs opensearch-node`
- **Ollama**: Check terminal where `ollama serve` is running
- **API**: Check terminal where `python api.py` is running
- **Streamlit**: Check terminal where `streamlit run app.py` is running

## 📈 Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB+ |
| **CPU** | 4 cores | 8+ cores |
| **Storage** | 10GB | 50GB+ |
| **GPU** | None | CUDA-compatible |

### Optimization Tips

1. **Use GPU**: Install CUDA-enabled PyTorch for faster embeddings
2. **Increase Memory**: Allocate more memory to OpenSearch
3. **Better Models**: Use larger models for better quality
4. **SSD Storage**: Use SSD for better I/O performance
5. **Load Balancing**: Use multiple API instances for high load

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .

# Lint code
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

## 🙏 Acknowledgments

- **LlamaIndex** - RAG framework
- **OpenSearch** - Vector database
- **Ollama** - Local LLM runtime
- **Streamlit** - Web interface
- **FastAPI** - API framework
- **เมดไทย (Medthai)** - Medical knowledge source

## 🔄 Changelog

### v1.0.0 (2024-XX-XX)
- Initial release
- Hybrid search implementation
- Thai medical knowledge base
- Streamlit UI
- FastAPI backend
- Docker deployment

---

Made with ❤️ for the AI community