# 🧠 Paper Critic Agent (Hybrid LLM)

A lightweight, agentic RAG project that critiques research papers using multiple LLM backends. **No more manual PDF downloads** — just paste an arXiv URL and get instant, detailed academic critiques. Choose between local models (Hugging Face), free APIs (Groq), or Ollama. No LangChain, no LlamaIndex — just pure Python.

## 🚀 Key Features

- 📎 **Direct arXiv Integration** - Paste URLs, get instant analysis
- 🤖 **Hybrid LLM Backends** - Auto-selects best available option
- 🧾 **Smart PDF Processing** - Extracts abstracts, methods, and metadata
- 🔍 **Contextual Paper Retrieval** - Finds related work using paper titles
- 🧠 **Academic-Grade Critiques** - Structured, detailed methodology reviews
- 🔁 **Self-Reflection Loop** - Revises and improves initial critiques
- 🔄 **Intelligent Fallbacks** - Never fails due to single backend issues
- 🧹 **Auto-Cleanup** - Manages temporary files automatically

## 🎯 LLM Backend Options

### Option 1: Groq API (Recommended - Fast & Free)
1. **Get free API key**: Sign up at [console.groq.com](https://console.groq.com)
2. **Set environment variable**:
```bash
export GROQ_API_KEY="your_api_key_here"
```

### Option 2: Hugging Face Transformers (Local & Private)
1. **Install transformers**:
```bash
pip install transformers torch
```

### Option 3: Ollama (Original)
1. **Install Ollama**: https://ollama.com/download
2. **Pull a model**:
```bash
ollama pull llama3
```

## 📦 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/josephbak/paper-critic-agent.git
cd paper-critic-agent

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Choose Your LLM Backend
Pick one of the options above (Groq recommended for beginners)

### 3. Run the Agent
```bash
# With arXiv URL (recommended)
python main.py "https://arxiv.org/abs/2301.07041"

# With arXiv ID
python main.py "2301.07041"

# With local PDF
python main.py "path/to/paper.pdf"

# Default (uses GPT-4 paper as example)
python main.py
```

The script will automatically detect which backend is available and use the best option.

## 🔧 Advanced Usage

You can also specify the backend manually:

```python
# Force specific backend
critique = run_llm(prompt, backend="groq")
critique = run_llm(prompt, backend="huggingface") 
critique = run_llm(prompt, backend="ollama")

# Use different models
critique = run_llm(prompt, backend="groq", model="mixtral-8x7b-32768")
critique = run_llm(prompt, backend="huggingface", model="microsoft/DialoGPT-large")
```

### Recommended Models by Backend:
- **Groq**: `llama3-8b-8192` (default), `mixtral-8x7b-32768`, `gemma-7b-it`
- **Hugging Face**: `microsoft/DialoGPT-medium` (default), `microsoft/DialoGPT-large`
- **Ollama**: `llama3` (default), `mistral`, `codellama`

## 📋 Example Output

```
🔍 Detected arXiv ID: 1706.03762
📋 Fetching paper metadata...
📄 Title: Attention Is All You Need

🤖 Step 1: Extracting abstract & method...
📝 Abstract length: 2259 chars
📝 Method length: 2143 chars

🤖 Step 2: Retrieving similar papers from arXiv...
📚 Retrieved related papers.

🤖 Step 3: Critiquing the paper...
📝 Critique Result:

**Methodology Clarity**: The methodology is partially clear...
**Experimental Flaws**: There are some concerns regarding experimental design...
**Methodology Soundness**: While the Transformer architecture is interesting...

🔁 Reflecting and revising critique...
[Detailed structured critique with specific recommendations]
```

## 🔒 Privacy & Security

- **Local Models (HF/Ollama)**: 100% private, nothing leaves your machine
- **Groq API**: Data sent to Groq servers (check their privacy policy)
- **arXiv Downloads**: Temporary files auto-deleted after processing
- **Git Safety**: Models stored outside project, won't be committed

## 🛠 Troubleshooting

**"Groq API Error"**: Check your API key in `.env` file
**"Ollama not found"**: Make sure Ollama service is running (`ollama serve`)
**"No method section found"**: Try papers with clear "Method" or "Methodology" sections
**"NumPy compatibility"**: Run `pip install "numpy<2"` if you see warnings

## ✅ Next Steps

- [ ] 🔌 Add XML parsing of arXiv output to extract metadata
- [ ] 📊 Use vector DB (like FAISS) for deeper similarity comparison
- [ ] 🖼 Add Streamlit UI to upload a paper and display critique
- [ ] 🧠 Let LLM generate search query from abstract
- [ ] 📑 Save critique as a formatted PDF or Markdown report
