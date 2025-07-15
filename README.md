# 🧠 Paper Critic Agent (Hybrid LLM)

A lightweight, agentic RAG project that critiques research papers (PDFs) using multiple LLM backends. Choose between local models (Hugging Face), free APIs (Groq), or Ollama. No LangChain, no LlamaIndex — just pure Python.

## 🚀 Features

- 🧾 Extracts **Abstract** and **Method** from any PDF
- 🔍 Retrieves **similar papers** using the arXiv API
- 🧠 Uses **hybrid LLM backends** to critically evaluate the method section
- 🔁 Includes a **self-reflection step** to revise the critique
- 🔄 **Auto-fallback** between different LLM providers

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

## 📦 Setup

1. **Install base dependencies**:
```bash
pip install requests PyPDF2
```

2. **Choose your LLM backend** (see options above)

3. **Run the script**:
```bash
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

## ✅ Next Steps

- [ ] 🔌 Add XML parsing of arXiv output to extract metadata
- [ ] 📊 Use vector DB (like FAISS) for deeper similarity comparison
- [ ] 🖼 Add Streamlit UI to upload a paper and display critique
- [ ] 🧠 Let LLM generate search query from abstract
- [ ] 📑 Save critique as a formatted PDF or Markdown report
