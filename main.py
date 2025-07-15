import subprocess
import requests
import os
from PyPDF2 import PdfReader

# Optional imports for different backends
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Hugging Face Transformers not installed. Install with: pip install transformers torch")


def extract_pdf_sections(pdf_path):
    """
    Extract abstract and method section from a PDF file.
    Naively splits based on common section headers.
    """
    reader = PdfReader(pdf_path)
    text = "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

    abstract = text.split("Abstract")[1].split("Introduction")[0].strip() if "Abstract" in text else ""
    method = text.split("Method")[1].split("Results")[0].strip() if "Method" in text and "Results" in text else ""
    return abstract, method


def retrieve_similar_papers(query="transformer"):
    """
    Retrieves a list of related papers from arXiv using their public API.
    """
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"
    res = requests.get(url)
    return res.text  # You could parse this XML further for titles/abstracts


# Global variable to cache the HF model
_hf_generator = None

def run_llm(prompt, backend="auto", model="microsoft/DialoGPT-medium"):
    """
    Run a prompt through different LLM backends.
    
    Args:
        prompt: The text prompt to send to the LLM
        backend: "auto", "huggingface", "groq", or "ollama"
        model: Model name (varies by backend)
    
    Returns:
        Generated text response
    """
    global _hf_generator
    
    # Auto-select backend based on availability
    if backend == "auto":
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            backend = "groq"
        elif HF_AVAILABLE:
            backend = "huggingface"
        else:
            backend = "ollama"
            print("⚠️  Falling back to Ollama. Install transformers or set GROQ_API_KEY for other options.")
    
    try:
        if backend == "groq":
            return _run_groq(prompt, model)
        elif backend == "huggingface":
            return _run_huggingface(prompt, model)
        elif backend == "ollama":
            return _run_ollama(prompt, model)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except Exception as e:
        print(f"❌ Error with {backend}: {e}")
        # Fallback to next available option
        if backend != "ollama":
            print("🔄 Falling back to Ollama...")
            return _run_ollama(prompt, "llama3")
        raise

def _run_groq(prompt, model="llama3-8b-8192"):
    """Run prompt through Groq API"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def _run_huggingface(prompt, model="microsoft/DialoGPT-medium"):
    """Run prompt through Hugging Face Transformers locally"""
    global _hf_generator
    
    if not HF_AVAILABLE:
        raise ImportError("Hugging Face Transformers not available")
    
    # Initialize model once and cache it
    if _hf_generator is None:
        print(f"🤖 Loading {model} (this may take a moment on first run)...")
        _hf_generator = pipeline(
            "text-generation", 
            model=model,
            device_map="auto" if model != "microsoft/DialoGPT-medium" else None
        )
    
    # Generate response
    response = _hf_generator(
        prompt, 
        max_length=len(prompt.split()) + 200,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=_hf_generator.tokenizer.eos_token_id
    )
    
    # Extract just the new generated text
    generated = response[0]['generated_text']
    return generated[len(prompt):].strip()

def _run_ollama(prompt, model="llama3"):
    """Run prompt through Ollama (original implementation)"""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama failed: {result.stderr.decode()}")
    return result.stdout.decode()


def run_agent(pdf_path):
    print("🤖 Step 1: Extracting abstract & method...")
    abstract, method = extract_pdf_sections(pdf_path)

    print("\n🤖 Step 2: Retrieving similar papers from arXiv...")
    similar = retrieve_similar_papers(query="transformer")
    print("📚 Retrieved related papers.")

    print("\n🤖 Step 3: Critiquing the paper...")
    critique_prompt = f"""
You're an academic reviewer. Analyze this research method critically:
Method Section:
{method}

Questions to address:
- Is it clearly explained?
- Are there any experimental flaws?
- Is the methodology sound?
- Are comparisons with baselines made?
- Any red flags?

Be honest, detailed, and constructive.
"""
    critique = run_llm(critique_prompt)

    print("\n📝 Critique Result:\n")
    print(critique)

    # Optional self-review loop
    print("\n🔁 Reflecting and revising critique...\n")
    self_review_prompt = f"""
Review this critique and revise it for clarity and depth:
{critique}
"""
    revised = run_llm(self_review_prompt)
    print(revised)


if __name__ == "__main__":
    run_agent("sample_paper.pdf")  # Replace with your paper file
