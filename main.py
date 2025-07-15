import subprocess
import requests
import os
import re
import tempfile
from urllib.parse import urlparse
from PyPDF2 import PdfReader

# Optional imports for different backends
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Hugging Face Transformers not installed. Install with: pip install transformers torch")


def extract_arxiv_id(url_or_id):
    """
    Extract arXiv ID from various URL formats or return the ID if already clean.
    
    Examples:
    - https://arxiv.org/abs/2301.07041 -> 2301.07041
    - https://arxiv.org/pdf/2301.07041.pdf -> 2301.07041
    - 2301.07041 -> 2301.07041
    """
    if not url_or_id:
        return None
    
    # If it's already just an ID, return it
    arxiv_id_pattern = r'^\d{4}\.\d{4,5}(v\d+)?$'
    if re.match(arxiv_id_pattern, url_or_id):
        return url_or_id
    
    # Extract from URL
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?',
        r'(\d{4}\.\d{4,5}(?:v\d+)?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    return None

def download_arxiv_pdf(arxiv_id):
    """
    Download PDF from arXiv and return the temporary file path.
    """
    if not arxiv_id:
        raise ValueError("Invalid arXiv ID")
    
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"📥 Downloading PDF from {pdf_url}...")
    
    response = requests.get(pdf_url)
    response.raise_for_status()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.write(response.content)
    temp_file.close()
    
    print(f"✅ Downloaded to temporary file: {temp_file.name}")
    return temp_file.name

def get_arxiv_metadata(arxiv_id):
    """
    Get paper metadata from arXiv API.
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    response.raise_for_status()
    
    # Basic XML parsing to extract title and abstract
    xml_content = response.text
    
    title_match = re.search(r'<title>(.*?)</title>', xml_content, re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Unknown Title"
    
    summary_match = re.search(r'<summary>(.*?)</summary>', xml_content, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else ""
    
    return {
        'title': title,
        'abstract': summary,
        'arxiv_id': arxiv_id
    }

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


def run_agent(arxiv_url_or_pdf_path):
    """
    Main agent function that can handle either arXiv URLs or local PDF paths.
    """
    pdf_path = None
    temp_file = False
    
    try:
        # Check if input is an arXiv URL/ID
        arxiv_id = extract_arxiv_id(arxiv_url_or_pdf_path)
        
        if arxiv_id:
            print(f"🔍 Detected arXiv ID: {arxiv_id}")
            
            # Get metadata from arXiv
            print("📋 Fetching paper metadata...")
            metadata = get_arxiv_metadata(arxiv_id)
            print(f"📄 Title: {metadata['title']}")
            
            # Download PDF
            pdf_path = download_arxiv_pdf(arxiv_id)
            temp_file = True
            
            # Use arXiv abstract if PDF extraction fails
            arxiv_abstract = metadata['abstract']
        else:
            # Assume it's a local PDF path
            pdf_path = arxiv_url_or_pdf_path
            arxiv_abstract = None
            print(f"📁 Using local PDF: {pdf_path}")
        
        print("\n🤖 Step 1: Extracting abstract & method...")
        abstract, method = extract_pdf_sections(pdf_path)
        
        # Fallback to arXiv abstract if PDF extraction failed
        if not abstract and arxiv_abstract:
            abstract = arxiv_abstract
            print("📋 Using arXiv abstract (PDF extraction failed)")
        
        print(f"📝 Abstract length: {len(abstract)} chars")
        print(f"📝 Method length: {len(method)} chars")
        
        if not method:
            print("⚠️  No method section found. Trying alternative section names...")
            # Try alternative method section names
            reader = PdfReader(pdf_path)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            
            alt_patterns = ["Methodology", "Approach", "Methods", "Implementation"]
            for pattern in alt_patterns:
                if pattern in text:
                    method = text.split(pattern)[1].split("Results")[0].strip() if "Results" in text else text.split(pattern)[1][:1000]
                    if method:
                        print(f"✅ Found {pattern} section")
                        break
        
        print(f"\n🤖 Step 2: Retrieving similar papers from arXiv...")
        # Use paper title or abstract for better search
        search_query = metadata['title'][:50] if arxiv_id and metadata['title'] else "transformer"
        similar = retrieve_similar_papers(query=search_query)
        print("📚 Retrieved related papers.")
        
        print(f"\n🤖 Step 3: Critiquing the paper...")
        critique_prompt = f"""
You're an academic reviewer. Analyze this research method critically:

Paper Title: {metadata.get('title', 'Unknown') if arxiv_id else 'Local PDF'}
Abstract: {abstract[:500]}...

Method Section:
{method}

Questions to address:
- Is the methodology clearly explained?
- Are there any experimental flaws?
- Is the methodology sound?
- Are comparisons with baselines made?
- Any red flags or concerns?
- What are the strengths and weaknesses?

Be honest, detailed, and constructive in your critique.
"""
        critique = run_llm(critique_prompt)
        
        print(f"\n📝 Critique Result:\n")
        print(critique)
        
        # Optional self-review loop
        print(f"\n🔁 Reflecting and revising critique...\n")
        self_review_prompt = f"""
Review this critique and revise it for clarity and depth:
{critique}

Make it more structured and add specific recommendations for improvement.
"""
        revised = run_llm(self_review_prompt)
        print(revised)
        
    finally:
        # Clean up temporary file
        if temp_file and pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
            print(f"\n🧹 Cleaned up temporary file")


if __name__ == "__main__":
    # Example usage - you can pass either:
    # - arXiv URL: "https://arxiv.org/abs/2301.07041"
    # - arXiv ID: "2301.07041"  
    # - Local PDF: "sample_paper.pdf"
    
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Default example - replace with your preferred paper
        input_path = "https://arxiv.org/abs/2301.07041"  # GPT-4 paper
    
    run_agent(input_path)