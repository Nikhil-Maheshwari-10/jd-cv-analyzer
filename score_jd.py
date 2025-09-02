import os
import json
import litellm
from litellm import completion
from dotenv import load_dotenv
import PyPDF2
import timeit
import time
from typing import List, Dict, Tuple

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# litellm.enable_json_schema_validation = True

# Define the prompt template
PROMPT_TEMPLATE = """
You are an expert JD Reviewer & Evaluator. Your job is to analyze multiple job descriptions (JDs) against a given CV and generate a precise, data-driven matching score.

## **Task Instructions**
- Analyze the provided CV against the given JD.
- Use a **fixed step-by-step calculation process** to ensure repeatability.
- Ensure exact decimal precision (e.g., 87.56%)—no estimations or rounding.
- Return JSON output with only `overall_match_score`.
## **Scoring Methodology (Strict Formula)**

Follow this step-by-step method to compute the score:

# **Skill Match**  
    - Count how many required skills exist in the CV.  
    - Compute: (Matched Skills / Total Required Skills)  

# **Experience Match (0-100 scale)**  
    - Compute: (Min(CV experience, JD experience) / JD experience) × 100

# **Education Match**

# **Industry & Role Relevance**

# **Final Score Calculation**
   - Compute **Final Score** as the **average of the four components**:  
    ```python
    Match_score = (skill_match + experience_match + education_match + industry_match) / 4
    ```
   - Ensure strict **decimal precision** (e.g., `87.56`).  

---

## **Expected Output (No Additional Text)**
{
"Match_score": <Score in decimal format>
}

"""

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts and normalizes PDF text"""
    text = ""
    try:
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        # Normalize text for consistency
        return ' '.join(text.strip().split())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def process_jd(cv_text: str, jd_path: str) -> Dict:
    """Process one JD with error handling"""
    jd_text = extract_text_from_pdf(jd_path)
    if not jd_text:
        return {"jd_file": os.path.basename(jd_path), "score": 0.0, "error": "Empty JD"}
    
    prompt = f"{PROMPT_TEMPLATE}\nCV:\n{cv_text}\nJD:\n{jd_text}"
    
    try:
        time.sleep(4) 
        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            api_key=GEMINI_API_KEY,
            response_format={"type": "json_object"},
            temperature=0.1,  
            verbose = False,
            max_tokens=100
        )
        
        content = json.loads(response.choices[0].message.content)
        score = max(0.0, min(100.0, float(content["Match_score"])))
        
        return {
            "jd_file": os.path.basename(jd_path),
            "score": score,
            # "tokens": response.usage.dict()
        }
    except Exception as e:
        return {"jd_file": os.path.basename(jd_path), "score": 0.0, "error": str(e)}

def score_jds(jd_paths: List[str], cv_path: str) -> Tuple[List[Dict], Dict]:
    """Score all JDs against one CV with token tracking"""
    cv_text = extract_text_from_pdf(cv_path)
    if not cv_text:
        raise ValueError("CV text extraction failed")
    
    results = []
    # total_tokens = {"input": 0, "output": 0}
    
    # Process each JD in sequence
    for jd_path in jd_paths:
        result = process_jd(cv_text, jd_path)
        results.append(result)
        
        # if "tokens" in result:
        #     total_tokens["input"] += result["tokens"]["prompt_tokens"]
        #     total_tokens["output"] += result["tokens"]["completion_tokens"]
        
        # Print progress
        print(f"Processed {result['jd_file']} → {result['score']:.2f}%")
    
    return sorted(results, key=lambda x: x["score"], reverse=True),# total_tokens

if __name__ == "__main__":
    jd_paths = [
        
        "Jd_ML-1.pdf", 
        "Jd_ML-2.pdf",
        "Jd_Ai.pdf",
        "Jd_Ml_senior.pdf",
        "Jd_Ml_gpt.pdf"
    ]
    cv_path = "senior-machine-learning-4.pdf"
    
    print("🚀 Starting JD evaluation...")
    start_time = timeit.default_timer()
    
    try:
        results, tokens = score_jds(jd_paths, cv_path)
        
        print("\n🏆 Final Rankings:")
        for i, res in enumerate(results, 1):
            status = "" if "error" not in res else f" | Error: {res['error']}"
            print(f"{i:>2}. {res['jd_file']:<25} {res['score']:>6.2f}%{status}")
        
        print(f"\n📊 Token Usage: Input={tokens['input']} | Output={tokens['output']} | Total={tokens['input'] + tokens['output']}")
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
    finally:
        print(f"\n⏱️ Total time: {timeit.default_timer() - start_time:.2f}s")