import os
import json
import PyPDF2
from dotenv import load_dotenv
import litellm
from litellm import completion
import re
import io
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import timeit
import time

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

litellm.enable_json_schema_validation=True

# Load SentenceTransformer Model for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_input):
    """Extract text from a PDF file or a BytesIO object."""
    if isinstance(pdf_input, (str, os.PathLike)):  # If given a file path
        with open(pdf_input, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif isinstance(pdf_input, io.BytesIO):  # If given a BytesIO object
        pdf_input.seek(0)  # Ensure the pointer is at the start
        reader = PyPDF2.PdfReader(pdf_input)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        raise TypeError("Invalid pdf_input type. Expected file path or BytesIO object.")

    return text.strip()


def generate_json_from_text(text):
    """Use LLM to convert extracted text into structured JSON."""
    prompt = f"""
    
    "Please parse all data as much as possible from document as json, pls use meaningful key name in json and return in valid format, keep data in just on dictionary.
    Also give total experience in dedicated "total_experience" section by combining all given experiences in int or float years but dont add project years in total experience.
    if no experience or company name is given then return 0 years."

    Extracted Text:
    {text}
    """
    # time.sleep(4.1)
    response = completion(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("GOOGLE_API_KEY"),
        response_format={'type': 'json_object'}
        
    )
    # response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
    input_tokens = response['usage']['prompt_tokens']
    output_tokens = response['usage']['completion_tokens']
    
    

    # # Print token counts
    # print(f"Input Tokens: {input_tokens}")
    # print(f"Output Tokens: {output_tokens}")
    # print(f"Total Tokens: {total_tokens}")
    

    # print("Response",response.choices[0]['message']['content'])
    # Clean Markdown formatting
    # cleaned_response = re.sub(r"```json\n(.*?)\n```", r"\1", response.text.strip(), flags=re.DOTALL)
    
    json_data = json.loads(response.choices[0].message.content)  # Ensure JSON is valid
    # print (json_data)
    return json_data, input_tokens, output_tokens



def generate_embedding(text):
    """Generate semantic embeddings for a given text."""
    return embedding_model.encode(text).tolist()  # Convert tensor to list for storage


def match_jd_with_cvs(jd_text: str, cv_store: List[Tuple[str, dict, List[float]]]) -> List[Tuple[str, float]]:
    """Match JD with stored CVs using semantic similarity."""
    jd_embedding = generate_embedding(jd_text)

    scores = []
    for filename, cv_embedding in cv_store.items():
        # Compute cosine similarity
        similarity = np.dot(jd_embedding, cv_embedding) / (np.linalg.norm(jd_embedding) * np.linalg.norm(cv_embedding))
        scores.append((filename, similarity))
    
    # Sort by similarity score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # print(scores)
    return scores

def sort_top_cvs_with_llm(ranked_cvs: List[Tuple[str, float]], jd_text: str, cv_store: List[Tuple[str, dict, List[float]]]) -> dict:
    """Use LLM to sort the top CVs based on final ranking and provide insights."""
    
    # Get top 5 CVs
    top_cvs = ranked_cvs[:5]
    cv_data = []

    for cv_name, _ in top_cvs:
        for filename, json_data in cv_store.items():
            if filename == cv_name:
                cv_data.append({
                    "filename": filename,
                    "json_data": json_data
                })
                break

    if not cv_data:
        return {"ranked_cvs": []}

    # Prepare prompt for LLM
    prompt = f"""
    You are a hiring assistant. Below is a job description and the JSON data of the top 5 CVs.
    Your task is to analyze the CVs and rank them in order of best fit for the job description.
    Only give ranking as per given format, do not give any additional information.

    Job Description:
    {jd_text}

    CV Data:
    {json.dumps(cv_data, indent=2)}

    Return the result in the following JSON format:
    {{
        "ranked_cvs": [
            {{
                "filename": "cv1.pdf",
                "ranking": "1"
            }},
            {{
                "filename": "cv2.pdf",
                "ranking": "2"
            }},
            ...
        ]
    }}
    """
    # time.sleep(4.1)
    response = completion(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("GOOGLE_API_KEY"),
        response_format={'type': 'json_object'}
    )
    # # Call Gemini LLM
    # response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
    input_tokens = response['usage']['prompt_tokens']
    output_tokens = response['usage']['completion_tokens']
    # # Clean and parse the response
    # cleaned_response = re.sub(r"```json\n(.*?)\n```", r"\1", response.text.strip(), flags=re.DOTALL)
    try:
        llm_ranking = json.loads(response.choices[0].message.content)
        return llm_ranking, input_tokens, output_tokens
    except json.JSONDecodeError:
        raise ValueError("Failed to parse LLM response as JSON.")


def process_and_rank_cvs(cv_contents: list, jd_content: bytes):
    """Complete pipeline: Process CVs, match with JD, and rank using LLM."""
    
    # Process all CVs (extract text from in-memory files)
    cv_store = {}
    start_time = timeit.default_timer()
    for filename, content in cv_contents:
        text = extract_text_from_pdf(io.BytesIO(content))
        cv_json, input_tokens, output_tokens = generate_json_from_text(text)
        cv_store[filename] = embedding_model.encode(json.dumps(cv_json)).tolist()

    # Process JD (extract text from in-memory JD file)
    jd_text = extract_text_from_pdf(io.BytesIO(jd_content))  # Pass BytesIO object

    # Match JD with CVs using semantic similarity
    ranked_cvs = match_jd_with_cvs(jd_text, cv_store)

    if not ranked_cvs:
        print("No matching CVs found.")
        return {"message": "No matching CVs found"}

    # Use LLM to rank top 5 CVs
    final_ranking, input_tokens2, output_tokens2 = sort_top_cvs_with_llm(ranked_cvs, jd_text, cv_store)
    end_time = timeit.default_timer()
    print("Time taken:", end_time - start_time)
    
    # Create a dictionary to store match scores with CV names from initial ranking
    cv_score_dict = {cv[0]: round(cv[1] * 100, 2) for cv in ranked_cvs}
    print("Total input tokens for json", input_tokens)
    print("Total output tokens for json", output_tokens)
    print("Total input tokens for llm", input_tokens2)
    print("Total output tokens for llm", output_tokens2)
    llm_results = [
        {
            "id": str(i + 1),  #  Ensure each result has an ID
            "name": cv["filename"],
            "matchScore": cv_score_dict.get(cv["filename"], 0)  # Convert similarity to percentage
        }
        for i, cv in enumerate(final_ranking["ranked_cvs"])
    ]

    return llm_results
