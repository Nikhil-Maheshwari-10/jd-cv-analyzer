import os
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile
from typing import List
from pathlib import Path
from rank_cv import process_and_rank_cvs
from score_jd import score_jds 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
@app.post("/jd-cvs")
async def upload_files(jd: UploadFile = File(...), cvs: List[UploadFile] = File(...)):
    """Upload multiple CVs and one Job Description (JD), then process them in real time."""
    
    # Read JD content from memory
    jd_content = await jd.read()
    
    # Read CVs content from memory
    cv_contents = [(cv.filename, await cv.read()) for cv in cvs]

    # Call the processing function with in-memory files
    result = process_and_rank_cvs(cv_contents, jd_content)

    return {"message": "Processing complete", "result": result}

@app.post("/score-jds")
async def score_jds_endpoint(jds: List[UploadFile] = File(...), cv: UploadFile = File(...)):
    """Upload multiple JDs and one CV, then return matching scores."""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded JD files temporarily
        jd_paths = []
        for jd in jds:
            jd_path = os.path.join(temp_dir, jd.filename)
            jd_paths.append(jd_path)
            with open(jd_path, "wb") as f:
                f.write(await jd.read())

        # Save CV file temporarily
        cv_path = os.path.join(temp_dir, cv.filename)
        with open(cv_path, "wb") as f:
            f.write(await cv.read())

        # Call the score_jds function
        results = score_jds(jd_paths, cv_path)

        # Files are automatically cleaned up when exiting the context manager

    return {"message": "Scoring complete", "results": results}
if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app
    uvicorn.run(app,reload=True, host="0.0.0.0", port=8000)
