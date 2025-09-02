# JD-CV Analyzer: AI-Powered Resume Matching System

An advanced CV/Resume analysis system that leverages Large Language Models (LLM) and semantic similarity to intelligently match candidates with job descriptions. Built with modern Python technologies including Streamlit for a beautiful UI and FastAPI for robust API services.

## 🚀 Key Features

### Intelligent CV Ranking
- Upload a Job Description and multiple CVs for instant analysis
- Smart ranking algorithm combining LLM analysis and semantic matching
- Detailed match scoring with percentage-based compatibility
- Professional visualization of results with interactive cards
- Downloadable summary reports

### Job Description Scoring
- Analyze multiple Job Descriptions against a single CV
- Get detailed compatibility scores for each position
- Identify best-matching career opportunities
- Smart parsing of both structured and unstructured content

### Advanced Technology Stack
- **LLM Integration**: Utilizes Google's Gemini model for deep content understanding
- **Semantic Analysis**: Implements SentenceTransformer for accurate text similarity matching
- **Modern UI**: Clean, professional interface with animated components
- **RESTful API**: Full-featured API for system integration

## 🛠️ Technical Architecture

### Components
1. **Frontend (Streamlit)**
   - Interactive file upload interface
   - Real-time analysis visualization
   - Professional styling with custom CSS
   - Progress tracking and error handling

2. **Backend (FastAPI)**
   - RESTful API endpoints
   - Asynchronous file processing
   - Efficient memory management
   - Cross-Origin Resource Sharing (CORS) support

3. **Analysis Engine**
   - PDF text extraction and preprocessing
   - LLM-based content parsing
   - Semantic similarity computation
   - Intelligent ranking algorithms

## 📋 Installation

### Prerequisites
- Python 3.8 or higher
- pip or Poetry for dependency management
- Google API key for Gemini LLM

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/Nikhil-Maheshwari-10/jd-cv-analyzer.git
cd jd-cv-analyzer
```

2. Install dependencies using pip:
```bash
pip install -r requirements.txt
```

Or using Poetry:
```bash
poetry install
```

3. Set up environment variables:
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_api_key
```

## 🚀 Usage

### Interactive UI
Launch the Streamlit interface:
```bash
streamlit run streamlit.py
```
Access the UI at: http://localhost:8501

### API Server
Start the FastAPI server:
```bash
python main_api.py
```
API documentation available at: http://localhost:8000/docs

## 📚 API Documentation

### Endpoints

#### 1. CV Ranking
```http
POST /jd-cvs
```
- **Purpose**: Rank multiple CVs against a Job Description
- **Input**: 
  - `jd`: Job Description file (PDF/DOCX/TXT)
  - `cvs`: List of CV files (PDF/DOCX/TXT)
- **Output**: Ranked list of CVs with match scores

#### 2. JD Scoring
```http
POST /score-jds
```
- **Purpose**: Score multiple Job Descriptions against a CV
- **Input**:
  - `jds`: List of Job Description files
  - `cv`: Single CV file
- **Output**: Scored list of JDs with compatibility metrics

## 📁 Project Structure
```
├── streamlit.py          # Interactive UI implementation
├── main_api.py          # FastAPI server and endpoints
├── rank_cv.py           # CV ranking and matching logic
├── score_jd.py          # JD scoring implementation
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🛠️ Core Technologies
- **Streamlit**: Interactive UI framework
- **FastAPI**: Modern API framework
- **PyPDF2**: PDF processing
- **SentenceTransformer**: Text embeddings
- **Google Gemini**: LLM for content understanding
- **NumPy**: Numerical computations
- **Python-dotenv**: Environment management

## ⚡ Performance Features
- Efficient PDF text extraction
- Optimized token usage for LLM calls
- Parallel processing capabilities
- Memory-efficient file handling
- Caching for improved response times

## 🔒 Security Considerations
- Secure API key handling
- Temporary file cleanup
- Input validation and sanitization
- Error handling and logging

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Future Enhancements
- Multi-language support
- Advanced CV parsing
- Custom scoring algorithms
- Batch processing
- Analytics dashboard
- Export formats (PDF, XLSX)

## 👥 Author
Nikhil Maheshwari

---

**Note**: This project is continuously evolving. For the latest updates and features, please check the repository regularly.
