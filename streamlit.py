import streamlit as st
import os
import tempfile
import json
import pandas as pd
from typing import List
from rank_cv import process_and_rank_cvs
from score_jd import score_jds

def load_custom_css():
    """Load custom CSS with clean, professional file name styling"""
    st.markdown("""
    <style>
    .rank-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        color: white;
        margin-right: 1rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Classic Professional Ranking Badge Colors */
    .rank-1 { 
        background: linear-gradient(135deg, #f5c146 0%, #d4a017 100%); 
        color: #333;
        border: 2px solid rgba(255,255,255,0.4);
    }
    .rank-2 { 
        background: linear-gradient(135deg, #c9c9c9 0%, #a9a9a9 100%); 
        color: #333;
        border: 2px solid rgba(255,255,255,0.4);
    }
    .rank-3 { 
        background: linear-gradient(135deg, #cd7f32 0%, #b06c18 100%); 
        color: white;
        border: 2px solid rgba(255,255,255,0.3);
    }
    .rank-default { 
        background: linear-gradient(135deg, #6a82fb 0%, #fc5c7d 100%); 
        color: white;
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .score-display {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* FIXED: Clean and Professional File Name Styling */
    .file-name {
        font-size: 1.1rem;
        font-weight: 500;
        color: #34495e;
        text-shadow: none !important;
        margin-bottom: 0.4rem;
        line-height: 1.3;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .percentage-bar {
        background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
        border-radius: 15px;
        height: 26px;
        overflow: hidden;
        margin: 0.5rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #d5dbdb;
    }
    
    .percentage-fill {
        height: 100%;
        border-radius: 15px;
        transition: width 0.8s ease-in-out;
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }
    
    .percentage-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background-image: linear-gradient(45deg, transparent 35%, rgba(255,255,255,0.3) 50%, transparent 65%);
        animation: shine 2s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .success-banner {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    /* Enhanced Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fc 0%, #ecf0f1 100%);
        border: 2px solid #d5dbdb;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Professional Button Enhancement */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(52, 152, 219, 0.4);
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
    }
    
    /* Clean Container - No Effects */
    .main-container {
        background: transparent;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Professional Header Styling */
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    /* Clean ID Caption */
    .stCaption {
        color: #7f8c8d !important;
        font-weight: 400;
        font-size: 0.85rem;
    }
    
    /* Enhanced Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #34495e 0%, #2c3e50 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def get_score_color(score):
    """Return professional color gradients based on score"""
    if score >= 80:
        return "linear-gradient(135deg, #27ae60 0%, #2ecc71 100%)"  # Professional Green for excellent
    elif score >= 60:
        return "linear-gradient(135deg, #f39c12 0%, #e67e22 100%)"  # Professional Orange for good
    else:
        return "linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)"   # Professional Red for fair

def unwrap_result(result):
    """Unwrap tuple results that contain lists"""
    # Handle tuple containing list: ([{...}],) -> [{...}]
    if isinstance(result, tuple) and len(result) == 1:
        if isinstance(result[0], list):
            return result[0]
        elif isinstance(result[0], dict):
            return [result[0]]
    return result

def display_result_card(data, rank, is_cv=True):
    """Display individual result as a card with clean, professional styling"""
    try:
        if is_cv:
            # CV format: {'name': 'filename', 'matchScore': XX, 'id': 'XX'}
            file_name = data.get('name', 'Unknown CV')
            score = float(data.get('matchScore', 0))
            file_id = data.get('id', 'N/A')
            icon = "📄"
        else:
            # JD format: {'jd_file': 'filename', 'score': XX}
            file_name = data.get('jd_file', 'Unknown JD')
            score = float(data.get('score', 0))
            file_id = "N/A"
            icon = "📋"
        
        # Determine rank badge class
        rank_class = f"rank-{rank}" if rank <= 3 else "rank-default"
        
        # Create the clean, professional card
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f'<div class="rank-badge {rank_class}">#{rank}</div>', unsafe_allow_html=True)
            
            with col2:
                # Clean file name display
                st.markdown(f'<div class="file-name">{icon} {file_name}</div>', unsafe_allow_html=True)
                
                # Professional progress bar
                percentage = min(score, 100)  # Cap at 100%
                bar_color = get_score_color(score)
                
                st.markdown(f"""
                <div class="percentage-bar">
                    <div class="percentage-fill" style="width: {percentage}%; background: {bar_color};"></div>
                </div>
                """, unsafe_allow_html=True)
                
                if is_cv and file_id != "N/A":
                    st.caption(f"ID: {file_id}")
            
            with col3:
                st.markdown(f'<div class="score-display">{score:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
    except Exception as e:
        st.error(f"Error displaying result: {e}")
        st.json(data)

def create_summary_table(results_list, is_cv=True):
    """Create a professional summary table with download options"""
    try:
        if isinstance(results_list, list) and len(results_list) > 0:
            # Create DataFrame for better formatting
            df_data = []
            for idx, item in enumerate(results_list):
                if isinstance(item, dict):
                    if is_cv:
                        df_data.append({
                            'Rank': idx + 1,
                            'File Name': item.get('name', 'Unknown'),
                            'Match Score': f"{float(item.get('matchScore', 0)):.2f}%",
                            'ID': item.get('id', 'N/A')
                        })
                    else:
                        df_data.append({
                            'Rank': idx + 1,
                            'File Name': item.get('jd_file', 'Unknown'),
                            'Match Score': f"{float(item.get('score', 0)):.2f}%"
                        })
            
            if df_data:
                df = pd.DataFrame(df_data)
                
                # Display as professional table
                st.markdown("### 📊 Executive Summary")
                st.dataframe(
                    df, 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Professional download options
                col1, col2 = st.columns(2)
                with col1:
                    file_prefix = "cv_ranking" if is_cv else "jd_scoring"
                    st.download_button(
                        label="📊 Export to CSV",
                        data=df.to_csv(index=False),
                        file_name=f"{file_prefix}_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        label="📋 Export to JSON",
                        data=json.dumps(results_list, indent=2),
                        file_name=f"{file_prefix}_results.json",
                        mime="application/json",
                        use_container_width=True
                    )
    except Exception as e:
        st.error(f"Error creating summary: {e}")

def display_results(result, is_cv=True):
    """Universal result display function with clean styling"""
    if not result:
        st.warning("No results to display")
        return
    
    # Unwrap tuple results first
    result = unwrap_result(result)
    
    # Convert result to list format
    if isinstance(result, dict):
        # Single result
        results_list = [result]
    elif isinstance(result, list):
        results_list = result
    else:
        st.error("Unexpected result format after unwrapping")
        st.write(f"Result type: {type(result)}")
        st.write(f"Result content: {result}")
        return
    
    # Validate that we have proper data
    if not results_list or len(results_list) == 0:
        st.warning("No results to display")
        return
    
    # Sort results by score (descending)
    try:
        if is_cv:
            results_list = sorted(results_list, key=lambda x: float(x.get('matchScore', 0)), reverse=True)
        else:
            results_list = sorted(results_list, key=lambda x: float(x.get('score', 0)), reverse=True)
    except Exception as e:
        st.warning(f"Could not sort results: {e}")
    
    # Professional success banner
    banner_text = "✅ CV Analysis Complete!" if is_cv else "✅ JD Analysis Complete!"
    st.markdown(f'<div class="success-banner">{banner_text}</div>', unsafe_allow_html=True)
    
    # Professional Summary Statistics
    if len(results_list) > 0:
        col1, col2, col3 = st.columns(3)
        
        try:
            if is_cv:
                scores = [float(item.get('matchScore', 0)) for item in results_list if isinstance(item, dict)]
                file_type = "CVs"
            else:
                scores = [float(item.get('score', 0)) for item in results_list if isinstance(item, dict)]
                file_type = "JDs"
            
            if scores:
                with col1:
                    st.metric(f"📁 Total {file_type}", len(results_list))
                with col2:
                    st.metric("🏆 Top Score", f"{max(scores):.2f}%")
                with col3:
                    st.metric("📊 Average", f"{sum(scores)/len(scores):.2f}%")
        except Exception as e:
            with col1:
                st.metric(f"📁 Total {file_type}", len(results_list))
    
    st.markdown("---")
    
    # Display individual results with clean styling
    header_text = "🏆 CV Performance Rankings" if is_cv else "🏆 JD Match Rankings"
    st.subheader(header_text)
    
    for idx, item in enumerate(results_list):
        if isinstance(item, dict):
            display_result_card(item, idx + 1, is_cv)
        else:
            st.write(f"**Result {idx + 1}:** {item}")
    
    # Create professional downloadable summary
    create_summary_table(results_list, is_cv)

def main():
    st.set_page_config(
        page_title="JD-CV Analyzer",
        page_icon="💼",
        layout="wide"
    )
    
    load_custom_css()
    
    st.title("💼 JD-CV Analyzer")
    st.markdown("---")
    
    # Sidebar for selecting functionality
    st.sidebar.title("🛠️ Analysis Tools")
    option = st.sidebar.radio(
        "Select Analysis Type:",
        ("🔄 Rank CVs against JD", "📊 Score JDs against CV")
    )
    
    if option == "🔄 Rank CVs against JD":
        rank_cvs_interface()
    else:
        score_jds_interface()

def rank_cvs_interface():
    st.header("🔄 CV Performance Analysis")
    st.markdown("Upload a Job Description and multiple CV files to analyze candidate rankings.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Job Description")
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=['pdf', 'docx', 'txt'],
            key="jd_upload",
            help="Upload the job description file (PDF, DOCX, or TXT format)"
        )
        
        if jd_file is not None:
            st.success(f"✅ JD uploaded: {jd_file.name}")
    
    with col2:
        st.subheader("📑 CV Files")
        cv_files = st.file_uploader(
            "Upload CV files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="cv_upload",
            help="Upload multiple CV files (PDF, DOCX, or TXT format)"
        )
        
        if cv_files:
            st.success(f"✅ {len(cv_files)} CV(s) uploaded")
            with st.expander("📂 View uploaded CVs"):
                for i, cv in enumerate(cv_files, 1):
                    st.write(f"{i}. {cv.name}")
    
    st.markdown("---")
    
    if st.button("🚀 Analyze CV Performance", type="primary", use_container_width=True):
        if jd_file is not None and cv_files:
            with st.spinner("🔄 Analyzing CVs... Please wait."):
                try:
                    # Read JD content
                    jd_content = jd_file.read()
                    
                    # Read CV contents
                    cv_contents = [(cv.name, cv.read()) for cv in cv_files]
                    
                    # Process and rank CVs
                    result = process_and_rank_cvs(cv_contents, jd_content)
                    
                    # Display results
                    display_results(result, is_cv=True)
                    
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
                    st.info("Please check if the uploaded files are in the correct format.")
        else:
            st.error("⚠️ Please upload both a Job Description and at least one CV file.")

def score_jds_interface():
    st.header("📊 Job Description Match Analysis")
    st.markdown("Upload multiple Job Descriptions and one CV file to analyze job compatibility.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Job Descriptions")
        jd_files = st.file_uploader(
            "Upload Job Description files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="jds_upload",
            help="Upload multiple job description files (PDF, DOCX, or TXT format)"
        )
        
        if jd_files:
            st.success(f"✅ {len(jd_files)} JD(s) uploaded")
            with st.expander("📂 View uploaded JDs"):
                for i, jd in enumerate(jd_files, 1):
                    st.write(f"{i}. {jd.name}")
    
    with col2:
        st.subheader("📑 CV File")
        cv_file = st.file_uploader(
            "Upload CV file",
            type=['pdf', 'docx', 'txt'],
            key="cv_score_upload",
            help="Upload a single CV file (PDF, DOCX, or TXT format)"
        )
        
        if cv_file is not None:
            st.success(f"✅ CV uploaded: {cv_file.name}")
    
    st.markdown("---")
    
    if st.button("🎯 Analyze Job Compatibility", type="primary", use_container_width=True):
        if jd_files and cv_file is not None:
            with st.spinner("🎯 Analyzing job compatibility... Please wait."):
                try:
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded JD files temporarily
                        jd_paths = []
                        for jd in jd_files:
                            jd_path = os.path.join(temp_dir, jd.name)
                            jd_paths.append(jd_path)
                            with open(jd_path, "wb") as f:
                                f.write(jd.read())
                        
                        # Save CV file temporarily
                        cv_path = os.path.join(temp_dir, cv_file.name)
                        with open(cv_path, "wb") as f:
                            f.write(cv_file.read())
                        
                        # Call the score_jds function
                        results = score_jds(jd_paths, cv_path)
                        
                        # Display results using the same format as CV ranking
                        display_results(results, is_cv=False)
                        
                except Exception as e:
                    st.error(f"❌ Error scoring files: {str(e)}")
                    st.info("Please check if the uploaded files are in the correct format.")
        else:
            st.error("⚠️ Please upload both Job Description files and a CV file.")

if __name__ == "__main__":
    main()
