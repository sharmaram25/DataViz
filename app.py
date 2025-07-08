"""
DataViz - Intelligent EDA Web App
Main Streamlit application with minimalist steel-toned UI
"""

import streamlit as st
import pandas as pd
import io
from pathlib import Path
import base64

# Import custom modules
from eda_utils import DataAnalyzer
from report_generator import ReportGenerator

# Configure page
st.set_page_config(
    page_title="DataViz - Intelligent EDA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for steel-toned theme
def load_css():
    st.markdown("""
    <style>
    /* Global theme */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Sidebar styling for homepage */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50, #34495e);
    }
    
    .css-1lcbmhc .css-1outpf7 {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        border-right: 3px solid #7f8c8d;
    }
    
    /* Sidebar content */
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3 {
        color: #ecf0f1 !important;
        text-align: center;
    }
    
    .css-1lcbmhc .markdown-text-container {
        color: #bdc3c7 !important;
    }
    
    /* Upload widget styling */
    .css-1cpxqw2 {
        background: rgba(52, 73, 94, 0.3);
        border: 2px dashed #7f8c8d;
        border-radius: 10px;
    }
    
    /* Main content area */
    .main .block-container {
        background: #ecf0f1;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9, #21618c);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* DataFrames */
    .dataframe {
        border: 1px solid #bdc3c7;
        border-radius: 8px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Hide sidebar on exploration page */
    .exploration-mode .css-1lcbmhc {
        display: none !important;
    }
    
    .exploration-mode .css-1d391kg {
        margin-left: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sample_data():
    """Create sample dataset for demonstration"""
    import numpy as np
    np.random.seed(42)
    
    n_samples = 500  # Reduced for better performance
    data = {
        'employee_id': [f'EMP{str(i).zfill(3)}' for i in range(1, n_samples + 1)],
        'age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
        'salary': np.random.lognormal(10.5, 0.5, n_samples).clip(30000, 150000).round(-3),  # Round to thousands
        'experience_years': np.random.normal(8, 5, n_samples).clip(0, 30).round(1),
        'performance_score': np.random.normal(7.5, 1.2, n_samples).clip(1, 10).round(1),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.2, 0.5, 0.25, 0.05]),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations'], n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.45, 0.1]),
        'remote_work': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'bonus': np.where(np.random.random(n_samples) < 0.15, np.nan, np.random.uniform(1000, 15000, n_samples).round(-2)),  # 15% missing
        'satisfaction_rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.3, 0.35, 0.15]),
        'years_at_company': np.random.exponential(4, n_samples).clip(0.1, 25).round(1),
        'training_hours': np.random.gamma(2, 15, n_samples).clip(0, 120).round(0),
        'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Austin', 'Seattle', 'Boston', 'Denver'], n_samples),
        'team_size': np.random.poisson(8, n_samples).clip(1, 25),
        'overtime_hours': np.where(np.random.random(n_samples) < 0.1, np.nan, np.random.exponential(5, n_samples).clip(0, 40).round(1))  # 10% missing
    }
    
    df = pd.DataFrame(data)
    
    # Add some interesting correlations
    # Higher performance should correlate with higher salary
    performance_bonus = (df['performance_score'] - 5) * 5000
    df['salary'] = df['salary'] + performance_bonus
    
    # Experience should correlate with salary
    experience_bonus = df['experience_years'] * 1000
    df['salary'] = df['salary'] + experience_bonus
    
    # Ensure realistic salary ranges
    df['salary'] = df['salary'].clip(25000, 200000).round(-3)
    
    return df

def homepage():
    """Render the homepage with sidebar"""
    # Sidebar content
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1>🧠 DataViz</h1>
            <h3>Intelligent EDA Platform</h3>
            <p style='color: #bdc3c7; font-style: italic;'>
                Unlock insights from your data with advanced exploratory analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # File upload
        st.markdown("### 📁 Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset to begin analysis"
        )
        
        if uploaded_file:
            return uploaded_file
        
        st.markdown("---")
        
        # Sample data option
        st.markdown("### 🎯 Try Sample Data")
        if st.button("Load Sample Dataset", use_container_width=True):
            sample_df = create_sample_data()
            st.session_state.data = sample_df
            st.session_state.filename = "sample_employee_data.csv"
            st.rerun()
        
        st.markdown("---")
        
        # About section
        st.markdown("""
        ### 📋 About DataViz
        
        **Features:**
        - 📊 Comprehensive EDA
        - 🎨 Interactive Visualizations  
        - 🔍 Missing Value Analysis
        - 📈 Statistical Insights
        - 📄 Export Reports
        
        **Supported Formats:**
        - CSV files
        - Excel files (.xlsx, .xls)
        
        **Author:** Ram Sharma
        """)
    
    # Main content area for homepage
    if 'data' not in st.session_state:
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h1 style='color: #2c3e50; font-size: 3rem; margin-bottom: 1rem;'>
                🧠 Welcome to DataViz
            </h1>
            <h3 style='color: #7f8c8d; margin-bottom: 2rem;'>
                Intelligent Exploratory Data Analysis Platform
            </h3>
            <p style='font-size: 1.2rem; color: #34495e; max-width: 800px; margin: 0 auto 2rem;'>
                Transform your raw data into actionable insights with our comprehensive EDA toolkit. 
                Upload your dataset and discover patterns, outliers, and relationships that matter.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-container'>
                <h3>📊 Deep Analysis</h3>
                <p>Complete statistical profiling with skewness, kurtosis, and distribution analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-container'>
                <h3>🎨 Rich Visuals</h3>
                <p>Interactive charts and plots using Plotly for immersive data exploration</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-container'>
                <h3>🤖 Smart Insights</h3>
                <p>AI-powered suggestions for data cleaning and missing value treatment</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Getting started
        st.markdown("""
        <div style='background: #f8f9fa; padding: 2rem; border-radius: 10px; margin-top: 2rem; border-left: 4px solid #3498db;'>
            <h3 style='color: #2c3e50; margin-bottom: 1rem;'>🚀 Getting Started</h3>
            <ol style='color: #34495e; font-size: 1.1rem;'>
                <li>Upload your CSV or Excel file using the sidebar</li>
                <li>Or try our sample dataset to explore features</li>
                <li>Dive into comprehensive data analysis</li>
                <li>Export detailed reports for sharing</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    return None

def load_data(uploaded_file):
    """Load data from uploaded file with improved type detection"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings and separators
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        else:  # Excel file
            df = pd.read_excel(uploaded_file)
        
        # Basic data cleaning
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Try to infer better data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    # Check if it looks numeric (ignoring common formatting)
                    sample = df[col].dropna().astype(str).str.replace(r'[,$%\s]', '', regex=True)
                    if sample.str.match(r'^-?\d*\.?\d+$').all() and len(sample) > 0:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[,$%\s]', '', regex=True), errors='coerce')
                except:
                    pass
                
                # Try to convert to datetime
                try:
                    if 'date' in col.lower() or 'time' in col.lower():
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
        
        # Store processed data
        st.session_state.data = df
        st.session_state.filename = uploaded_file.name
        
        # Show basic info about loaded data
        st.success(f"✅ Successfully loaded {uploaded_file.name}")
        st.info(f"📊 Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.error("Please ensure your file is a valid CSV or Excel file.")
        return None

def exploration_dashboard():
    """Render the data exploration dashboard"""
    if 'data' not in st.session_state:
        st.error("No data loaded. Please upload a file first.")
        return
    
    df = st.session_state.data
    filename = st.session_state.get('filename', 'uploaded_data')
    
    # Add custom CSS to hide sidebar in exploration mode
    st.markdown("""
    <style>
    .css-1lcbmhc { display: none !important; }
    .css-1d391kg { margin-left: 0 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with back button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f"# 🧠 DataViz Analysis: {filename}")
    with col2:
        if st.button("🏠 Home", type="secondary"):
            del st.session_state.data
            if 'filename' in st.session_state:
                del st.session_state.filename
            st.rerun()
    
    # Initialize analyzer
    analyzer = DataAnalyzer(df)
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "📉 Statistics", 
        "❓ Missing Values", 
        "📈 Visualizations", 
        "📋 Data Preview", 
        "📤 Export"
    ])
    
    with tab1:
        analyzer.show_dataset_overview()
    
    with tab2:
        analyzer.show_descriptive_statistics()
    
    with tab3:
        analyzer.show_missing_values_analysis()
    
    with tab4:
        analyzer.show_visual_analytics()
    
    with tab5:
        analyzer.show_data_preview()
    
    with tab6:
        show_export_options(df, filename, analyzer)

def show_export_options(df, filename, analyzer):
    """Show export options for reports"""
    st.markdown("## 📤 Export Analysis Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Export Options")
        
        # HTML Report
        if st.button("📄 Generate HTML Report", use_container_width=True):
            with st.spinner("Generating HTML report..."):
                report_gen = ReportGenerator(df, filename)
                html_content = report_gen.generate_html_report(analyzer)
                
                # Download button for HTML
                st.download_button(
                    label="Download HTML Report",
                    data=html_content,
                    file_name=f"{filename}_analysis_report.html",
                    mime="text/html",
                    use_container_width=True
                )
        
        # PDF Report
        if st.button("📑 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                try:
                    report_gen = ReportGenerator(df, filename)
                    pdf_content = report_gen.generate_pdf_report(analyzer)
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_content,
                        file_name=f"{filename}_analysis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
                    st.info("Try downloading the HTML report instead.")
    
    with col2:
        st.markdown("### Summary Statistics CSV")
        
        if st.button("📊 Export Summary Stats", use_container_width=True):
            # Create summary statistics
            summary_stats = analyzer.get_summary_statistics()
            csv_buffer = io.StringIO()
            summary_stats.to_csv(csv_buffer, index=True)
            
            st.download_button(
                label="Download Summary CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{filename}_summary_statistics.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    """Main application function"""
    load_css()
    
    # Check if we're in exploration mode
    if 'data' in st.session_state:
        exploration_dashboard()
    else:
        # Homepage mode
        uploaded_file = homepage()
        
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.rerun()

if __name__ == "__main__":
    main()
