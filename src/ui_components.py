import streamlit as st

def render_header():
    """Renders the main header of the app."""
    # Custom HTML Header with Gradient Text
    st.markdown("""
    <div style="padding-bottom: 2rem;">
        <h1 style="margin-bottom: 0;">DataViz</h1>
        <p style="color: #64748B; font-size: 1.2rem; font-weight: 400; margin-top: -10px;">
            Production-Grade Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Renders the sidebar navigation and info."""
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        # Using a more visual radio but custom styled CSS handles it
        page = st.radio(
            "Go to", 
            ["Dataset Overview", "Advanced Exploration", "Visualization Studio"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # New About Section
        st.markdown("### ℹ️ About")
        st.write("Built to Outperform, By **Ram Sharma**.")
        st.caption("A production-ready solution for automated data analysis and storytelling.")
        
        return page

def render_metrics(summary, score):
    """Renders key metrics in a row with card styling."""
    
    # Custom HTML for metrics to break out of Streamlit's constraints
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        {"label": "Rows", "value": summary['rows']},
        {"label": "Columns", "value": summary['cols']},
        {"label": "Missing", "value": summary['missing_cells']},
        {"label": "Duplicates", "value": summary['duplicates']},
        {"label": "Quality Score", "value": f"{score}/100"}
    ]
    
    for col, metric in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.metric(metric["label"], metric["value"])

def setup_page():
    """Injects custom CSS."""
    try:
        with open('assets/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass
