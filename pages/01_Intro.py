import streamlit as st
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Graph Maker Analysis - Intro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mantener consistencia
def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        padding: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    load_css()
    
    # Sidebar limpio - solo logo
    with st.sidebar:
        # Logo en la parte superior
        st.markdown("""
        <div class="logo-container">
            <h2 style="color: white; margin: 0;">📊 Graph Maker</h2>
        </div>
        """, unsafe_allow_html=True)

def show_page():
    """Introduction and overview page"""
    
    # Main title with style
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🏠 Welcome to Graph Maker Analysis
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Background image placeholder
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # You can add your background image here
        st.image("https://via.placeholder.com/600x300/667eea/white?text=Graph+Maker+Analysis", 
                caption="Welcome to Graph Maker Analysis", use_column_width=True)
    
    st.markdown("---")
    
    # Introduction
    st.header("📋 Overview")
    st.markdown("""
    **Graph Maker Analysis** is a comprehensive tool for data visualization and analysis.
    This application allows you to import, process, and create interactive charts in a simple and intuitive way.
    """)
    
    # Application sections
    st.header("🗂️ Application Sections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 Input Selection and Management
        - **Function:** Data source management and selection
        - **Features:**
          - Load CSV, Excel, JSON files
          - Data preview
          - Data cleaning and preparation
          - Format validation
        """)
        
        st.markdown("""
        ### 🔍 Insight Analysis
        - **Function:** Advanced insight analysis
        - **Features:**
          - Descriptive statistical analysis
          - Pattern and trend detection
          - Automatic report generation
          - Data-driven recommendations
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Graph Maker
        - **Function:** Create visualizations
        - **Features:**
          - Multiple chart types
          - Complete customization
          - Export in various formats
          - Interactive charts
        """)
        
        st.markdown("""
        ### 🚀 Additional Features
        - **Intuitive interface:** Simple navigation
        - **Responsive:** Adapts to different screens
        - **Export:** Save your analysis and charts
        - **Support:** Multiple data formats
        """)
    
    st.markdown("---")
    
    # How to get started
    st.header("🎯 How to Get Started?")
    
    steps_col1, steps_col2, steps_col3 = st.columns(3)
    
    with steps_col1:
        st.markdown("""
        ### Step 1️⃣
        **Load Data**
        
        Navigate to *Input Selection and Management* to upload your data files.
        """)
        
    with steps_col2:
        st.markdown("""
        ### Step 2️⃣
        **Create Charts**
        
        Go to *Graph Maker* to create custom visualizations of your data.
        """)
        
    with steps_col3:
        st.markdown("""
        ### Step 3️⃣
        **Analyze Insights**
        
        Use *Insight Analysis* to get automatic analysis and recommendations.
        """)
    
    st.markdown("---")
    
    # Tips and advice
    st.header("💡 Tips and Best Practices")
    
    tip_tabs = st.tabs(["📊 Charts", "📁 Data", "🔍 Analysis"])
    
    with tip_tabs[0]:
        st.markdown("""
        **Tips for creating better charts:**
        - Choose the right chart type for your data
        - Use colors that contrast well
        - Add descriptive titles and labels
        - Consider your target audience
        """)
        
    with tip_tabs[1]:
        st.markdown("""
        **Tips for data management:**
        - Check your data quality before importing
        - Make sure columns have descriptive names
        - Remove duplicates and null values
        - Verify data types
        """)
        
    with tip_tabs[2]:
        st.markdown("""
        **Tips for analysis:**
        - Clearly define your analysis objectives
        - Look for significant patterns and trends
        - Validate your conclusions with additional data
        - Document your findings
        """)
    
    # Quick start section
    st.markdown("---")
    st.header("⚡ Quick Start")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Load Sample Data", use_container_width=True):
            st.switch_page("pages/02_Input.py")
    
    with col2:
        if st.button("📊 Explore Charts", use_container_width=True):
            st.switch_page("pages/03_Graph_Creation.py")
    
    with col3:
        if st.button("🔍 View Analysis", use_container_width=True):
            st.switch_page("pages/04_Insights_Analysis.py")
    
    with col4:
        if st.button("📚 Documentation", use_container_width=True):
            st.info("Use the sidebar menu to navigate between sections")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
            <h4>Ready to Start?</h4>
            <p>Use the sidebar menu to navigate between different sections</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_page()