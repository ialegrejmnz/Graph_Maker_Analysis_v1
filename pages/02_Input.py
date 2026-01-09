import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import json
import sys
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Input Selection and Management",
    page_icon="📁",
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
    
    # Sidebar para navegación
    with st.sidebar:
        # Logo en la parte superior
        st.markdown("""
        <div class="logo-container">
            <h2 style="color: white; margin: 0;">📊 Graph Maker</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navegación con page_link
        st.markdown("### Main Navigation")
        st.page_link("streamlit_app.py", label="Intro", icon="🏠")
        st.page_link("pages/02_Input.py", label="Input Selection and Management", icon="📁")
        st.page_link("pages/03_Graph_Creation.py", label="Graph Maker", icon="📊")
        st.page_link("pages/04_Insights_Analysis.py", label="Insight Analysis", icon="🔍")

def show_page():
    """Data input management and selection page"""
    
    # Page title
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            📁 Input Selection and Management
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for data
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Load Data", "👁️ Preview", "🧹 Process Data", "💾 Save/Load"])
    
    with tab1:
        st.header("📤 Upload Data Files")
        
        # Upload type selector
        upload_option = st.radio(
            "Select upload method:",
            ["Upload File", "Sample Data", "Manual Entry"]
        )
        
        if upload_option == "Upload File":
            uploaded_file = st.file_uploader(
                "Select a file",
                type=['csv', 'xlsx', 'json', 'txt'],
                help="Supported formats: CSV, Excel, JSON, TXT"
            )
            
            if uploaded_file is not None:
                try:
                    # Determine file type and process it
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension == 'csv':
                        # CSV options
                        col1, col2 = st.columns(2)
                        with col1:
                            delimiter = st.selectbox("Separator", [',', ';', '\t', '|'])
                        with col2:
                            encoding = st.selectbox("Encoding", ['utf-8', 'latin-1', 'cp1252'])
                        
                        df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
                        
                    elif file_extension in ['xlsx', 'xls']:
                        # Excel options
                        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
                        selected_sheet = st.selectbox("Select sheet", sheet_names)
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                        
                    elif file_extension == 'json':
                        json_data = json.load(uploaded_file)
                        df = pd.json_normalize(json_data)
                        
                    else:  # txt
                        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        df = pd.read_csv(stringio, delimiter='\t')
                    
                    st.session_state.uploaded_data = df
                    st.success(f"✅ File loaded successfully: {uploaded_file.name}")
                    st.info(f"📊 Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        elif upload_option == "Sample Data":
            st.subheader("📊 Sample Datasets")
            
            example_option = st.selectbox(
                "Select a sample dataset:",
                ["Monthly Sales", "Financial Data", "Web Analytics", "Inventory"]
            )
            
            if st.button("Load Sample Dataset"):
                if example_option == "Monthly Sales":
                    df = pd.DataFrame({
                        'Month': ['January', 'February', 'March', 'April', 'May', 'June'],
                        'Sales': [15000, 18000, 22000, 19000, 25000, 28000],
                        'Region': ['North', 'South', 'East', 'West', 'North', 'South'],
                        'Product': ['A', 'B', 'A', 'C', 'B', 'A']
                    })
                elif example_option == "Financial Data":
                    df = pd.DataFrame({
                        'Date': pd.date_range('2024-01-01', periods=100),
                        'Price': np.random.uniform(100, 200, 100),
                        'Volume': np.random.randint(1000, 5000, 100),
                        'Sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], 100)
                    })
                elif example_option == "Web Analytics":
                    df = pd.DataFrame({
                        'Page': [f'page_{i}' for i in range(1, 21)],
                        'Visits': np.random.randint(100, 1000, 20),
                        'Avg_Time': np.random.uniform(30, 300, 20),
                        'Bounce_Rate': np.random.uniform(0.1, 0.8, 20)
                    })
                else:  # Inventory
                    df = pd.DataFrame({
                        'Product': [f'Product_{i}' for i in range(1, 16)],
                        'Stock': np.random.randint(0, 100, 15),
                        'Unit_Price': np.random.uniform(10, 100, 15),
                        'Category': np.random.choice(['A', 'B', 'C'], 15)
                    })
                
                st.session_state.uploaded_data = df
                st.success(f"✅ Sample dataset '{example_option}' loaded successfully")
        
        else:  # Manual Entry
            st.subheader("✍️ Manual Data Entry")
            st.markdown("Enter data in CSV format (comma separated):")
            
            manual_data = st.text_area(
                "CSV Data:",
                height=200,
                placeholder="Name,Age,City\nJohn,25,Madrid\nAna,30,Barcelona\nPeter,35,Valencia"
            )
            
            if st.button("Process Manual Data"):
                try:
                    df = pd.read_csv(StringIO(manual_data))
                    st.session_state.uploaded_data = df
                    st.success("✅ Manual data processed successfully")
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
    
    with tab2:
        st.header("👁️ Data Preview")
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            # General dataset information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Null Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.1f}")
            
            # Data preview
            st.subheader("📋 First Rows")
            rows_to_show = st.slider("Number of rows to show", 5, min(50, len(df)), 10)
            st.dataframe(df.head(rows_to_show), use_container_width=True)
            
            # Column information
            st.subheader("📊 Column Information")
            
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Nulls': df.isnull().sum(),
                'Unique': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Descriptive statistics
            if st.checkbox("Show descriptive statistics"):
                st.subheader("📈 Descriptive Statistics")
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    st.dataframe(df[numeric_columns].describe(), use_container_width=True)
                else:
                    st.info("No numeric columns to show statistics")
        else:
            st.info("📝 Load data in the 'Load Data' tab to see preview")
    
    with tab3:
        st.header("🧹 Process and Clean Data")
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data.copy()
            
            # Processing options
            st.subheader("⚙️ Processing Options")
            
            # Handling null values
            if df.isnull().sum().sum() > 0:
                st.write("**Null Values Found:**")
                null_counts = df.isnull().sum()
                null_counts = null_counts[null_counts > 0]
                st.write(null_counts)
                
                null_action = st.selectbox(
                    "How to handle null values?",
                    ["Keep", "Remove rows", "Fill with mean", "Fill with median", "Fill with custom value"]
                )
                
                if null_action == "Remove rows":
                    df = df.dropna()
                    st.info(f"Rows removed. New dimensions: {df.shape}")
                elif null_action == "Fill with mean":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif null_action == "Fill with median":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif null_action == "Fill with custom value":
                    fill_value = st.text_input("Value to fill with:")
                    if fill_value:
                        df = df.fillna(fill_value)
            
            # Remove duplicates
            if st.checkbox("Remove duplicate rows"):
                original_shape = df.shape[0]
                df = df.drop_duplicates()
                st.info(f"Duplicates removed: {original_shape - df.shape[0]} rows")
            
            # Data filters
            st.subheader("🔍 Data Filters")
            
            # Column selection
            selected_columns = st.multiselect(
                "Select columns to keep:",
                df.columns.tolist(),
                default=df.columns.tolist()
            )
            
            if selected_columns:
                df = df[selected_columns]
            
            # Save processed data
            if st.button("💾 Save Processed Data"):
                st.session_state.processed_data = df
                st.success("✅ Processed data saved successfully")
                st.info(f"📊 Final dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            st.info("📝 Load data first in the 'Load Data' tab")
    
    with tab4:
        st.header("💾 Save and Load Configurations")
        
        # Export processed data
        if st.session_state.processed_data is not None:
            st.subheader("💾 Export Processed Data")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download CSV"):
                    csv = st.session_state.processed_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV file",
                        data=csv,
                        file_name="processed_data.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Download Excel"):
                    # For Excel you would need to install openpyxl
                    st.info("Excel functionality pending implementation")
        
        # Session state
        st.subheader("🔄 Session State")
        if st.session_state.uploaded_data is not None:
            st.success("✅ Original data loaded")
        if st.session_state.processed_data is not None:
            st.success("✅ Processed data available")
        
        if st.button("🗑️ Clear All"):
            st.session_state.uploaded_data = None
            st.session_state.processed_data = None
            st.success("✅ Data removed from session")
            st.rerun()

if __name__ == "__main__":
    main()
    show_page()