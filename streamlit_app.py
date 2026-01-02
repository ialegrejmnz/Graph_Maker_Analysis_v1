# app.py
import streamlit as st
import pandas as pd
from utils.data_validator import validate_dataframe
from utils.session_manager import initialize_session_state
import io
import json
from datetime import datetime

def main():
    st.set_page_config(
        page_title="Financial Data Visualization Suite",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    st.title("📊 Financial Data Visualization Suite")
    
    # Sidebar para carga de datos
    with st.sidebar:
        render_data_upload_section()
        
        # Solo mostrar opciones adicionales si hay datos cargados
        if st.session_state.data_loaded:
            render_sidebar_controls()
    
    # Main content area
    if st.session_state.data_loaded:
        render_main_dashboard()
    else:
        render_welcome_page()

def render_data_upload_section():
    """Sección de carga de datos en sidebar"""
    st.header("📂 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your financial dataset (CSV format, 1K-10K rows, ~140-170 columns)"
    )
    
    if uploaded_file is not None:
        if st.button("🔄 Load Data", type="primary"):
            load_and_process_data(uploaded_file)
    
    # Mostrar estado actual de los datos
    if st.session_state.data_loaded:
        st.success(f"✅ **Data loaded successfully**")
        st.info(f"📊 **{st.session_state.df.shape[0]:,} rows** × **{st.session_state.df.shape[1]} columns**")
        
        if st.button("🗑️ Clear Data", type="secondary"):
            clear_session_data()
            st.rerun()

def load_and_process_data(uploaded_file):
    """Carga y procesa el archivo CSV"""
    try:
        with st.spinner("Loading and validating data..."):
            # Leer CSV
            df = pd.read_csv(uploaded_file)
            
            # Validar estructura
            validation_result = validate_dataframe(df)
            
            if validation_result['is_valid']:
                # Guardar en session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.upload_timestamp = datetime.now()
                st.session_state.filename = uploaded_file.name
                
                # Mostrar warnings si existen
                if validation_result['warnings']:
                    for warning in validation_result['warnings']:
                        st.warning(f"⚠️ {warning}")
                
                st.success("Data loaded successfully!")
                st.rerun()
                
            else:
                st.error("❌ Data validation failed:")
                for error in validation_result['errors']:
                    st.error(f"• {error}")
                
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {str(e)}")

def render_sidebar_controls():
    """Controles adicionales en sidebar cuando hay datos"""
    st.divider()
    
    # Información del dataset
    st.subheader("📋 Dataset Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Companies", f"{st.session_state.df.shape[0]:,}")
    with col2:
        st.metric("Variables", st.session_state.df.shape[1])
    
    # Estadísticas rápidas
    with st.expander("📊 Quick Stats"):
        st.write(f"**Filename:** {st.session_state.filename}")
        st.write(f"**Loaded:** {st.session_state.upload_timestamp.strftime('%H:%M:%S')}")
        st.write(f"**Memory usage:** {st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Top 5 países
        if 'Country ISO code' in st.session_state.df.columns:
            top_countries = st.session_state.df['Country ISO code'].value_counts().head()
            st.write("**Top countries:**")
            for country, count in top_countries.items():
                st.write(f"• {country}: {count}")

def render_welcome_page():
    """Página de bienvenida"""
    st.markdown("""
    ## Welcome to Financial Data Visualization Suite
    
    Please upload your financial dataset using the sidebar to get started.
    
    ### 📋 Expected Data Format:
    - **File type**: CSV format only
    - **Size**: 1,000 - 10,000 companies (rows)
    - **Variables**: ~140-170 columns
    
    ### 🔧 Required Columns:
    - `Company name Latin alphabet` - Company names in Latin script
    - `Country ISO code` - ISO country codes (e.g., US, DE, FR)
    - `Website address` - Company website URLs
    
    ### 🎯 What you'll get:
    - Interactive data visualization suite
    - Multiple downloadable charts
    - Comprehensive analysis JSON report
    
    ---
    
    **Ready to start?** Upload your CSV file in the sidebar! 👈
    """)

def render_main_dashboard():
    """Dashboard principal con tabs para diferentes secciones"""
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📊 Chart Builder", "📈 Analysis Results", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Chart Builder")
        st.write("Chart creation interface will be implemented here...")
        
        # Preview de datos
        with st.expander("🔍 Data Preview", expanded=False):
            st.dataframe(
                st.session_state.df.head(10),
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Analysis Results")
        st.write("Generated charts and insights will appear here...")
        
        # Placeholder para resultados
        if 'generated_charts' in st.session_state and st.session_state.generated_charts:
            st.write("📊 Generated charts will be displayed here")
        else:
            st.info("No charts generated yet. Use the Chart Builder to create visualizations.")
    
    with tab3:
        st.subheader("Settings")
        render_settings_panel()

def render_settings_panel():
    """Panel de configuraciones"""
    st.write("### Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox(
            "Chart export format",
            options=["PNG", "PDF", "SVG"],
            key="export_format"
        )
        
        st.selectbox(
            "Chart resolution",
            options=["Standard (300 DPI)", "High (600 DPI)", "Print (1200 DPI)"],
            key="export_resolution"
        )
    
    with col2:
        st.checkbox("Include data source info", value=True, key="include_metadata")
        st.checkbox("Transparent backgrounds", value=True, key="transparent_bg")

def clear_session_data():
    """Limpia los datos de la sesión"""
    keys_to_clear = ['df', 'data_loaded', 'upload_timestamp', 'filename', 'generated_charts']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

if __name__ == "__main__":
    main()