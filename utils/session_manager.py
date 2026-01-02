# utils/session_manager.py
import streamlit as st

def initialize_session_state():
    """Inicializa las variables de estado de la sesión"""
    
    # Estado de datos
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'filename' not in st.session_state:
        st.session_state.filename = None
    
    if 'upload_timestamp' not in st.session_state:
        st.session_state.upload_timestamp = None
    
    # Estado de gráficas generadas
    if 'generated_charts' not in st.session_state:
        st.session_state.generated_charts = []
    
    # Estado de configuración de exportación
    if 'export_format' not in st.session_state:
        st.session_state.export_format = 'PNG'
    
    if 'export_resolution' not in st.session_state:
        st.session_state.export_resolution = 'Standard (300 DPI)'
    
    if 'include_metadata' not in st.session_state:
        st.session_state.include_metadata = True
    
    if 'transparent_bg' not in st.session_state:
        st.session_state.transparent_bg = True