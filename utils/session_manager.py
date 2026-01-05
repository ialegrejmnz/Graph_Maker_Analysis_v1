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
    
    # Estado de gráficas guardadas (colección final)
    if 'saved_charts' not in st.session_state:
        st.session_state.saved_charts = []