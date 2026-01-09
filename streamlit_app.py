import streamlit as st
import os
import sys
from pathlib import Path

# Configuración de la página principal
st.set_page_config(
    page_title="Graph Maker Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Añadir la carpeta pages al path para importar las páginas
current_dir = Path(__file__).parent
pages_dir = current_dir / "pages"
sys.path.append(str(pages_dir))

# CSS personalizado para estilo de la aplicación
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
    
    .nav-button {
        margin: 0.5rem 0 !important;
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
    
    # Contenido principal - Solo mostrar página Intro ya que las otras son archivos separados
    # Cargar y mostrar la página de Intro
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("intro_page", pages_dir / "01_Intro.py")
        intro_page = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(intro_page)
        intro_page.show_page()
        
    except ImportError as e:
        st.error(f"""
        **Error loading Intro page**
        
        Make sure the file exists: `pages/01_Intro.py`
        
        **Technical error:** {str(e)}
        """)
        
        st.markdown("### 🔧 Expected file structure:")
        st.code("""
Graph_Maker_Analysis_v1/
├── streamlit_app.py
└── pages/
    ├── __init__.py
    ├── 01_Intro.py
    ├── 02_Input.py
    ├── 03_Graph_Creation.py
    └── 04_Insights_Analysis.py
        """)

if __name__ == "__main__":
    main()