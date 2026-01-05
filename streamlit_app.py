# app.py
import streamlit as st
import pandas as pd
from utils.data_validator import validate_dataframe
from utils.session_manager import initialize_session_state
from charts.chart_registry import CHART_REGISTRY
from ui.final_graphs import render_final_graphs_tab

def main():
    st.set_page_config(
        page_title="Financial Data Visualization Suite",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("📊 Financial Data Visualization Suite")
    
    # Sidebar para carga de datos
    with st.sidebar:
        render_data_upload_section()
        
        if st.session_state.data_loaded:
            render_sidebar_info()
    
    # Main content area
    if st.session_state.data_loaded:
        render_chart_tabs()
    else:
        render_welcome_page()

def render_chart_tabs():
    """Renderiza las pestañas para cada tipo de gráfica + Final Graphs"""
    
    # Crear lista de pestañas: una por cada tipo de gráfica + Final Graphs
    tab_names = list(CHART_REGISTRY.keys()) + ["Final Graphs"]
    tabs = st.tabs(tab_names)
    
    # Renderizar cada pestaña de gráfica
    for i, (chart_name, chart_config) in enumerate(CHART_REGISTRY.items()):
        with tabs[i]:
            render_chart_tab(chart_name, chart_config)
    
    # Pestaña Final Graphs (última)
    with tabs[-1]:
        render_final_graphs_tab()

def render_chart_tab(chart_name: str, chart_config: dict):
    """Renderiza una pestaña individual para un tipo de gráfica"""
    
    st.subheader(f"{chart_config['icon']} {chart_name}")
    
    # Descripción del tipo de gráfica
    st.markdown(f"**Description:** {chart_config['description']}")
    
    # Mostrar parámetros requeridos
    with st.expander("📋 Required Parameters", expanded=False):
        st.write(f"**Main Variable:** {chart_config['main_var_description']}")
        st.write(f"**Extra Variables:** {chart_config['extra_vars_description']}")
        
        if chart_config['additional_params']:
            st.write("**Additional Parameters:**")
            for param, desc in chart_config['additional_params'].items():
                st.write(f"• **{param}**: {desc}")
    
    st.divider()
    
    # Sección 1: Selección de variables
    st.markdown("### 📊 Variable Selection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Main variable selection
        main_variable = render_main_variable_selector(chart_name, chart_config)
        
        # Extra variables selection  
        extra_variables = render_extra_variables_selector(chart_name, chart_config)
    
    with col2:
        # Preview de combinaciones
        if main_variable and extra_variables:
            st.write("**Chart combinations to be generated:**")
            for i, extra_var in enumerate(extra_variables, 1):
                st.write(f"{i}. {main_variable} × {extra_var}")
            
            st.info(f"Total charts: **{len(extra_variables)}**")
    
    # Solo mostrar parámetros adicionales si hay variables seleccionadas
    if main_variable and extra_variables:
        st.divider()
        
        # Sección 2: Parámetros adicionales
        st.markdown("### ⚙️ Chart Parameters")
        additional_params = render_additional_parameters(chart_name, chart_config)
        
        st.divider()
        
        # Sección 3: Generar gráficas
        st.markdown("### 🚀 Generate Charts")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button(f"Show {chart_name} Graphs", type="primary", key=f"show_{chart_name}"):
                generate_charts(chart_name, chart_config, main_variable, extra_variables, additional_params)
        
        with col2:
            if st.button(f"Clear {chart_name}", type="secondary", key=f"clear_{chart_name}"):
                clear_chart_results(chart_name)
                st.rerun()
        
        # Mostrar gráficas generadas
        display_generated_charts(chart_name)

def render_main_variable_selector(chart_name: str, chart_config: dict):
    """Selector para la variable principal"""
    
    # Filtrar columnas según el tipo requerido
    if chart_config['main_var_type'] == 'numeric':
        available_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    else:
        available_cols = st.session_state.df.columns.tolist()
    
    if not available_cols:
        st.error(f"No {chart_config['main_var_type']} columns found in dataset")
        return None
    
    main_variable = st.selectbox(
        f"Select Main Variable ({chart_config['main_var_type']})",
        options=[None] + available_cols,
        key=f"main_var_{chart_name}",
        help=chart_config['main_var_description']
    )
    
    return main_variable

def render_extra_variables_selector(chart_name: str, chart_config: dict):
    """Selector para las variables extra"""
    
    # Filtrar columnas según el tipo requerido
    if chart_config['extra_vars_type'] == 'categorical':
        available_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        # También incluir numéricas con pocos valores únicos
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if st.session_state.df[col].nunique() <= 20:  # Threshold para considerar como categórica
                available_cols.append(col)
    elif chart_config['extra_vars_type'] == 'numeric':
        available_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    else:
        available_cols = st.session_state.df.columns.tolist()
    
    if not available_cols:
        st.error(f"No {chart_config['extra_vars_type']} columns found in dataset")
        return []
    
    extra_variables = st.multiselect(
        f"Select Extra Variables ({chart_config['extra_vars_type']})",
        options=available_cols,
        key=f"extra_vars_{chart_name}",
        help=chart_config['extra_vars_description']
    )
    
    return extra_variables

def render_additional_parameters(chart_name: str, chart_config: dict):
    """Renderiza los parámetros adicionales específicos de cada gráfica"""
    
    params = {}
    
    # Usar el componente específico para cada tipo de gráfica
    if hasattr(chart_config['component'], 'render_parameters'):
        params = chart_config['component'].render_parameters()
    
    return params

def generate_charts(chart_name: str, chart_config: dict, main_variable: str, 
                   extra_variables: list, additional_params: dict):
    """Genera las gráficas para las combinaciones seleccionadas"""
    
    if f"results_{chart_name}" not in st.session_state:
        st.session_state[f"results_{chart_name}"] = []
    
    with st.spinner(f"Generating {len(extra_variables)} {chart_name} charts..."):
        results = []
        
        for extra_var in extra_variables:
            try:
                # Llamar a la función de creación de gráfica
                fig, insights = chart_config['component'].create_chart(
                    st.session_state.df,
                    main_variable,
                    extra_var,
                    **additional_params
                )
                
                results.append({
                    'chart_name': chart_name,
                    'main_variable': main_variable,
                    'extra_variable': extra_var,
                    'figure': fig,
                    'insights': insights,
                    'parameters': additional_params,
                    'selected': False  # Para el sistema de selección
                })
                
            except Exception as e:
                st.error(f"Error generating chart for {main_variable} × {extra_var}: {str(e)}")
        
        # Guardar resultados
        st.session_state[f"results_{chart_name}"] = results
        
    st.success(f"Generated {len(results)} charts successfully!")
    st.rerun()

def display_generated_charts(chart_name: str):
    """Muestra las gráficas generadas con opción de selección"""
    
    if f"results_{chart_name}" not in st.session_state:
        return
    
    results = st.session_state[f"results_{chart_name}"]
    
    if not results:
        return
    
    st.markdown("### 📈 Generated Charts")
    st.write(f"Select the charts you want to keep for final export:")
    
    for i, result in enumerate(results):
        with st.container():
            # Header con checkbox de selección
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Chart {i+1}:** {result['main_variable']} × {result['extra_variable']}")
            
            with col2:
                # Checkbox para seleccionar
                selected = st.checkbox(
                    "Select",
                    value=result['selected'],
                    key=f"select_{chart_name}_{i}"
                )
                
                # Actualizar estado
                if selected != result['selected']:
                    st.session_state[f"results_{chart_name}"][i]['selected'] = selected
            
            # Mostrar gráfica
            st.pyplot(result['figure'])
            
            # Mostrar insights si los hay
            if result['insights']:
                with st.expander(f"📊 Insights for Chart {i+1}", expanded=False):
                    st.json(result['insights'])
            
            st.divider()

def clear_chart_results(chart_name: str):
    """Limpia los resultados de un tipo de gráfica"""
    if f"results_{chart_name}" in st.session_state:
        del st.session_state[f"results_{chart_name}"]

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
                width='stretch'
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