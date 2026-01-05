# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
from utils.data_validator import validate_dataframe
from utils.session_manager import initialize_session_state
from charts.bar_charts import BarChartComponent

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
        st.write(f"📁 **File:** {st.session_state.filename}")
        
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

def render_sidebar_info():
    """Información del dataset en sidebar"""
    st.divider()
    
    # Información básica del dataset
    st.subheader("📋 Dataset Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Companies", f"{st.session_state.df.shape[0]:,}")
    with col2:
        st.metric("Variables", st.session_state.df.shape[1])
    
    # Estadísticas adicionales
    with st.expander("📊 Quick Stats", expanded=False):
        st.write(f"**Loaded:** {st.session_state.upload_timestamp.strftime('%H:%M:%S')}")
        st.write(f"**Memory usage:** {st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Top 5 países si existe la columna
        if 'Country ISO code' in st.session_state.df.columns:
            top_countries = st.session_state.df['Country ISO code'].value_counts().head(5)
            st.write("**Top 5 countries:**")
            for country, count in top_countries.items():
                st.write(f"• {country}: {count:,}")
        
        # Información sobre columnas numéricas
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
        st.write(f"**Numeric columns:** {len(numeric_cols)}")
        
        # Información sobre columnas categóricas
        categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns
        st.write(f"**Categorical columns:** {len(categorical_cols)}")
    
    # Estado de gráficas guardadas
    if 'saved_charts' in st.session_state and st.session_state.saved_charts:
        st.divider()
        st.subheader("💾 Saved Charts")
        st.info(f"**{len(st.session_state.saved_charts)}** charts saved")
        
        # Resumen por tipo
        chart_types = {}
        for chart in st.session_state.saved_charts:
            chart_type = chart['chart_type']
            if chart_type not in chart_types:
                chart_types[chart_type] = 0
            chart_types[chart_type] += 1
        
        for chart_type, count in chart_types.items():
            st.write(f"• {chart_type}: {count}")

def render_chart_tabs():
    """Renderiza las pestañas para cada tipo de gráfica + Final Graphs"""
    
    # Por ahora solo Bar Chart y Final Graphs
    # Más adelante aquí agregaremos todas las demás pestañas
    tabs = st.tabs([
        "📊 Bar Chart", 
        "📋 Final Graphs"
        # Aquí se agregarán más pestañas:
        # "📈 Ridge Distribution",
        # "📉 KDE Distribution", 
        # "📚 Stacked Charts",
        # "⏰ Time Series",
        # "🎯 Scatter Plot",
        # "🔥 Heatmap",
        # "🎪 Cluster Chart"
    ])
    
    # Pestaña Bar Chart
    with tabs[0]:
        bar_chart_component = BarChartComponent()
        bar_chart_component.render()
    
    # Pestaña Final Graphs
    with tabs[1]:
        render_final_graphs_tab()

def render_final_graphs_tab():
    """Renderiza la pestaña Final Graphs"""
    
    st.subheader("📋 Final Graphs Collection")
    st.write("All saved charts from different tabs will appear here for final export.")
    
    # Verificar si hay gráficas guardadas
    if 'saved_charts' not in st.session_state or not st.session_state.saved_charts:
        st.info("🎯 No charts saved yet. Create and save charts from the individual chart tabs.")
        
        # Instrucciones para el usuario
        st.markdown("""
        ### 📖 How to use:
        
        1. **Go to any chart tab** (e.g., Bar Chart)
        2. **Select your variables** and configure parameters
        3. **Generate charts** using the "Generate" button
        4. **Select the charts** you like using the checkboxes
        5. **Save selected charts** using the "Save" button
        6. **Come back here** to see all your saved charts
        7. **Export everything** when ready
        """)
        return
    
    saved_charts = st.session_state.saved_charts
    
    # Header con información general
    st.success(f"✅ **{len(saved_charts)} charts saved** for final export")
    
    # Resumen por tipo de gráfica
    chart_summary = {}
    for chart in saved_charts:
        chart_type = chart['chart_type']
        if chart_type not in chart_summary:
            chart_summary[chart_type] = []
        chart_summary[chart_type].append(chart)
    
    # Mostrar resumen
    st.markdown("### 📊 Charts Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Charts", len(saved_charts))
    
    with col2:
        st.metric("Chart Types", len(chart_summary))
    
    with col3:
        dataset_name = st.session_state.get('filename', 'Unknown dataset')
        st.write(f"**Dataset:** {dataset_name}")
    
    # Mostrar gráficas agrupadas por tipo
    st.divider()
    
    for chart_type, charts in chart_summary.items():
        with st.expander(f"**{chart_type}** ({len(charts)} charts)", expanded=True):
            
            for i, chart in enumerate(charts, 1):
                st.markdown(f"#### Chart {i}: {chart['main_variable']} × {chart['extra_variable']}")
                
                # Información de la gráfica
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Mostrar la gráfica
                    st.pyplot(chart['figure'])
                
                with col2:
                    # Información y controles
                    st.write("**Parameters:**")
                    for param, value in chart['parameters'].items():
                        if isinstance(value, (tuple, list)):
                            st.write(f"• {param}: {value}")
                        else:
                            st.write(f"• {param}: {value}")
                    
                    # Botón para remover de la colección
                    if st.button(f"🗑️ Remove", key=f"remove_{chart['id']}", type="secondary"):
                        remove_chart_from_saved(chart['id'])
                        st.rerun()
                
                # Mostrar insights si existen
                if chart.get('insights'):
                    with st.expander(f"📊 Insights - Chart {i}", expanded=False):
                        st.json(chart['insights'])
                
                if i < len(charts):  # No agregar divider después del último
                    st.divider()
    
    # Sección de exportación
    if saved_charts:
        st.divider()
        render_export_section(saved_charts)

def render_export_section(saved_charts):
    """Renderiza la sección de exportación"""
    
    st.markdown("### 💾 Export Options")
    st.write("Configure how you want to export your saved charts:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_format = st.selectbox(
            "📄 Image Format",
            options=["PNG", "PDF", "SVG"],
            key="final_export_format",
            help="Format for individual chart images"
        )
        
        include_insights = st.checkbox(
            "📊 Include Insights JSON",
            value=True,
            key="include_insights",
            help="Include analysis JSON file with insights"
        )
    
    with col2:
        image_dpi = st.selectbox(
            "🎨 Image Quality",
            options=["300 DPI (Standard)", "600 DPI (High)", "1200 DPI (Print)"],
            key="final_dpi",
            help="Resolution for exported images"
        )
        
        transparent_bg = st.checkbox(
            "🔍 Transparent Background",
            value=True,
            key="final_transparent",
            help="Export charts with transparent backgrounds"
        )
    
    with col3:
        zip_name = st.text_input(
            "📦 Export Filename",
            value=f"financial_charts_{datetime.now().strftime('%Y%m%d_%H%M')}",
            key="export_filename",
            help="Name for the exported ZIP file"
        )
        
        # Mostrar información del archivo
        estimated_size = len(saved_charts) * 0.5  # Estimación aproximada en MB
        st.write(f"**Estimated size:** ~{estimated_size:.1f} MB")
    
    st.divider()
    
    # Botones de exportación
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📦 Download All", type="primary", key="download_all"):
            st.info("🚧 Export functionality will be implemented next!")
            # Aquí implementaremos la descarga
            
    with col2:
        if st.button("📊 JSON Only", type="secondary", key="download_json"):
            download_json_only(saved_charts)
            
    with col3:
        if st.button("🖼️ Images Only", type="secondary", key="download_images"):
            st.info("🚧 Image export will be implemented next!")
            
    with col4:
        if st.button("🗑️ Clear All", key="clear_all_saved"):
            if st.checkbox("⚠️ Confirm clear all", key="confirm_clear"):
                st.session_state.saved_charts = []
                st.success("All saved charts cleared!")
                st.rerun()

def download_json_only(saved_charts):
    """Descarga solo el archivo JSON con insights"""
    
    # Compilar todos los insights
    export_data = {
        'metadata': {
            'dataset': st.session_state.get('filename', 'Unknown'),
            'export_timestamp': datetime.now().isoformat(),
            'total_charts': len(saved_charts),
            'dataset_shape': list(st.session_state.df.shape) if 'df' in st.session_state else None
        },
        'charts': []
    }
    
    for chart in saved_charts:
        chart_data = {
            'id': chart['id'],
            'chart_type': chart['chart_type'],
            'main_variable': chart['main_variable'],
            'extra_variable': chart['extra_variable'],
            'parameters': chart['parameters'],
            'insights': chart.get('insights', {})
        }
        export_data['charts'].append(chart_data)
    
    # Crear archivo JSON para descarga
    import json
    json_str = json.dumps(export_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name=f"{st.session_state.get('export_filename', 'charts')}_insights.json",
        mime="application/json"
    )

def remove_chart_from_saved(chart_id):
    """Remueve una gráfica de la colección guardada"""
    if 'saved_charts' in st.session_state:
        st.session_state.saved_charts = [
            chart for chart in st.session_state.saved_charts 
            if chart['id'] != chart_id
        ]

def render_welcome_page():
    """Página de bienvenida cuando no hay datos cargados"""
    st.markdown("""
    ## Welcome to Financial Data Visualization Suite 🎯
    
    This tool helps you create professional financial data visualizations with just a few clicks.
    
    ### 🚀 Getting Started:
    1. **Upload your CSV file** using the sidebar
    2. **Choose a chart type** from the available tabs
    3. **Select your variables** and configure parameters  
    4. **Generate and review** your charts
    5. **Save the ones you like** to your collection
    6. **Export everything** from the Final Graphs tab
    
    ---
    
    ### 📋 Expected Data Format:
    - **File type:** CSV format only
    - **Size:** 1,000 - 10,000 companies (rows)
    - **Variables:** ~140-170 columns
    
    ### 🔧 Required Columns:
    - `Company name Latin alphabet` - Company names in Latin script
    - `Country ISO code` - ISO country codes (e.g., US, DE, FR)  
    - `Website address` - Company website URLs
    
    ### 📊 Available Chart Types:
    - **Bar Charts** - Compare values across categories
    - **Ridge Distributions** - Analyze data distributions  
    - **KDE Plots** - Kernel density estimation
    - **Stacked Charts** - Part-to-whole relationships
    - **Time Series** - Trends over time
    - **Scatter Plots** - Relationships between variables
    - **Heatmaps** - Density visualization
    - **Cluster Charts** - Group identification
    
    ### 🎯 What You'll Get:
    - **Interactive charts** with professional styling
    - **Statistical insights** and analysis JSON
    - **Customizable parameters** for each chart type
    - **Export-ready files** (PNG, PDF, SVG + JSON)
    
    ---
    
    **Ready to start?** Upload your CSV file in the sidebar! 👈
    """)

def clear_session_data():
    """Limpia los datos de la sesión"""
    keys_to_clear = [
        'df', 'data_loaded', 'filename', 'upload_timestamp',
        'saved_charts', 'bar_chart_results'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

if __name__ == "__main__":
    main()