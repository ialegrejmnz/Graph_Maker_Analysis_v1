# ui/final_graphs.py
import streamlit as st
import json
from datetime import datetime
import zipfile
import io

def render_final_graphs_tab():
    """Renderiza la pestaña Final Graphs"""
    
    st.subheader("📋 Final Graphs Collection")
    
    # Recopilar todas las gráficas seleccionadas
    selected_charts = collect_selected_charts()
    
    if not selected_charts:
        st.info("No charts selected yet. Go to individual chart tabs and select the graphs you want to keep.")
        return
    
    st.success(f"**{len(selected_charts)} charts selected** for final export")
    
    # Mostrar resumen
    display_selection_summary(selected_charts)
    
    st.divider()
    
    # Opciones de exportación
    render_export_options(selected_charts)

def collect_selected_charts():
    """Recopila todas las gráficas seleccionadas de todas las pestañas"""
    
    selected_charts = []
    
    for chart_type in st.session_state:
        if chart_type.startswith("results_"):
            chart_name = chart_type.replace("results_", "")
            results = st.session_state[chart_type]
            
            for result in results:
                if result.get('selected', False):
                    selected_charts.append(result)
    
    return selected_charts

def display_selection_summary(selected_charts):
    """Muestra resumen de gráficas seleccionadas"""
    
    # Agrupar por tipo
    by_type = {}
    for chart in selected_charts:
        chart_type = chart['chart_name']
        if chart_type not in by_type:
            by_type[chart_type] = []
        by_type[chart_type].append(chart)
    
    st.markdown("### 📊 Selection Summary")
    
    for chart_type, charts in by_type.items():
        with st.expander(f"**{chart_type}** ({len(charts)} charts)", expanded=True):
            for i, chart in enumerate(charts, 1):
                st.write(f"{i}. {chart['main_variable']} × {chart['extra_variable']}")

def render_export_options(selected_charts):
    """Renderiza las opciones de exportación"""
    
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_format = st.selectbox(
            "Image Format",
            options=["PNG", "PDF", "SVG"],
            key="final_export_format"
        )
        
        include_insights = st.checkbox(
            "Include Insights JSON",
            value=True,
            key="include_insights"
        )
    
    with col2:
        image_dpi = st.selectbox(
            "Image Quality",
            options=["300 DPI (Standard)", "600 DPI (High)", "1200 DPI (Print)"],
            key="final_dpi"
        )
        
        transparent_bg = st.checkbox(
            "Transparent Background",
            value=True,
            key="final_transparent"
        )
    
    with col3:
        zip_name = st.text_input(
            "Export Filename",
            value=f"financial_charts_{datetime.now().strftime('%Y%m%d_%H%M')}",
            key="export_filename"
        )
    
    st.divider()
    
    # Botones de exportación
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.button("🗂️ Download All", type="primary")
            
    with col2:
        st.button("📋 Download JSON Only", type="secondary")