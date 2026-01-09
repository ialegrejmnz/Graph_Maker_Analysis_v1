import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Graph Maker",
    page_icon="📊",
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
    """Página de creación de gráficos"""
    
    # Título de la página
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            📊 Graph Maker
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar si hay datos disponibles
    data_source = None
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        data_source = st.session_state.processed_data
        st.success("✅ Usando datos procesados")
    elif 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        data_source = st.session_state.uploaded_data
        st.info("ℹ️ Usando datos originales (considera procesarlos primero)")
    else:
        st.warning("⚠️ No hay datos disponibles. Ve a 'Input Selection and Management' para cargar datos.")
        
        # Opción para crear datos de ejemplo rápidos
        if st.button("🎲 Generar Datos de Ejemplo para Gráficos"):
            sample_data = pd.DataFrame({
                'Categoría': ['A', 'B', 'C', 'D', 'E'],
                'Valores': [23, 45, 56, 78, 32],
                'Fecha': pd.date_range('2024-01-01', periods=5),
                'Precio': [100, 150, 120, 180, 90]
            })
            st.session_state.uploaded_data = sample_data
            data_source = sample_data
            st.success("✅ Datos de ejemplo generados")
            st.rerun()
    
    if data_source is not None:
        df = data_source
        
        # Sidebar para configuración de gráficos
        with st.sidebar:
            st.markdown("### ⚙️ Configuración de Gráfico")
            
            # Tipo de gráfico
            chart_type = st.selectbox(
                "Tipo de Gráfico:",
                [
                    "Gráfico de Barras",
                    "Gráfico de Líneas", 
                    "Gráfico de Dispersión",
                    "Histograma",
                    "Gráfico de Pastel",
                    "Box Plot",
                    "Mapa de Calor",
                    "Gráfico de Área"
                ]
            )
            
            # Selección de columnas basada en el tipo de gráfico
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            all_columns = df.columns.tolist()
        
        # Layout principal con tabs
        tab1, tab2, tab3 = st.tabs(["📊 Crear Gráfico", "🎨 Personalizar", "📥 Exportar"])
        
        with tab1:
            st.header("📊 Configuración del Gráfico")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("📋 Opciones")
                
                if chart_type == "Gráfico de Barras":
                    x_column = st.selectbox("Eje X:", all_columns)
                    y_column = st.selectbox("Eje Y:", numeric_columns)
                    color_column = st.selectbox("Color por:", [None] + all_columns)
                    
                elif chart_type == "Gráfico de Líneas":
                    x_column = st.selectbox("Eje X:", all_columns)
                    y_column = st.selectbox("Eje Y:", numeric_columns)
                    color_column = st.selectbox("Líneas por:", [None] + categorical_columns)
                    
                elif chart_type == "Gráfico de Dispersión":
                    x_column = st.selectbox("Eje X:", numeric_columns)
                    y_column = st.selectbox("Eje Y:", numeric_columns)
                    color_column = st.selectbox("Color por:", [None] + all_columns)
                    size_column = st.selectbox("Tamaño por:", [None] + numeric_columns)
                    
                elif chart_type == "Histograma":
                    x_column = st.selectbox("Columna:", numeric_columns)
                    bins = st.slider("Número de bins:", 10, 100, 30)
                    
                elif chart_type == "Gráfico de Pastel":
                    values_column = st.selectbox("Valores:", numeric_columns)
                    names_column = st.selectbox("Etiquetas:", categorical_columns)
                    
                elif chart_type == "Box Plot":
                    y_column = st.selectbox("Variable numérica:", numeric_columns)
                    x_column = st.selectbox("Categoría:", [None] + categorical_columns)
                    
                elif chart_type == "Mapa de Calor":
                    st.info("Selecciona solo columnas numéricas para el mapa de calor")
                    heatmap_columns = st.multiselect("Columnas:", numeric_columns, default=numeric_columns[:5])
                    
                elif chart_type == "Gráfico de Área":
                    x_column = st.selectbox("Eje X:", all_columns)
                    y_column = st.selectbox("Eje Y:", numeric_columns)
                    color_column = st.selectbox("Área por:", [None] + categorical_columns)
            
            with col1:
                st.subheader("📈 Vista Previa del Gráfico")
                
                try:
                    # Crear el gráfico basado en el tipo seleccionado
                    if chart_type == "Gráfico de Barras":
                        fig = px.bar(df, x=x_column, y=y_column, color=color_column,
                                   title=f"Gráfico de Barras: {y_column} por {x_column}")
                        
                    elif chart_type == "Gráfico de Líneas":
                        fig = px.line(df, x=x_column, y=y_column, color=color_column,
                                    title=f"Gráfico de Líneas: {y_column} vs {x_column}")
                        
                    elif chart_type == "Gráfico de Dispersión":
                        fig = px.scatter(df, x=x_column, y=y_column, color=color_column,
                                       size=size_column, title=f"Dispersión: {y_column} vs {x_column}")
                        
                    elif chart_type == "Histograma":
                        fig = px.histogram(df, x=x_column, nbins=bins,
                                         title=f"Histograma: {x_column}")
                        
                    elif chart_type == "Gráfico de Pastel":
                        fig = px.pie(df, values=values_column, names=names_column,
                                   title=f"Gráfico de Pastel: {values_column}")
                        
                    elif chart_type == "Box Plot":
                        fig = px.box(df, y=y_column, x=x_column,
                                   title=f"Box Plot: {y_column}")
                        
                    elif chart_type == "Mapa de Calor":
                        if heatmap_columns and len(heatmap_columns) > 1:
                            corr_matrix = df[heatmap_columns].corr()
                            fig = px.imshow(corr_matrix, text_auto=True,
                                          title="Mapa de Calor - Matriz de Correlación")
                        else:
                            st.warning("Selecciona al menos 2 columnas numéricas")
                            fig = None
                        
                    elif chart_type == "Gráfico de Área":
                        fig = px.area(df, x=x_column, y=y_column, color=color_column,
                                    title=f"Gráfico de Área: {y_column} vs {x_column}")
                    
                    if fig is not None:
                        # Configuraciones generales del gráfico
                        fig.update_layout(
                            height=500,
                            showlegend=True,
                            hovermode='x unified'
                        )
                        
                        # Almacenar el gráfico en session state
                        st.session_state.current_figure = fig
                        
                        # Mostrar el gráfico
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error al crear el gráfico: {str(e)}")
                    st.info("Verifica que las columnas seleccionadas sean compatibles con el tipo de gráfico.")
        
        with tab2:
            st.header("🎨 Personalizar Gráfico")
            
            if 'current_figure' in st.session_state:
                fig = st.session_state.current_figure
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Títulos y Etiquetas")
                    
                    # Título principal
                    new_title = st.text_input("Título del gráfico:", value=fig.layout.title.text or "")
                    
                    # Etiquetas de ejes
                    new_xlabel = st.text_input("Etiqueta eje X:", value="")
                    new_ylabel = st.text_input("Etiqueta eje Y:", value="")
                    
                    st.subheader("🎨 Colores y Estilo")
                    
                    # Tema del gráfico
                    theme = st.selectbox("Tema:", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
                    
                    # Paleta de colores
                    color_palette = st.selectbox("Paleta de colores:", 
                                               ["Default", "viridis", "plasma", "inferno", "magma", "cividis"])
                
                with col2:
                    st.subheader("📐 Dimensiones")
                    
                    # Tamaño del gráfico
                    chart_width = st.slider("Ancho:", 400, 1200, 800)
                    chart_height = st.slider("Alto:", 300, 800, 500)
                    
                    st.subheader("📊 Opciones Avanzadas")
                    
                    # Mostrar/ocultar leyenda
                    show_legend = st.checkbox("Mostrar leyenda", value=True)
                    
                    # Mostrar/ocultar grilla
                    show_grid = st.checkbox("Mostrar grilla", value=True)
                    
                    # Orientación de las etiquetas del eje X
                    x_label_angle = st.slider("Ángulo etiquetas X:", 0, 90, 0)
                
                # Aplicar personalizaciones
                if st.button("🔄 Aplicar Cambios"):
                    # Actualizar título
                    if new_title:
                        fig.update_layout(title=new_title)
                    
                    # Actualizar etiquetas de ejes
                    if new_xlabel:
                        fig.update_xaxes(title_text=new_xlabel)
                    if new_ylabel:
                        fig.update_yaxes(title_text=new_ylabel)
                    
                    # Actualizar tema
                    fig.update_layout(template=theme)
                    
                    # Actualizar dimensiones
                    fig.update_layout(width=chart_width, height=chart_height)
                    
                    # Mostrar/ocultar leyenda
                    fig.update_layout(showlegend=show_legend)
                    
                    # Configurar grilla
                    fig.update_xaxes(showgrid=show_grid)
                    fig.update_yaxes(showgrid=show_grid)
                    
                    # Ángulo de etiquetas X
                    fig.update_xaxes(tickangle=x_label_angle)
                    
                    # Actualizar el gráfico en session state
                    st.session_state.current_figure = fig
                    
                    st.success("✅ Cambios aplicados")
                    st.rerun()
                
                # Mostrar gráfico personalizado
                st.subheader("📊 Gráfico Personalizado")
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Crea un gráfico primero en la pestaña 'Crear Gráfico'")
        
        with tab3:
            st.header("📥 Exportar Gráfico")
            
            if 'current_figure' in st.session_state:
                fig = st.session_state.current_figure
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💾 Descargar Imagen")
                    
                    # Formato de imagen
                    image_format = st.selectbox("Formato:", ["png", "jpg", "pdf", "svg"])
                    
                    # Calidad/resolución
                    if image_format in ["png", "jpg"]:
                        scale = st.slider("Escala (calidad):", 1, 5, 2)
                    
                    # Nombre del archivo
                    filename = st.text_input("Nombre del archivo:", value="grafico")
                    
                    st.info("💡 Usa el menú de la gráfica (botón de cámara) para descargar directamente")
                
                with col2:
                    st.subheader("🔗 Compartir")
                    
                    # Código HTML embebido
                    if st.checkbox("Generar código HTML"):
                        html_code = fig.to_html(include_plotlyjs='cdn')
                        st.code(html_code, language='html')
                        
                        st.download_button(
                            label="📥 Descargar HTML",
                            data=html_code,
                            file_name=f"{filename}.html",
                            mime="text/html"
                        )
                    
                    # JSON del gráfico
                    if st.checkbox("Exportar configuración JSON"):
                        json_data = fig.to_json()
                        st.download_button(
                            label="📥 Descargar JSON",
                            data=json_data,
                            file_name=f"{filename}_config.json",
                            mime="application/json"
                        )
                
                # Galería de gráficos guardados
                st.subheader("🖼️ Galería de Gráficos")
                
                if 'saved_charts' not in st.session_state:
                    st.session_state.saved_charts = []
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    chart_name = st.text_input("Nombre para guardar:", value=f"Gráfico_{len(st.session_state.saved_charts)+1}")
                with col2:
                    if st.button("💾 Guardar en Galería"):
                        chart_info = {
                            'name': chart_name,
                            'figure': fig,
                            'type': chart_type
                        }
                        st.session_state.saved_charts.append(chart_info)
                        st.success(f"✅ '{chart_name}' guardado en la galería")
                
                # Mostrar galería
                if st.session_state.saved_charts:
                    st.markdown("**Gráficos Guardados:**")
                    for i, chart in enumerate(st.session_state.saved_charts):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"📊 {chart['name']} ({chart['type']})")
                        with col2:
                            if st.button("👁️ Ver", key=f"view_{i}"):
                                st.plotly_chart(chart['figure'], use_container_width=True)
                        with col3:
                            if st.button("🗑️ Eliminar", key=f"delete_{i}"):
                                st.session_state.saved_charts.pop(i)
                                st.rerun()
            
            else:
                st.info("Crea y personaliza un gráfico primero")
    
    # Información adicional en el sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Usa gráficos de barras para comparaciones
        - Líneas son ideales para tendencias temporales
        - Dispersión muestra relaciones entre variables
        - Box plots revelan distribuciones
        - Mapas de calor muestran correlaciones
        """)
        
        if data_source is not None:
            st.markdown("### 📊 Info del Dataset")
            st.write(f"Filas: {data_source.shape[0]}")
            st.write(f"Columnas: {data_source.shape[1]}")