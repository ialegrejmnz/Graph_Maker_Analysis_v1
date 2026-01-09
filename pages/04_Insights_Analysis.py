import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Insight Analysis",
    page_icon="🔍",
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
    """Página de análisis de insights automático"""
    
    # Título de la página
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🔍 Insight Analysis
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
        st.info("ℹ️ Usando datos originales")
    else:
        st.warning("⚠️ No hay datos disponibles. Ve a 'Input Selection and Management' para cargar datos.")
        
        # Generar datos de ejemplo para análisis
        if st.button("🎲 Generar Dataset de Ejemplo para Análisis"):
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'ventas': np.random.normal(1000, 300, 200),
                'marketing_gasto': np.random.normal(500, 150, 200),
                'satisfaccion_cliente': np.random.uniform(1, 10, 200),
                'región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 200),
                'categoria_producto': np.random.choice(['A', 'B', 'C'], 200),
                'mes': np.random.choice(range(1, 13), 200),
                'empleados': np.random.randint(10, 100, 200),
                'precio': np.random.uniform(50, 500, 200)
            })
            # Crear algunas correlaciones realistas
            sample_data['ventas'] = sample_data['ventas'] + sample_data['marketing_gasto'] * 0.5
            sample_data['satisfaccion_cliente'] = 5 + (sample_data['ventas'] / 1000) * 2 + np.random.normal(0, 1, 200)
            
            st.session_state.uploaded_data = sample_data
            data_source = sample_data
            st.success("✅ Dataset de ejemplo generado con correlaciones")
            st.rerun()
    
    if data_source is not None:
        df = data_source.copy()
        
        # Tabs para diferentes tipos de análisis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Resumen Ejecutivo", 
            "🔍 Análisis Descriptivo", 
            "📈 Correlaciones", 
            "🎯 Segmentación", 
            "📋 Recomendaciones"
        ])
        
        with tab1:
            st.header("📊 Resumen Ejecutivo")
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            with col1:
                st.metric("📈 Total de Registros", f"{len(df):,}")
            with col2:
                st.metric("🔢 Variables Numéricas", len(numeric_cols))
            with col3:
                st.metric("🏷️ Variables Categóricas", len(categorical_cols))
            with col4:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("❓ % Datos Faltantes", f"{missing_pct:.1f}%")
            
            st.markdown("---")
            
            # Insights automáticos principales
            st.subheader("🎯 Insights Principales")
            
            insights = []
            
            # Análisis de valores nulos
            if df.isnull().sum().sum() > 0:
                null_cols = df.isnull().sum()
                null_cols = null_cols[null_cols > 0].sort_values(ascending=False)
                worst_col = null_cols.index[0]
                worst_pct = (null_cols.iloc[0] / len(df)) * 100
                insights.append(f"🚨 La columna '{worst_col}' tiene {worst_pct:.1f}% de valores faltantes")
            
            # Análisis de outliers en variables numéricas
            if len(numeric_cols) > 0:
                outlier_counts = {}
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                    outlier_counts[col] = len(outliers)
                
                if outlier_counts:
                    max_outlier_col = max(outlier_counts, key=outlier_counts.get)
                    max_outlier_count = outlier_counts[max_outlier_col]
                    if max_outlier_count > 0:
                        outlier_pct = (max_outlier_count / len(df)) * 100
                        insights.append(f"📊 '{max_outlier_col}' tiene {max_outlier_count} outliers ({outlier_pct:.1f}%)")
            
            # Análisis de correlaciones altas
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                # Encontrar correlaciones altas (>0.7 o <-0.7)
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                            high_corr.append((col1, col2, corr_val))
                
                if high_corr:
                    best_corr = max(high_corr, key=lambda x: abs(x[2]))
                    insights.append(f"🔗 Correlación fuerte entre '{best_corr[0]}' y '{best_corr[1]}' (r={best_corr[2]:.2f})")
            
            # Análisis de distribuciones asimétricas
            if len(numeric_cols) > 0:
                skewed_vars = []
                for col in numeric_cols:
                    skewness = stats.skew(df[col].dropna())
                    if abs(skewness) > 1:
                        skewed_vars.append((col, skewness))
                
                if skewed_vars:
                    most_skewed = max(skewed_vars, key=lambda x: abs(x[1]))
                    direction = "positivamente" if most_skewed[1] > 0 else "negativamente"
                    insights.append(f"📐 '{most_skewed[0]}' está {direction} sesgada (skew={most_skewed[1]:.2f})")
            
            # Mostrar insights
            if insights:
                for i, insight in enumerate(insights[:5], 1):  # Mostrar máximo 5
                    st.write(f"{i}. {insight}")
            else:
                st.info("No se detectaron patrones anómalos significativos en los datos.")
            
            # Gráfico resumen de distribuciones
            if len(numeric_cols) > 0:
                st.subheader("📊 Distribución de Variables Numéricas")
                
                # Seleccionar hasta 4 variables para mostrar
                cols_to_show = numeric_cols[:4]
                
                fig = make_subplots(rows=2, cols=2, 
                                  subplot_titles=[f"Distribución de {col}" for col in cols_to_show])
                
                for i, col in enumerate(cols_to_show):
                    row = (i // 2) + 1
                    col_pos = (i % 2) + 1
                    
                    fig.add_trace(
                        go.Histogram(x=df[col], name=col, showlegend=False),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(height=500, title_text="Vista General de Distribuciones")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("🔍 Análisis Descriptivo Detallado")
            
            # Selector de variable para análisis detallado
            analysis_type = st.radio("Tipo de análisis:", ["Variables Numéricas", "Variables Categóricas"])
            
            if analysis_type == "Variables Numéricas" and len(numeric_cols) > 0:
                selected_var = st.selectbox("Selecciona variable numérica:", numeric_cols)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Histograma con estadísticas
                    fig = px.histogram(df, x=selected_var, marginal="box",
                                     title=f"Distribución de {selected_var}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Test de normalidad
                    statistic, p_value = stats.normaltest(df[selected_var].dropna())
                    st.subheader("🧪 Test de Normalidad (D'Agostino)")
                    if p_value > 0.05:
                        st.success(f"✅ Distribución normal (p-valor: {p_value:.4f})")
                    else:
                        st.warning(f"⚠️ Distribución no normal (p-valor: {p_value:.4f})")
                
                with col2:
                    # Estadísticas descriptivas detalladas
                    st.subheader("📊 Estadísticas")
                    
                    stats_data = df[selected_var].describe()
                    
                    for stat, value in stats_data.items():
                        if stat == 'count':
                            st.metric(stat.title(), f"{int(value):,}")
                        else:
                            st.metric(stat.title(), f"{value:.2f}")
                    
                    # Estadísticas adicionales
                    st.metric("Asimetría", f"{stats.skew(df[selected_var].dropna()):.3f}")
                    st.metric("Curtosis", f"{stats.kurtosis(df[selected_var].dropna()):.3f}")
                    
                    # Percentiles adicionales
                    st.subheader("📈 Percentiles")
                    percentiles = [10, 90, 95, 99]
                    for p in percentiles:
                        value = np.percentile(df[selected_var].dropna(), p)
                        st.metric(f"P{p}", f"{value:.2f}")
            
            elif analysis_type == "Variables Categóricas" and len(categorical_cols) > 0:
                selected_var = st.selectbox("Selecciona variable categórica:", categorical_cols)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Gráfico de barras de frecuencias
                    value_counts = df[selected_var].value_counts()
                    fig = px.bar(x=value_counts.values, y=value_counts.index,
                               orientation='h', title=f"Frecuencia de {selected_var}")
                    fig.update_layout(yaxis_title=selected_var, xaxis_title="Frecuencia")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Análisis de diversidad
                    unique_values = df[selected_var].nunique()
                    most_common = df[selected_var].mode().iloc[0]
                    most_common_pct = (df[selected_var] == most_common).mean() * 100
                    
                    st.info(f"📈 La categoría más común es '{most_common}' ({most_common_pct:.1f}%)")
                
                with col2:
                    st.subheader("📊 Análisis de Frecuencia")
                    
                    # Tabla de frecuencias
                    freq_table = df[selected_var].value_counts()
                    freq_df = pd.DataFrame({
                        'Categoría': freq_table.index,
                        'Frecuencia': freq_table.values,
                        'Porcentaje': (freq_table.values / len(df) * 100).round(1)
                    })
                    
                    st.dataframe(freq_df, use_container_width=True)
                    
                    # Métricas de diversidad
                    st.subheader("🎯 Métricas")
                    st.metric("Valores únicos", unique_values)
                    st.metric("Moda", most_common)
                    st.metric("% de la moda", f"{most_common_pct:.1f}%")
                    
                    # Índice de diversidad (entropía)
                    entropy = stats.entropy(freq_table.values)
                    st.metric("Entropía", f"{entropy:.2f}")
            
            else:
                st.warning("No hay variables del tipo seleccionado para analizar.")
        
        with tab3:
            st.header("📈 Análisis de Correlaciones")
            
            if len(numeric_cols) > 1:
                # Matriz de correlación
                st.subheader("🔗 Matriz de Correlación")
                
                corr_method = st.selectbox("Método de correlación:", ["pearson", "spearman", "kendall"])
                corr_matrix = df[numeric_cols].corr(method=corr_method)
                
                # Mapa de calor de correlaciones
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title=f"Matriz de Correlación ({corr_method.title()})",
                              color_continuous_scale="RdBu_r")
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlaciones
                st.subheader("🎯 Correlaciones Más Fuertes")
                
                # Extraer pares de correlaciones
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        var1 = corr_matrix.columns[i]
                        var2 = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        correlations.append({
                            'Variable 1': var1,
                            'Variable 2': var2,
                            'Correlación': corr_val,
                            'Fuerza': 'Fuerte' if abs(corr_val) > 0.7 else 'Moderada' if abs(corr_val) > 0.5 else 'Débil'
                    })
                
                # Ordenar por valor absoluto de correlación
                correlations_df = pd.DataFrame(correlations)
                correlations_df = correlations_df.reindex(
                    correlations_df['Correlación'].abs().sort_values(ascending=False).index
                )
                
                # Mostrar top 10 correlaciones
                st.dataframe(correlations_df.head(10), use_container_width=True)
                
                # Análisis detallado de una correlación específica
                st.subheader("🔍 Análisis de Correlación Específica")
                
                if len(correlations_df) > 0:
                    selected_pair = st.selectbox(
                        "Selecciona un par de variables:",
                        [f"{row['Variable 1']} vs {row['Variable 2']}" for _, row in correlations_df.iterrows()]
                    )
                    
                    if selected_pair:
                        var1, var2 = selected_pair.split(" vs ")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Gráfico de dispersión
                            fig = px.scatter(df, x=var1, y=var2, 
                                           trendline="ols",
                                           title=f"Relación entre {var1} y {var2}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Estadísticas de la correlación
                            corr_val = df[var1].corr(df[var2])
                            st.metric("Correlación de Pearson", f"{corr_val:.3f}")
                            
                            # R-cuadrado
                            r_squared = corr_val ** 2
                            st.metric("R²", f"{r_squared:.3f}")
                            st.caption(f"{r_squared*100:.1f}% de la varianza explicada")
                            
                            # Test de significancia
                            stat, p_val = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                            st.metric("P-valor", f"{p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.success("✅ Correlación significativa")
                            else:
                                st.warning("⚠️ Correlación no significativa")
            
            else:
                st.warning("Se necesitan al menos 2 variables numéricas para el análisis de correlación.")
        
        with tab4:
            st.header("🎯 Análisis de Segmentación")
            
            if len(numeric_cols) > 1:
                st.subheader("🔍 Segmentación por K-Means")
                
                # Selección de variables para clustering
                cluster_vars = st.multiselect(
                    "Selecciona variables para segmentación:",
                    numeric_cols,
                    default=list(numeric_cols[:3]) if len(numeric_cols) >= 3 else list(numeric_cols)
                )
                
                if len(cluster_vars) >= 2:
                    # Número de clusters
                    n_clusters = st.slider("Número de segmentos:", 2, 8, 3)
                    
                    if st.button("🔍 Realizar Segmentación"):
                        # Preparar datos
                        data_for_clustering = df[cluster_vars].dropna()
                        
                        # Normalizar datos
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(data_for_clustering)
                        
                        # Aplicar K-Means
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(data_scaled)
                        
                        # Añadir clusters al dataframe
                        df_clustered = data_for_clustering.copy()
                        df_clustered['Segmento'] = clusters
                        
                        # Guardar en session state
                        st.session_state.segmented_data = df_clustered
                        st.session_state.cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Visualización de clusters (2D)
                            if len(cluster_vars) >= 2:
                                fig = px.scatter(df_clustered, 
                                               x=cluster_vars[0], 
                                               y=cluster_vars[1],
                                               color='Segmento',
                                               title="Segmentación de Clientes")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Estadísticas por segmento
                            st.subheader("📊 Resumen por Segmento")
                            segment_summary = df_clustered.groupby('Segmento').agg({
                                cluster_vars[0]: ['mean', 'count']
                            }).round(2)
                            
                            for segment in range(n_clusters):
                                segment_data = df_clustered[df_clustered['Segmento'] == segment]
                                st.write(f"**Segmento {segment}** ({len(segment_data)} registros)")
                                
                                for var in cluster_vars[:3]:  # Mostrar máximo 3 variables
                                    mean_val = segment_data[var].mean()
                                    st.write(f"- {var}: {mean_val:.1f}")
                        
                        # Análisis de características por segmento
                        st.subheader("🔍 Perfil de Segmentos")
                        
                        segment_profiles = []
                        for segment in range(n_clusters):
                            segment_data = df_clustered[df_clustered['Segmento'] == segment]
                            profile = {
                                'Segmento': f'Segmento {segment}',
                                'Tamaño': len(segment_data),
                                'Porcentaje': f"{len(segment_data)/len(df_clustered)*100:.1f}%"
                            }
                            
                            # Características principales
                            for var in cluster_vars:
                                profile[f'{var}_media'] = segment_data[var].mean()
                            
                            segment_profiles.append(profile)
                        
                        profiles_df = pd.DataFrame(segment_profiles)
                        st.dataframe(profiles_df, use_container_width=True)
                
                # Análisis PCA para visualización
                st.subheader("📊 Análisis de Componentes Principales (PCA)")
                
                if len(numeric_cols) > 2:
                    if st.button("🔍 Realizar PCA"):
                        # Preparar datos
                        pca_data = df[numeric_cols].dropna()
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(pca_data)
                        
                        # Aplicar PCA
                        pca = PCA()
                        components = pca.fit_transform(data_scaled)
                        
                        # Crear dataframe con componentes
                        pca_df = pd.DataFrame(
                            components[:, :3], 
                            columns=['PC1', 'PC2', 'PC3']
                        )
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Gráfico 3D de componentes principales
                            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                                              title="Análisis de Componentes Principales")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Varianza explicada
                            st.subheader("📈 Varianza Explicada")
                            
                            explained_var = pca.explained_variance_ratio_
                            cumulative_var = explained_var.cumsum()
                            
                            for i, (var, cum_var) in enumerate(zip(explained_var[:5], cumulative_var[:5])):
                                st.metric(
                                    f"PC{i+1}", 
                                    f"{var:.1%}",
                                    help=f"Acumulada: {cum_var:.1%}"
                                )
                            
                            # Componentes más importantes
                            st.subheader("🎯 Variables Importantes")
                            feature_importance = pd.DataFrame(
                                pca.components_[:3].T,
                                columns=['PC1', 'PC2', 'PC3'],
                                index=numeric_cols
                            ).abs()
                            
                            st.dataframe(feature_importance.round(3), use_container_width=True)
            
            else:
                st.warning("Se necesitan al menos 2 variables numéricas para segmentación.")
        
        with tab5:
            st.header("📋 Recomendaciones y Conclusiones")
            
            # Generar recomendaciones automáticas basadas en el análisis
            st.subheader("🎯 Recomendaciones Automáticas")
            
            recommendations = []
            
            # Recomendaciones basadas en correlaciones
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                if high_corr_pairs:
                    best_corr = max(high_corr_pairs, key=lambda x: abs(x[2]))
                    recommendations.append({
                        'Tipo': 'Correlación',
                        'Prioridad': 'Alta',
                        'Recomendación': f"Explorar la fuerte relación entre '{best_corr[0]}' y '{best_corr[1]}' (r={best_corr[2]:.2f}). Considera si una variable puede predecir la otra.",
                        'Acción': f"Crear un modelo predictivo usando '{best_corr[0]}' para estimar '{best_corr[1]}'"
                    })
            
            # Recomendaciones basadas en outliers
            outlier_recommendations = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                outlier_pct = len(outliers) / len(df) * 100
                
                if outlier_pct > 5:
                    recommendations.append({
                        'Tipo': 'Calidad de Datos',
                        'Prioridad': 'Media',
                        'Recomendación': f"La variable '{col}' tiene {outlier_pct:.1f}% de outliers. Investiga si son errores o casos especiales.",
                        'Acción': f"Revisar y validar los valores extremos en '{col}'"
                    })
            
            # Recomendaciones basadas en valores faltantes
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                worst_missing = missing_data.idxmax()
                missing_pct = (missing_data.max() / len(df)) * 100
                
                if missing_pct > 20:
                    recommendations.append({
                        'Tipo': 'Calidad de Datos',
                        'Prioridad': 'Alta',
                        'Recomendación': f"La variable '{worst_missing}' tiene {missing_pct:.1f}% de datos faltantes. Considera estrategias de imputación o eliminación.",
                        'Acción': f"Desarrollar estrategia para manejar valores faltantes en '{worst_missing}'"
                    })
            
            # Recomendaciones de visualización
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                recommendations.append({
                    'Tipo': 'Visualización',
                    'Prioridad': 'Media',
                    'Recomendación': "Crear gráficos segmentados por variables categóricas para identificar patrones por grupos.",
                    'Acción': f"Usar '{categorical_cols[0]}' para segmentar análisis de '{numeric_cols[0]}'"
                })
            
            # Mostrar recomendaciones
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"💡 Recomendación {i}: {rec['Tipo']} - Prioridad {rec['Prioridad']}"):
                        st.write(f"**Observación:** {rec['Recomendación']}")
                        st.write(f"**Acción sugerida:** {rec['Acción']}")
            else:
                st.info("No se generaron recomendaciones automáticas para este dataset.")
            
            # Plan de acción
            st.subheader("📋 Plan de Acción Sugerido")
            
            action_plan = [
                "🔍 **Fase 1: Limpieza de Datos**",
                "- Manejar valores faltantes según las recomendaciones",
                "- Investigar y tratar outliers identificados",
                "- Validar la consistencia de los datos",
                "",
                "📊 **Fase 2: Análisis Exploratorio**",
                "- Profundizar en las correlaciones significativas",
                "- Crear visualizaciones segmentadas por categorías",
                "- Realizar análisis de tendencias temporales (si aplica)",
                "",
                "🎯 **Fase 3: Modelado y Segmentación**",
                "- Implementar modelos predictivos basados en correlaciones fuertes",
                "- Refinar la segmentación de clientes/entidades",
                "- Validar los segmentos con expertos de negocio",
                "",
                "📈 **Fase 4: Implementación**",
                "- Crear dashboards automáticos",
                "- Establecer métricas de seguimiento",
                "- Programar análisis periódicos"
            ]
            
            for step in action_plan:
                if step.startswith("🔍") or step.startswith("📊") or step.startswith("🎯") or step.startswith("📈"):
                    st.markdown(f"### {step}")
                elif step:
                    st.markdown(step)
            
            # Resumen de insights clave
            st.subheader("🔑 Insights Clave para Recordar")
            
            key_insights = []
            
            # Dataset overview
            key_insights.append(f"📊 Dataset con {len(df):,} registros y {len(df.columns)} variables")
            
            # Best correlations
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                max_corr = 0
                max_pair = None
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > max_corr:
                            max_corr = corr_val
                            max_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                
                if max_pair and max_corr > 0.5:
                    key_insights.append(f"🔗 Correlación más fuerte: {max_pair[0]} - {max_pair[1]} ({max_corr:.2f})")
            
            # Data quality
            if df.isnull().sum().sum() > 0:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                key_insights.append(f"❓ {missing_pct:.1f}% de datos faltantes en el dataset")
            
            # Most variable column
            if len(numeric_cols) > 0:
                cv_values = {}
                for col in numeric_cols:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if mean_val != 0:
                        cv_values[col] = abs(std_val / mean_val)
                
                if cv_values:
                    most_variable = max(cv_values, key=cv_values.get)
                    key_insights.append(f"📈 Variable con mayor variabilidad: {most_variable}")
            
            # Show key insights
            for insight in key_insights:
                st.write(f"• {insight}")
            
            # Export report button
            st.subheader("📄 Generar Reporte")
            
            if st.button("📋 Generar Reporte Completo"):
                # Create a comprehensive report
                report_content = f"""
# Reporte de Análisis de Datos
Generado automáticamente por Graph Maker Analysis

## Resumen Ejecutivo
- **Total de registros:** {len(df):,}
- **Variables numéricas:** {len(numeric_cols)}
- **Variables categóricas:** {len(categorical_cols)}
- **Datos faltantes:** {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%

## Principales Hallazgos
"""
                
                for insight in key_insights:
                    report_content += f"- {insight}\n"
                
                report_content += "\n## Recomendaciones\n"
                
                for i, rec in enumerate(recommendations, 1):
                    report_content += f"{i}. **{rec['Tipo']}** - {rec['Recomendación']}\n"
                
                st.download_button(
                    label="📥 Descargar Reporte (Markdown)",
                    data=report_content,
                    file_name="reporte_analisis.md",
                    mime="text/markdown"
                )
                
                st.success("✅ Reporte generado exitosamente")
    
    # Información en el sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧠 Análisis Disponibles")
        st.markdown("""
        - **Resumen Ejecutivo**: Vista general y métricas clave
        - **Análisis Descriptivo**: Estadísticas detalladas por variable
        - **Correlaciones**: Relaciones entre variables numéricas
        - **Segmentación**: K-means clustering y PCA
        - **Recomendaciones**: Insights automáticos y plan de acción
        """)
        
        if data_source is not None:
            st.markdown("### 📊 Info del Dataset Actual")
            st.write(f"📋 Filas: {data_source.shape[0]:,}")
            st.write(f"📊 Columnas: {data_source.shape[1]}")
            st.write(f"💾 Memoria: {data_source.memory_usage(deep=True).sum() / 1024:.1f} KB")

if __name__ == "__main__":
    main()
    show_page()