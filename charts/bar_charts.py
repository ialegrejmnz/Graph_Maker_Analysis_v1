import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import scipy for smooth interpolation
try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from charts.insights_functions import (
    AVAILABLE_ESTIMATORS,
    generate_multilevel_aggregations,
    add_pareto_insights,
    generate_comparison_insights,
    generate_strategic_insights
)

from charts.common_functions import (
    get_scale_and_format_eur,
    generate_intelligent_ylabel,
    setup_chart_style
)

def format_number_with_units(value):
    """
    Formats a number with appropriate units (Bn, Mn, K) and two decimal places.

    Parameters:
    -----------
    value : float
        The number to format

    Returns:
    --------
    str
        Formatted number string
    """
    abs_value = abs(value)

    if abs_value >= 1e9:
        formatted = f"{value / 1e9:,.2f}Bn"
    elif abs_value >= 1e6:
        formatted = f"{value / 1e6:,.2f}Mn"
    elif abs_value >= 1e3:
        formatted = f"{value / 1e3:,.2f}K"
    else:
        formatted = f"{value:,.2f}"

    return formatted

def plot_financial_barplot(df, numeric_column, categorical_column, estimator='mean',
                          figsize=(10, 6), rotation=45, mean_col=False,
                          percentile_range=(0, 1)):
    """
    Creates a bar plot for financial data grouped by category with enhanced formatting.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the financial data
    numeric_column : str
        Name of the numeric column to analyze
    categorical_column : str
        Name of the categorical column for grouping
    estimator : str, default 'mean'
        Statistical estimator to apply. Options:
        - 'mean': Arithmetic mean
        - 'sum': Total sum
        - 'median': Median value
        - 'std': Standard deviation
        - 'var': Variance
        - 'min': Minimum value
        - 'max': Maximum value
        - 'count': Count of observations
    figsize : tuple, default (10, 6)
        Figure size (width, height)
    rotation : int, default 45
        Rotation angle for x-axis labels
    mean_col : bool, default False
        If True, adds an additional bar showing the overall mean of the numeric column
    percentile_range : tuple, default (0, 1)
        Range of percentiles to include in the analysis (min_percentile, max_percentile)
        Values should be between 0 and 1. Default (0, 1) includes all data.

    Returns:
    --------
    tuple
        (matplotlib.axes.Axes, dict): Axes object of the created plot and JSON with category values
    """

    # Validations
    if numeric_column not in df.columns:
        raise ValueError(f"Column '{numeric_column}' does not exist in DataFrame")

    if categorical_column not in df.columns:
        raise ValueError(f"Column '{categorical_column}' does not exist in DataFrame")

    if estimator not in AVAILABLE_ESTIMATORS:
        raise ValueError(f"Invalid estimator '{estimator}'. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        raise ValueError(f"Column '{numeric_column}' must be numeric")

    # Validate percentile_range
    if not isinstance(percentile_range, tuple) or len(percentile_range) != 2:
        raise ValueError("percentile_range must be a tuple of length 2")

    min_percentile, max_percentile = percentile_range
    if not (0 <= min_percentile <= max_percentile <= 1):
        raise ValueError("Percentile values must be between 0 and 1, and min_percentile <= max_percentile")

    # Remove null values
    df_clean = df[[numeric_column, categorical_column]].dropna()

    # Apply percentile filtering
    if percentile_range != (0, 1):
        lower_bound = df_clean[numeric_column].quantile(min_percentile)
        upper_bound = df_clean[numeric_column].quantile(max_percentile)
        df_clean = df_clean[
            (df_clean[numeric_column] >= lower_bound) &
            (df_clean[numeric_column] <= upper_bound)
        ]

    # Calculate estimator by category
    grouped_data = df_clean.groupby(categorical_column)[numeric_column].apply(AVAILABLE_ESTIMATORS[estimator])

    # Format category names (primera letra mayúscula, resto minúsculas)
    formatted_categories = [str(cat).capitalize() for cat in grouped_data.index]

    # Prepare JSON result dictionary
    estimator_name = estimator.capitalize()
    result_json = {}

    # Add category results to JSON
    for original_cat, formatted_cat, value in zip(grouped_data.index, formatted_categories, grouped_data.values):
        key = f"{estimator_name} {numeric_column} for {categorical_column} {original_cat}"
        result_json[key] = value

    # Calculate overall mean if mean_col is True
    if mean_col:
        overall_mean = df_clean[numeric_column].apply(AVAILABLE_ESTIMATORS[estimator]) if estimator != 'mean' else df_clean[numeric_column].mean()
        # For most estimators, we need to apply them to the entire column
        if estimator in ['sum', 'count']:
            overall_stat = df_clean[numeric_column].apply(AVAILABLE_ESTIMATORS[estimator])
        elif estimator == 'mean':
            overall_stat = df_clean[numeric_column].mean()
        elif estimator == 'median':
            overall_stat = df_clean[numeric_column].median()
        elif estimator == 'std':
            overall_stat = df_clean[numeric_column].std()
        elif estimator == 'var':
            overall_stat = df_clean[numeric_column].var()
        elif estimator == 'min':
            overall_stat = df_clean[numeric_column].min()
        elif estimator == 'max':
            overall_stat = df_clean[numeric_column].max()
        else:
            overall_stat = df_clean[numeric_column].mean()  # fallback

        # Add mean to the data
        grouped_data_with_mean = grouped_data.copy()
        grouped_data_with_mean['Overall Mean'] = overall_stat
        formatted_categories_with_mean = formatted_categories + ['Overall Mean']

        # Add overall stat to JSON
        result_json[f"{estimator_name} Overall {numeric_column}"] = overall_stat
    else:
        grouped_data_with_mean = grouped_data
        formatted_categories_with_mean = formatted_categories

    # Add comparison keys to the insights JSON
    result_full_json = generate_comparison_insights(result_json)

    # Configure style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)

    # Setup transparent background and styling
    fig.patch.set_alpha(0)
    ax = setup_chart_style(ax)

    # Determine consistent scale for all values (including mean if applicable)
    all_values = list(grouped_data_with_mean.values)
    max_value = max(all_values)
    min_value = min(all_values)
    abs_max = max(abs(max_value), abs(min_value))

    # Get consistent scale
    scale, suffix = get_scale_and_format_eur(all_values, numeric_column)

    # Create bar plot with more separation from baseline
    baseline_offset = abs_max * 0.05  # 5% of max value as offset

    # Adjust y-limits to create separation
    y_min = min(all_values) - baseline_offset
    y_max = max(all_values) + abs_max * 0.1  # Extra space on top

    # Create bars with different colors
    colors = []
    for i, category in enumerate(formatted_categories_with_mean):
        if mean_col and category == 'Overall Mean':
            colors.append('#FF6B6B')  # Different color for mean bar (coral/red)
        else:
            colors.append('#7D4BEB')  # Original purple color

    bars = ax.bar(formatted_categories_with_mean, grouped_data_with_mean.values,
                  color=colors, alpha=0.8, edgecolor='white', linewidth=1)

    # Add thin black line at x-axis
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.8)

    # Add values on top of bars with new formatting
    for bar, value in zip(bars, grouped_data_with_mean.values):
        height = bar.get_height()
        formatted_value = format_number_with_units(height)
        ax.text(bar.get_x() + bar.get_width()/2., height,
                formatted_value, ha='center', va='bottom', fontsize=10)

    # Generate intelligent Y-axis label
    label, unit, year = generate_intelligent_ylabel(numeric_column, estimator)

    # Add scale suffix to unit for EUR columns if applicable
    if 'EUR' in numeric_column.upper() and suffix and estimator != 'count':
        unit = f'€ {suffix}' if suffix else '€'

    # Construct final label
    if year and unit:
        ylabel = f'{label} ({year}, {unit})'
    elif unit:
        ylabel = f'{label} ({unit})'
    elif year:
        ylabel = f'{label} ({year})'
    else:
        ylabel = label

    # Set labels
    ax.set_xlabel('')  # No title on x-axis
    ax.set_ylabel(ylabel, fontsize=12)

    # Set y-limits with separation
    ax.set_ylim(y_min, y_max)

    # Rotate x-axis labels if needed
    if rotation > 0:
        plt.xticks(rotation=rotation, ha='right')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    return ax, result_full_json

# charts/bar_charts.py
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import uuid

class BarChartComponent:
    """Componente para crear y gestionar Bar Charts"""
    
    def __init__(self):
        self.chart_type = "Bar Chart"
        
    def render(self):
        """Renderiza la interfaz completa del Bar Chart"""
        
        # Header y descripción
        st.subheader("📊 Bar Chart")
        st.markdown("""
        **Description:** Create financial bar charts with statistical estimators and customizable parameters.
        
        **How it works:**
        - Choose a **numeric variable** as your main variable (what to measure)
        - Choose **categorical variables** as extra variables (how to group the data)  
        - One chart will be created for each combination of main variable × extra variable
        """)
        
        # Mostrar parámetros requeridos
        with st.expander("📋 Parameters Guide", expanded=False):
            st.markdown("""
            **Main Variable:** Numeric column to aggregate (e.g., Revenue EUR 2024, EBITDA EUR 2024)
            
            **Extra Variables:** Categorical columns for grouping (e.g., Country ISO code, Industry)
            
            **Additional Parameters:**
            - **Estimator:** How to aggregate data (mean, sum, median, count, etc.)
            - **Figure Size:** Width and height of the chart
            - **Label Rotation:** Angle for x-axis labels (0-90 degrees)
            - **Show Overall Mean:** Add a column showing overall average
            - **Percentile Range:** Filter outliers by percentile range
            """)
        
        st.divider()
        
        # Sección 1: Selección de variables
        self._render_variable_selection()
        
        # Sección 2: Parámetros (solo si hay variables seleccionadas)
        main_var = st.session_state.get('bar_main_variable')
        extra_vars = st.session_state.get('bar_extra_variables', [])
        
        if main_var and extra_vars:
            st.divider()
            self._render_parameters_section()
            
            st.divider()
            self._render_generation_section()
        
        # Sección 3: Mostrar gráficas generadas
        self._render_generated_charts()
    
    def _render_variable_selection(self):
        """Renderiza la sección de selección de variables"""
        
        st.markdown("### 📊 Variable Selection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Main variable (numeric)
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            main_variable = st.selectbox(
                "Main Variable (Numeric)",
                options=[None] + numeric_cols,
                key="bar_main_variable",
                help="Select the numeric variable you want to analyze"
            )
            
            # Extra variables (categorical)
            categorical_cols = self._get_categorical_columns()
            
            extra_variables = st.multiselect(
                "Extra Variables (Categorical)",
                options=categorical_cols,
                key="bar_extra_variables",
                help="Select categorical variables for grouping. One chart per variable will be created."
            )
        
        with col2:
            # Preview de combinaciones
            if main_variable and extra_variables:
                st.markdown("**Charts to be generated:**")
                for i, extra_var in enumerate(extra_variables, 1):
                    st.write(f"{i}. **{main_variable}** grouped by **{extra_var}**")
                
                st.info(f"📊 Total charts: **{len(extra_variables)}**")
                
                # Preview estadístico rápido
                with st.expander("📈 Data Preview", expanded=False):
                    for extra_var in extra_variables:
                        unique_count = st.session_state.df[extra_var].nunique()
                        null_count = st.session_state.df[extra_var].isnull().sum()
                        st.write(f"**{extra_var}:** {unique_count} unique values, {null_count} null values")
            else:
                st.info("👈 Select variables to see chart preview")
    
    def _get_categorical_columns(self):
        """Obtiene las columnas categóricas disponibles"""
        df = st.session_state.df
        
        # Columnas object/category
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Agregar columnas numéricas con pocos valores únicos
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            unique_count = df[col].nunique()
            if unique_count <= 20:  # Threshold para considerar como categórica
                categorical_cols.append(col)
        
        return sorted(categorical_cols)
    
    def _render_parameters_section(self):
        """Renderiza la sección de parámetros adicionales"""
        
        st.markdown("### ⚙️ Chart Parameters")
        st.write("Configure parameters that will apply to all generated charts:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.selectbox(
                "Estimator",
                options=['mean', 'sum', 'median', 'count', 'std', 'min', 'max'],
                index=0,
                key="bar_estimator",
                help="Statistical function to apply when grouping data"
            )
            
            st.slider(
                "Figure Width", 
                min_value=6, 
                max_value=20, 
                value=10, 
                key="bar_figsize_width"
            )
        
        with col2:
            st.slider(
                "Label Rotation", 
                min_value=0, 
                max_value=90, 
                value=45, 
                key="bar_rotation",
                help="Angle for x-axis labels (degrees)"
            )
            
            st.slider(
                "Figure Height", 
                min_value=4, 
                max_value=12, 
                value=6, 
                key="bar_figsize_height"
            )
        
        with col3:
            st.checkbox(
                "Show Overall Mean", 
                value=False,
                key="bar_mean_col",
                help="Add a column showing the overall average"
            )
            
            # Percentile range
            st.write("**Outlier Filtering:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.number_input(
                    "Lower %", 
                    min_value=0.0, 
                    max_value=0.5, 
                    value=0.0, 
                    step=0.01,
                    key="bar_percentile_low"
                )
            with col_b:
                st.number_input(
                    "Upper %", 
                    min_value=0.5, 
                    max_value=1.0, 
                    value=1.0, 
                    step=0.01,
                    key="bar_percentile_high"
                )
    
    def _render_generation_section(self):
        """Renderiza la sección de generación de gráficas"""
        
        st.markdown("### 🚀 Generate Charts")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📊 Generate Bar Charts", type="primary", key="generate_bar_charts"):
                self._generate_charts()
        
        with col2:
            if st.button("🗑️ Clear Results", type="secondary", key="clear_bar_results"):
                self._clear_results()
                st.rerun()
        
        with col3:
            # Información de estado
            if 'bar_chart_results' in st.session_state:
                results_count = len(st.session_state.bar_chart_results)
                selected_count = sum(1 for r in st.session_state.bar_chart_results if r.get('selected', False))
                st.info(f"📈 {results_count} charts generated, {selected_count} selected")
    
    def _generate_charts(self):
        """Genera las gráficas según los parámetros seleccionados"""
        
        main_var = st.session_state.bar_main_variable
        extra_vars = st.session_state.bar_extra_variables
        
        # Recopilar parámetros
        params = {
            'estimator': st.session_state.bar_estimator,
            'figsize': (st.session_state.bar_figsize_width, st.session_state.bar_figsize_height),
            'rotation': st.session_state.bar_rotation,
            'mean_col': st.session_state.bar_mean_col,
            'percentile_range': (st.session_state.bar_percentile_low, st.session_state.bar_percentile_high)
        }
        
        with st.spinner(f"Generating {len(extra_vars)} bar charts..."):
            results = []
            
            for extra_var in extra_vars:
                try:
                    # Aquí llamarías a tu función plot_financial_barplot
                    # Por ahora usaré un placeholder
                    fig, insights = self._create_placeholder_chart(main_var, extra_var, params)
                    
                    result = {
                        'id': str(uuid.uuid4()),
                        'chart_type': self.chart_type,
                        'main_variable': main_var,
                        'extra_variable': extra_var,
                        'figure': fig,
                        'insights': insights,
                        'parameters': params.copy(),
                        'selected': False,
                        'saved': False
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    st.error(f"❌ Error generating chart for {main_var} × {extra_var}: {str(e)}")
            
            # Guardar resultados
            st.session_state.bar_chart_results = results
            
            if results:
                st.success(f"✅ Generated {len(results)} charts successfully!")
                st.rerun()
    
    def _create_placeholder_chart(self, main_var, extra_var, params):
        """Crea una gráfica usando la función real plot_financial_barplot"""
        
        try:
            # Llamar a tu función real con los parámetros correctos
            ax, insights_json = plot_financial_barplot(
                df=st.session_state.df,
                numeric_column=main_var,
                categorical_column=extra_var,
                estimator=params['estimator'],
                figsize=params['figsize'],
                rotation=params['rotation'],
                mean_col=params['mean_col'],
                percentile_range=params['percentile_range']
            )
            
            # Obtener la figura del axes
            fig = ax.get_figure()
            
            return fig, insights_json
            
        except Exception as e:
            # En caso de error, crear un gráfico placeholder con mensaje de error
            fig, ax = plt.subplots(figsize=params['figsize'])
            ax.text(0.5, 0.5, f"Error creating chart:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title(f'Error: {main_var} by {extra_var}')
            plt.tight_layout()
            
            # Insights de error
            insights = {
                'error': True,
                'error_message': str(e),
                'chart_type': 'Bar Chart',
                'main_variable': main_var,
                'extra_variable': extra_var
            }
            
            return fig, insights
    
    def _render_generated_charts(self):
        """Renderiza las gráficas generadas con opciones de selección"""
        
        if 'bar_chart_results' not in st.session_state:
            return
        
        results = st.session_state.bar_chart_results
        if not results:
            return
        
        st.markdown("### 📈 Generated Charts")
        st.write("Select the charts you want to keep and optionally edit them:")
        
        for i, result in enumerate(results):
            self._render_chart_item(result, i)
    
    def _render_chart_item(self, result, index):
        """Renderiza un item de gráfica individual"""
        
        with st.container():
            # Header con controles
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**Chart {index+1}:** {result['main_variable']} × {result['extra_variable']}")
            
            with col2:
                # Checkbox para seleccionar
                selected = st.checkbox(
                    "Select",
                    value=result['selected'],
                    key=f"bar_select_{result['id']}"
                )
                
                if selected != result['selected']:
                    st.session_state.bar_chart_results[index]['selected'] = selected
            
            with col3:
                # Botón editar (solo si está seleccionada)
                if result['selected']:
                    if st.button("✏️ Edit", key=f"bar_edit_{result['id']}"):
                        self._edit_chart(result, index)
            
            with col4:
                # Botón guardar (solo si está seleccionada)
                if result['selected'] and not result['saved']:
                    if st.button("💾 Save", key=f"bar_save_{result['id']}", type="primary"):
                        self._save_chart(result, index)
                elif result['saved']:
                    st.success("✅ Saved")
            
            # Mostrar gráfica
            st.pyplot(result['figure'])
            
            # Mostrar insights
            with st.expander(f"📊 Insights for Chart {index+1}", expanded=False):
                st.json(result['insights'])
            
            st.divider()
    
    def _edit_chart(self, result, index):
        """Permite editar una gráfica específica"""
        
        st.info(f"🔧 Editing chart: {result['main_variable']} × {result['extra_variable']}")
        
        with st.form(f"edit_form_{result['id']}"):
            st.write("**Edit Parameters:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_estimator = st.selectbox(
                    "Estimator",
                    options=['mean', 'sum', 'median', 'count', 'std', 'min', 'max'],
                    index=['mean', 'sum', 'median', 'count', 'std', 'min', 'max'].index(result['parameters']['estimator'])
                )
                
                new_rotation = st.slider(
                    "Label Rotation", 
                    min_value=0, 
                    max_value=90, 
                    value=result['parameters']['rotation']
                )
            
            with col2:
                new_mean_col = st.checkbox(
                    "Show Overall Mean", 
                    value=result['parameters']['mean_col']
                )
                
                new_figsize = st.select_slider(
                    "Figure Size",
                    options=[(8,6), (10,6), (12,7), (14,8), (16,9)],
                    value=result['parameters']['figsize'],
                    format_func=lambda x: f"{x[0]}×{x[1]}"
                )
            
            if st.form_submit_button("🔄 Regenerate Chart", type="primary"):
                # Actualizar parámetros
                new_params = result['parameters'].copy()
                new_params.update({
                    'estimator': new_estimator,
                    'rotation': new_rotation,
                    'mean_col': new_mean_col,
                    'figsize': new_figsize
                })
                
                # Regenerar gráfica
                try:
                    fig, insights = self._create_placeholder_chart(
                        result['main_variable'], 
                        result['extra_variable'], 
                        new_params
                    )
                    
                    # Actualizar resultado
                    st.session_state.bar_chart_results[index]['figure'] = fig
                    st.session_state.bar_chart_results[index]['insights'] = insights
                    st.session_state.bar_chart_results[index]['parameters'] = new_params
                    
                    st.success("✅ Chart regenerated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error regenerating chart: {str(e)}")
    
    def _save_chart(self, result, index):
        """Guarda una gráfica en la colección final"""
        
        # Inicializar saved_charts si no existe
        if 'saved_charts' not in st.session_state:
            st.session_state.saved_charts = []
        
        # Crear copia para guardar
        saved_chart = {
            'id': result['id'],
            'chart_type': result['chart_type'],
            'main_variable': result['main_variable'],
            'extra_variable': result['extra_variable'],
            'figure': result['figure'],
            'insights': result['insights'],
            'parameters': result['parameters'].copy(),
            'timestamp': st.session_state.get('filename', 'unknown_dataset')
        }
        
        # Agregar a saved_charts
        st.session_state.saved_charts.append(saved_chart)
        
        # Marcar como guardada
        st.session_state.bar_chart_results[index]['saved'] = True
        
        st.success(f"💾 Chart saved! Total saved charts: {len(st.session_state.saved_charts)}")
        st.rerun()
    
    def _clear_results(self):
        """Limpia los resultados de bar chart"""
        if 'bar_chart_results' in st.session_state:
            del st.session_state.bar_chart_results