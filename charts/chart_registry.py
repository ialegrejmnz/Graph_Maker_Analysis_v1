# charts/chart_registry.py
from charts.distributions import plot_distributions_by_category, create_custom_kde_plot
from charts.bar_charts import BarChart
from charts.stacked_charts import plot_financial_stacked_barplot, plot_financial_stacked_barplot_100, plot_financial_mekko_chart_100
from charts.timeseries_charts import plot_timeseries
from charts.scatter_charts import create_scatter_plot
from charts.heatmap_charts import create_smooth_density_heatmap
from charts.cluster_charts import create_cluster_chart

# Registry de todos los tipos de gráficas disponibles
CHART_REGISTRY = {
    "Ridge Distribution": {
        'icon': '📊',
        'description': 'Distribution plots with KDE curves and quantile bands for each category',
        'main_var_type': 'numeric',
        'main_var_description': 'Numeric variable to analyze distribution',
        'extra_vars_type': 'categorical', 
        'extra_vars_description': 'Categorical variables to group distributions by',
        'additional_params': {
            'figsize': 'Figure size (width, height)',
            'bandwidth': 'KDE bandwidth adjustment',
            'remove_outliers': 'Whether to remove outliers',
            'percentile_range': 'Percentile range for outlier filtering'
        },
        'component': plot_distributions_by_category()
    },
    
    "KDE Distribution": {
        'icon': '📈',
        'description': 'Kernel Density Estimation plots with customizable smoothing',
        'main_var_type': 'numeric',
        'main_var_description': 'Numeric variable for KDE analysis',
        'extra_vars_type': 'categorical',
        'extra_vars_description': 'Categorical variables for grouping',
        'additional_params': {
            'figsize': 'Figure size',
            'alpha': 'Transparency level',
            'common_norm': 'Whether to normalize across groups'
        },
        'component': create_custom_kde_plot()
    },
    
    "Bar Chart": {
        'icon': '📊',
        'description': 'Financial bar charts with statistical estimators',
        'main_var_type': 'numeric',
        'main_var_description': 'Numeric variable to aggregate',
        'extra_vars_type': 'categorical',
        'extra_vars_description': 'Categorical variables for grouping',
        'additional_params': {
            'estimator': 'Statistical estimator (mean, sum, median, etc.)',
            'rotation': 'X-axis label rotation angle',
            'mean_col': 'Show overall mean column'
        },
        'component': BarChart()
    },
    
    "Stacked Charts": {
        'icon': '📚',
        'description': 'Stacked bar charts (regular, 100%, Mekko) with oval metrics',
        'main_var_type': 'numeric', 
        'main_var_description': 'Numeric variable to stack',
        'extra_vars_type': 'categorical',
        'extra_vars_description': 'Categorical variables for main grouping',
        'additional_params': {
            'stack_column': 'Column for stacking within groups',
            'chart_type': 'Type of stacked chart',
            'horizontal_params': 'Parameters for horizontal ovals',
            'vertical_params': 'Parameters for vertical ovals'
        },
        'component': plot_financial_stacked_barplot()
    },
    
    "Time Series": {
        'icon': '📈',
        'description': 'Time series analysis with trend detection and CAGR calculation',
        'main_var_type': 'metric',
        'main_var_description': 'Time series metric to analyze',
        'extra_vars_type': 'categorical',
        'extra_vars_description': 'Categorical variables for grouping time series',
        'additional_params': {
            'subplot_layout': 'Custom subplot arrangement',
            'show_points': 'Show individual data points',
            'percentile_range': 'Outlier filtering range'
        },
        'component': plot_timeseries()
    },
    
    "Scatter Plot": {
        'icon': '🎯',
        'description': 'Scatter plots with regression analysis and bubble sizing',
        'main_var_type': 'numeric',
        'main_var_description': 'Numeric variable for X-axis',
        'extra_vars_type': 'numeric', 
        'extra_vars_description': 'Numeric variables for Y-axis',
        'additional_params': {
            'categorical_col': 'Categorical column for coloring',
            'size_col': 'Column for bubble sizing',
            'regression_type': 'Type of regression analysis',
            'polynomial_degree': 'Degree for polynomial regression'
        },
        'component': create_scatter_plot()
    },
    
    "Heatmap": {
        'icon': '🔥',
        'description': 'Smooth density heatmaps with Gaussian field interpolation',
        'main_var_type': 'numeric',
        'main_var_description': 'Numeric variable for X-axis positioning',
        'extra_vars_type': 'numeric',
        'extra_vars_description': 'Numeric variables for Y-axis positioning', 
        'additional_params': {
            'color_col': 'Column for heatmap coloring',
            'grid_resolution': 'Heatmap resolution',
            'bandwidth': 'Gaussian kernel bandwidth',
            'kernel': 'Type of smoothing kernel'
        },
        'component': create_smooth_density_heatmap()
    },
    
    "Cluster Chart": {
        'icon': '🎪',
        'description': 'Cluster visualization with characteristic zones and contours',
        'main_var_type': 'numeric',
        'main_var_description': 'Numeric variable for X-axis',
        'extra_vars_type': 'numeric',
        'extra_vars_description': 'Numeric variables for Y-axis',
        'additional_params': {
            'categorical_col': 'Categorical column for cluster coloring',
            'delimitation': 'Show cluster boundary lines',
            'smoothing_factor': 'Cluster boundary smoothing'
        },
        'component': create_cluster_chart()
    }
}