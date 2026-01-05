import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
from typing import Optional, Dict, Tuple, Union

# Try to import scipy for smooth interpolation
try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from insights_functions import (
    AVAILABLE_ESTIMATORS,
    generate_multilevel_aggregations,
    add_pareto_insights,
    generate_comparison_insights
)

from common_functions import (
    get_scale_and_format_eur
)

def create_scatter_plot(
    df: pd.DataFrame,
    x_axis_col: str,
    y_axis_col: str,
    percentiles: Optional[Tuple[float, float]] = None,
    categorical_col: Optional[str] = None,
    size_col: Optional[str] = None,
    regression_type: str = 'none',  # 'none', 'linear', 'polynomial'
    polynomial_degree: int = 2,     # Grado del polinomio (solo para 'polynomial')
    top_n: Optional[Dict[str, int]] = None,
    figsize: Tuple[int, int] = (12, 8),
    names: Optional[Dict[str, int]] = None,
    show_r2: bool = True  # Mostrar R² en la leyenda
):
    """
    Creates a scatter plot or bubble chart with linear and non-linear regression options.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot
    x_axis_col : str
        Name of the numeric column for the X-axis
    y_axis_col : str
        Name of the numeric column for the Y-axis
    percentiles : tuple, optional
        Tuple with percentiles (lower, upper) to filter outliers
    categorical_col : str, optional
        Categorical column to differentiate data families by color
    size_col : str, optional
        Numeric column to define point sizes (bubble chart)
    regression_type : str, optional
        Type of regression: 'none', 'linear', 'polynomial'. Default: 'none'
    polynomial_degree : int, optional
        Degree of polynomial for polynomial regression (2-10). Default: 2
        - 2: Quadratic (parabola)
        - 3: Cubic
        - 4: Quartic
        - Higher degrees: More complex curves (be careful with overfitting)
    top_n : dict, optional
        Dictionary to filter top N points
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (12, 8)
    names : dict, optional
        Dictionary to show company names for top/bottom N companies
    show_r2 : bool, optional
        Whether to show R² score in legend for regression lines. Default: True

    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object

    Examples:
    ---------
    # Regresión cuadrática básica
    fig = create_nonlinear_scatter_plot(df, 'Revenue EUR 2024', 'EBITDA EUR 2024',
                                       regression_type='polynomial', polynomial_degree=2)

    # Regresión cúbica por categorías
    fig = create_nonlinear_scatter_plot(df, 'Revenue EUR 2024', 'EBITDA EUR 2024',
                                       categorical_col='Industry',
                                       regression_type='polynomial', polynomial_degree=3)

    # Comparar lineal vs polinomial (ejecutar dos veces con diferentes parámetros)
    fig1 = create_nonlinear_scatter_plot(df, 'x', 'y', regression_type='linear')
    fig2 = create_nonlinear_scatter_plot(df, 'x', 'y', regression_type='polynomial', polynomial_degree=3)
    """

    # Validation
    if regression_type not in ['none', 'linear', 'polynomial']:
        raise ValueError("regression_type must be 'none', 'linear', or 'polynomial'")

    if polynomial_degree < 1 or polynomial_degree > 10:
        raise ValueError("polynomial_degree must be between 1 and 10")

    if polynomial_degree == 1 and regression_type == 'polynomial':
        print("Warning: polynomial_degree=1 is equivalent to linear regression")

    # Create copy of dataframe to avoid modifying the original
    data = df.copy()

    # Validate that columns exist
    required_cols = [x_axis_col, y_axis_col]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame")

    # Validate optional columns
    if categorical_col and categorical_col not in data.columns:
        raise ValueError(f"Categorical column '{categorical_col}' does not exist in DataFrame")

    if size_col and size_col not in data.columns:
        raise ValueError(f"Size column '{size_col}' does not exist in DataFrame")

    # Validate company names column if provided
    if names:
        if not isinstance(names, dict):
            raise ValueError("names must be a dictionary with format {'column_name': N}")

        if 'Company name Latin alphabet' not in data.columns:
            raise ValueError("Column 'Company name Latin alphabet' not found. Required when names is provided")

        for col in names.keys():
            if col not in data.columns:
                raise ValueError(f"Column '{col}' in names parameter does not exist in DataFrame")

    # Apply filtering
    if percentiles:
        lower_percentile, upper_percentile = percentiles
        if lower_percentile <= 1.0:
            lower_percentile *= 100
        if upper_percentile <= 1.0:
            upper_percentile *= 100

        x_lower = np.percentile(data[x_axis_col].dropna(), lower_percentile)
        x_upper = np.percentile(data[x_axis_col].dropna(), upper_percentile)
        data = data[(data[x_axis_col] >= x_lower) & (data[x_axis_col] <= x_upper)]

        y_lower = np.percentile(data[y_axis_col].dropna(), lower_percentile)
        y_upper = np.percentile(data[y_axis_col].dropna(), upper_percentile)
        data = data[(data[y_axis_col] >= y_lower) & (data[y_axis_col] <= y_upper)]

    if top_n:
        for column, n_points in top_n.items():
            if column not in data.columns:
                raise ValueError(f"Column '{column}' for top_n does not exist in DataFrame")
            if n_points > 0:
                data = data.nlargest(n_points, column)
            elif n_points < 0:
                data = data.nsmallest(abs(n_points), column)

    # Remove rows with NaN values in essential columns
    essential_cols = [x_axis_col, y_axis_col]
    if categorical_col:
        essential_cols.append(categorical_col)
    if size_col:
        essential_cols.append(size_col)
    if names:
        essential_cols.append('Company name Latin alphabet')

    data = data.dropna(subset=essential_cols)

    if len(data) == 0:
        raise ValueError("No data remaining after filtering and removing NaN values")

    # Determine which companies should have names shown
    companies_to_name = set()
    if names:
        for column, n_companies in names.items():
            if n_companies > 0:
                top_companies = data.nlargest(n_companies, column)['Company name Latin alphabet'].tolist()
                companies_to_name.update(top_companies)
            elif n_companies < 0:
                bottom_companies = data.nsmallest(abs(n_companies), column)['Company name Latin alphabet'].tolist()
                companies_to_name.update(bottom_companies)

    # Create figure and styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Prepare variables and scaling
    x = data[x_axis_col]
    y = data[y_axis_col]

    x_scale, x_suffix = get_scale_and_format_eur(x, x_axis_col)
    y_scale, y_suffix = get_scale_and_format_eur(y, y_axis_col)

    x_scaled = x / x_scale
    y_scaled = y / y_scale

    # Color configuration
    default_color = '#7D4BEB'

    # Scatter plot creation
    if categorical_col:
        categories = data[categorical_col].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories, colors))

        for i, category in enumerate(categories):
            mask = data[categorical_col] == category
            x_cat = x_scaled[mask]
            y_cat = y_scaled[mask]

            if size_col:
                sizes = data[size_col][mask]
                size_scale, size_suffix = get_scale_and_format_eur(sizes, size_col)
                sizes_scaled = sizes / size_scale
                if len(sizes_scaled) > 1 and sizes_scaled.max() != sizes_scaled.min():
                    sizes_norm = ((sizes_scaled - sizes_scaled.min()) / (sizes_scaled.max() - sizes_scaled.min()) * 180 + 20)
                else:
                    sizes_norm = 50
            else:
                sizes_norm = 50

            ax.scatter(x_cat, y_cat, c=[colors[i]], s=sizes_norm, alpha=0.7,
                      label=f'{categorical_col}: {category}', edgecolors='white', linewidth=0.5)

            # Add company names for selected companies
            if names and companies_to_name:
                mask_data = data[mask]
                for x_pos, y_pos, company_name in zip(x_cat, y_cat, mask_data['Company name Latin alphabet']):
                    if company_name in companies_to_name:
                        ax.annotate(str(company_name),
                                   (x_pos, y_pos),
                                   xytext=(3, 3),
                                   textcoords='offset points',
                                   fontsize=8,
                                   alpha=0.8,
                                   color='darkgray',
                                   ha='left',
                                   va='bottom')
    else:
        if size_col:
            sizes = data[size_col]
            size_scale, size_suffix = get_scale_and_format_eur(sizes, size_col)
            sizes_scaled = sizes / size_scale
            if len(sizes_scaled) > 1 and sizes_scaled.max() != sizes_scaled.min():
                sizes_norm = ((sizes_scaled - sizes_scaled.min()) / (sizes_scaled.max() - sizes_scaled.min()) * 180 + 20)
            else:
                sizes_norm = 50
        else:
            sizes_norm = 50

        ax.scatter(x_scaled, y_scaled, c=default_color, s=sizes_norm, alpha=0.7,
                  edgecolors='white', linewidth=0.5)

        # Add company names for selected companies
        if names and companies_to_name:
            for x_pos, y_pos, company_name in zip(x_scaled, y_scaled, data['Company name Latin alphabet']):
                if company_name in companies_to_name:
                    ax.annotate(str(company_name),
                               (x_pos, y_pos),
                               xytext=(3, 3),
                               textcoords='offset points',
                               fontsize=8,
                               alpha=0.8,
                               color='darkgray',
                               ha='left',
                               va='bottom')

    # REGRESSION FUNCTIONALITY
    def fit_and_plot_regression(x_data, y_data, color, label_prefix, category_name=None):
        """Helper function to fit and plot regression"""
        if len(x_data) <= polynomial_degree + 1:
            print(f"Warning: Not enough points for {label_prefix} {category_name if category_name else ''} "
                  f"(need at least {polynomial_degree + 2} points for degree {polynomial_degree})")
            return

        try:
            x_reg = x_data.values.reshape(-1, 1)
            y_reg = y_data.values

            if regression_type == 'linear':
                # Linear regression
                model = LinearRegression()
                model.fit(x_reg, y_reg)

                x_line = np.linspace(x_reg.min(), x_reg.max(), 100).reshape(-1, 1)
                y_line = model.predict(x_line)

                # Calculate R²
                y_pred = model.predict(x_reg)
                r2 = r2_score(y_reg, y_pred)

                line_style = '-'

            elif regression_type == 'polynomial':
                # Polynomial regression
                poly_model = Pipeline([
                    ('poly', PolynomialFeatures(degree=polynomial_degree)),
                    ('linear', LinearRegression())
                ])

                poly_model.fit(x_reg, y_reg)

                x_line = np.linspace(x_reg.min(), x_reg.max(), 200).reshape(-1, 1)  # More points for smooth curves
                y_line = poly_model.predict(x_line)

                # Calculate R²
                y_pred = poly_model.predict(x_reg)
                r2 = r2_score(y_reg, y_pred)

                # Different line styles for different degrees
                line_styles = {2: '-', 3: '-', 4: '-', 5: '-'}
                line_style = line_styles.get(polynomial_degree, '-')

            # Create label with R²
            if show_r2:
                if category_name:
                    regression_label = f'{label_prefix} {category_name} (R²={r2:.3f})'
                else:
                    regression_label = f'{label_prefix} (R²={r2:.3f})'
            else:
                if category_name:
                    regression_label = f'{label_prefix} {category_name}'
                else:
                    regression_label = label_prefix

            # Plot regression line
            ax.plot(x_line.flatten(), y_line, color=color, linestyle=line_style,
                   alpha=0.8, linewidth=2.5, label=regression_label)

        except Exception as e:
            print(f"Error fitting regression for {category_name if category_name else 'data'}: {e}")

    # Apply regression based on type
    if regression_type != 'none':
        if regression_type == 'linear':
            label_prefix = 'Linear Regression'
        elif regression_type == 'polynomial':
            degree_names = {2: 'Quadratic', 3: 'Cubic', 4: 'Quartic', 5: 'Quintic'}
            degree_name = degree_names.get(polynomial_degree, f'Degree-{polynomial_degree}')
            label_prefix = f'{degree_name} Regression'

        if categorical_col:
            # Regression by category
            for category in categories:
                mask = data[categorical_col] == category
                category_data = data[mask]

                if len(category_data) > polynomial_degree + 1:  # Need enough points
                    x_cat_reg = x_scaled[mask]
                    y_cat_reg = y_scaled[mask]

                    fit_and_plot_regression(x_cat_reg, y_cat_reg, color_map[category],
                                          label_prefix, category)
        else:
            # General regression
            if len(data) > polynomial_degree + 1:
                fit_and_plot_regression(x_scaled, y_scaled, 'red', label_prefix)

    # Labels and formatting
    x_label = f'{x_axis_col}'
    if x_suffix:
        x_label += f' ({x_suffix})'

    y_label = f'{y_axis_col}'
    if y_suffix:
        y_label += f' ({y_suffix})'

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Create custom legend
    legend_handles = []
    legend_labels = []

    # Add scatter plot entries
    if categorical_col:
        for i, category in enumerate(categories):
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=colors[i], markersize=8, alpha=0.7))
            legend_labels.append(f'{categorical_col}: {category}')
    else:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=default_color, markersize=8, alpha=0.7))
        legend_labels.append('Data Points')

    # Add regression entries (automatically added by ax.plot with label parameter)

    # Add size information if applicable
    if size_col:
        size_label = f'Bubble size ∝ {size_col}'
        if size_suffix:
            size_label += f' ({size_suffix})'

        legend_handles.extend([plt.Line2D([0], [0], color='none'),
                              plt.Line2D([0], [0], color='none')])
        legend_labels.extend(['', size_label])

    # Position legend
    if legend_handles or regression_type != 'none':
        ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    # Title
    title_parts = []
    if size_col:
        title_parts.append("Bubble Chart")
    else:
        title_parts.append("Scatter Plot")

    if regression_type != 'none':
        if regression_type == 'polynomial':
            title_parts.append(f"with Degree-{polynomial_degree} Polynomial Regression")
        else:
            title_parts.append("with Linear Regression")

    if categorical_col:
        title_parts.append(f"by {categorical_col}")

    ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig