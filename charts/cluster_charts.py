import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from typing import Optional, Dict, Tuple, Union, List

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')

# Try to import scipy for smooth interpolation
try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from insights_functions import (
    generate_strategic_insights
)

from common_functions import (
    get_scale_and_format_eur
)

def analyze_categories(data, categorical_col, x_axis_col, y_axis_col, insight_params=None):
    """
    Analyzes data by categories, calculating means for all involved variables.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with the data
    categorical_col : str
        Name of the categorical column
    x_axis_col : str
        Name of the X-axis column
    y_axis_col : str
        Name of the Y-axis column
    insight_params : list, optional
        List of additional numeric columns to analyze (max 3)

    Returns:
    --------
    dict: Analysis results for each category including total dataset analysis
    """
    analysis = {}

    # Define all variables to analyze
    analysis_vars = [x_axis_col, y_axis_col]
    if insight_params:
        analysis_vars.extend(insight_params)

    # Calculate total dataset statistics
    total_count = len(data)
    total_analysis = {
        "company_count": total_count,
        "company_percentage": 100.0,
        "category": "entire_dataset"
    }

    # Add means for all variables
    for var in analysis_vars:
        total_analysis[f"{var} Mean"] = data[var].mean()

    analysis["Total"] = total_analysis

    # Analyze each category
    categories = data[categorical_col].unique()

    for category in categories:
        # Filter data for this category
        category_mask = data[categorical_col] == category
        category_data = data[category_mask]
        category_count = len(category_data)
        category_percentage = (category_count / total_count * 100) if total_count > 0 else 0

        category_analysis = {
            "company_count": category_count,
            "company_percentage": category_percentage,
            "category": str(category)
        }

        # Calculate means for all variables
        for var in analysis_vars:
            if category_count == 0:
                category_analysis[f"{var} Mean"] = None
            else:
                category_analysis[f"{var} Mean"] = category_data[var].mean()

        analysis[str(category)] = category_analysis

    return analysis

def generate_category_insights(categories_analysis, categorical_col, x_axis_col, y_axis_col, insight_params=None):
    """
    Generates comparative insights between categories and between each category and total dataset.

    Parameters:
    -----------
    categories_analysis : dict
        Analysis results for each category including "Total"
    categorical_col : str
        Name of the categorical column
    x_axis_col : str
        Name of the X-axis column
    y_axis_col : str
        Name of the Y-axis column
    insight_params : list, optional
        List of additional columns that were analyzed

    Returns:
    --------
    list: Generated comparative insights
    """
    insights = []

    # Define all variables to analyze
    analysis_vars = [x_axis_col, y_axis_col]
    if insight_params:
        analysis_vars.extend(insight_params)

    # Get total data for comparison
    total_data = categories_analysis.get("Total", {})

    # Get category data (excluding Total)
    category_data = {name: data for name, data in categories_analysis.items()
                     if name != "Total" and data["company_count"] > 0}

    if len(category_data) == 0:
        insights.append("No categories contain data for analysis")
        return insights

    category_names = list(category_data.keys())

    # Helper function to get safe ratio
    def get_ratio(val1, val2):
        if abs(val2) < 0.001:
            return None
        return abs(val1 / val2)

    # 1. COMPARISONS BETWEEN CATEGORIES (if more than 1 category)
    if len(category_names) >= 2:
        for var in analysis_vars:
            var_values = {}

            # Collect valid values for this variable
            for name, data in category_data.items():
                mean_value = data.get(f"{var} Mean")
                if mean_value is not None and not pd.isna(mean_value):
                    var_values[name] = mean_value

            if len(var_values) >= 2:
                # Find highest and lowest for this variable
                highest_category = max(var_values.keys(), key=lambda k: var_values[k])
                lowest_category = min(var_values.keys(), key=lambda k: var_values[k])

                if highest_category != lowest_category:
                    ratio = get_ratio(var_values[highest_category], var_values[lowest_category])
                    if ratio and ratio > 1.2:
                        insights.append(f"The mean of {var} in {highest_category} category is {ratio:.1f}x higher than the mean of {var} in {lowest_category} category")

    # 2. COMPARISONS BETWEEN EACH CATEGORY AND TOTAL
    if total_data:
        for var in analysis_vars:
            total_mean = total_data.get(f"{var} Mean")
            if total_mean is not None and not pd.isna(total_mean):

                for category_name, category_info in category_data.items():
                    category_mean = category_info.get(f"{var} Mean")
                    if category_mean is not None and not pd.isna(category_mean):

                        ratio = get_ratio(category_mean, total_mean)
                        if ratio and ratio > 1.2:
                            insights.append(f"The mean of {var} in {category_name} category is {ratio:.1f}x higher than the mean of {var} in all the data")
                        elif ratio and ratio < 0.8:
                            inverse_ratio = get_ratio(total_mean, category_mean)
                            if inverse_ratio:
                                insights.append(f"The mean of {var} in all the data is {inverse_ratio:.1f}x higher than the mean of {var} in {category_name} category")

    # 3. COMPANY DISTRIBUTION INSIGHTS
    if len(category_names) >= 2:
        company_counts = {name: data["company_count"] for name, data in category_data.items()}
        most_populated = max(company_counts.keys(), key=lambda k: company_counts[k])
        least_populated = min(company_counts.keys(), key=lambda k: company_counts[k])

        if most_populated != least_populated:
            ratio = get_ratio(company_counts[most_populated], company_counts[least_populated])
            if ratio and ratio > 1.2:
                insights.append(f"The number of companies in {most_populated} category is {ratio:.1f}x higher than the number of companies in {least_populated} category ({company_counts[most_populated]} vs {company_counts[least_populated]})")

    # Limit insights to avoid overwhelm
    if len(insights) > 15:
        insights = insights[:15]

    return insights

def create_cluster_chart_with_insights(
    df: pd.DataFrame,
    x_axis_col: str,
    y_axis_col: str,
    categorical_col: str,
    insight_params: Optional[List[str]] = None,
    percentiles: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (12, 8),
    delimitation: bool = True,
    show_scatter_points: bool = True,
    smoothing_factor: float = 1.0,
    openai=False
):
    """
    Creates a cluster chart with insights analysis showing characteristic zones for each category
    with colored areas and optional contour lines, plus comprehensive statistical analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot
    x_axis_col : str
        Name of the numeric column for the X-axis
    y_axis_col : str
        Name of the numeric column for the Y-axis
    categorical_col : str
        Categorical column to differentiate data families by zones
    insight_params : list of str, optional
        List of additional numeric columns to analyze (maximum 3)
        Example: ['EBITDA EUR 2024', 'Employees', 'Market_Cap']
    percentiles : tuple, optional
        Tuple with percentiles (lower, upper) to filter outliers
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (12, 8)
    delimitation : bool, optional
        Whether to show contour lines around characteristic zones. Default: True
    show_scatter_points : bool, optional
        Whether to show individual data points. Default: True
    smoothing_factor : float, optional
        Controls the smoothing level of the density estimation. Default: 1.0
        - Values < 1.0: More smoothing, less overfitting, broader zones
        - Values > 1.0: Less smoothing, more overfitting, tighter zones
        - Recommended range: 0.3 to 3.0

    Returns:
    --------
    If insight analysis is performed:
        tuple: (matplotlib.figure.Figure, dict) - Figure and analysis JSON
    Otherwise:
        matplotlib.figure.Figure: Matplotlib figure object

    Examples:
    ---------
    # Basic cluster chart with insights
    fig, analysis = create_cluster_chart_with_insights(
        df, 'Revenue EUR 2024', 'EBITDA EUR 2024', 'Industry'
    )

    # With additional variables for analysis
    fig, analysis = create_cluster_chart_with_insights(
        df, 'Revenue EUR 2024', 'EBITDA EUR 2024', 'Industry',
        insight_params=['Employees', 'Market_Cap EUR']
    )

    # Access insights
    print("Generated Insights:")
    for insight in analysis['insights']:
        print(f"- {insight}")
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from typing import List, Optional, Tuple

    # Validation for insight_params
    if insight_params is not None:
        if not isinstance(insight_params, list):
            raise ValueError("insight_params must be a list of column names")

        if len(insight_params) > 3:
            raise ValueError("insight_params can contain maximum 3 additional variables")

        # Validate that insight_params columns exist and are numeric
        for param in insight_params:
            if param not in df.columns:
                raise ValueError(f"Column '{param}' specified in insight_params does not exist in DataFrame")
            if not pd.api.types.is_numeric_dtype(df[param]):
                raise ValueError(f"Column '{param}' specified in insight_params must be numeric")

    # Validation (keeping original validation logic)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise ValueError("figsize must be a tuple with 2 elements (width, height)")

    if not isinstance(smoothing_factor, (int, float)) or smoothing_factor <= 0:
        raise ValueError("smoothing_factor must be a positive number")

    if smoothing_factor < 0.1 or smoothing_factor > 5.0:
        print(f"Warning: smoothing_factor={smoothing_factor} is outside recommended range (0.1-5.0)")

    # Create copy of dataframe to avoid modifying the original
    data = df.copy()

    # Validate that required columns exist
    required_cols = [x_axis_col, y_axis_col, categorical_col]
    if insight_params:
        required_cols.extend(insight_params)

    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame")

    # Validate that x and y columns are numeric
    if not pd.api.types.is_numeric_dtype(data[x_axis_col]):
        raise ValueError(f"X-axis column '{x_axis_col}' must be numeric")

    if not pd.api.types.is_numeric_dtype(data[y_axis_col]):
        raise ValueError(f"Y-axis column '{y_axis_col}' must be numeric")

    # Apply outlier filtering if specified
    if percentiles:
        lower_percentile, upper_percentile = percentiles

        # Handle percentage format (0.05 vs 5)
        if lower_percentile <= 1.0:
            lower_percentile *= 100
        if upper_percentile <= 1.0:
            upper_percentile *= 100

        # Filter X-axis outliers
        x_lower = np.percentile(data[x_axis_col].dropna(), lower_percentile)
        x_upper = np.percentile(data[x_axis_col].dropna(), upper_percentile)
        data = data[(data[x_axis_col] >= x_lower) & (data[x_axis_col] <= x_upper)]

        # Filter Y-axis outliers
        y_lower = np.percentile(data[y_axis_col].dropna(), lower_percentile)
        y_upper = np.percentile(data[y_axis_col].dropna(), upper_percentile)
        data = data[(data[y_axis_col] >= y_lower) & (data[y_axis_col] <= y_upper)]

    # Remove rows with NaN values in essential columns
    data = data.dropna(subset=required_cols)

    if len(data) == 0:
        raise ValueError("No data remaining after filtering and removing NaN values")

    # Get unique categories and validate we have enough data
    categories = data[categorical_col].unique()
    n_categories = len(categories)

    if n_categories == 0:
        raise ValueError("No categories found in categorical column")

    # Check minimum points per category for meaningful clustering
    min_points_per_category = 3
    valid_categories = []
    for category in categories:
        category_data = data[data[categorical_col] == category]
        if len(category_data) >= min_points_per_category:
            valid_categories.append(category)
        else:
            print(f"Warning: Category '{category}' has only {len(category_data)} points. "
                  f"Minimum {min_points_per_category} required for meaningful clustering.")

    if len(valid_categories) == 0:
        raise ValueError(f"No categories have sufficient data points (minimum {min_points_per_category} per category)")

    categories = valid_categories

    # Create figure with transparent background and consistent styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Apply consistent styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Prepare variables and scaling using existing utility functions
    x = data[x_axis_col]
    y = data[y_axis_col]

    x_scale, x_suffix = get_scale_and_format_eur(x, x_axis_col)
    y_scale, y_suffix = get_scale_and_format_eur(y, y_axis_col)

    x_scaled = x / x_scale
    y_scaled = y / y_scale

    # Set up grid for density estimation
    x_min, x_max = x_scaled.min(), x_scaled.max()
    y_min, y_max = y_scaled.min(), y_scaled.max()

    # Add padding to ensure all data is captured
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_x = x_range * 0.1
    padding_y = y_range * 0.1

    x_min -= padding_x
    x_max += padding_x
    y_min -= padding_y
    y_max += padding_y

    # Create grid for contour plots
    grid_resolution = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))

    # Generate colors for categories using consistent color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))

    # Process each category
    for i, category in enumerate(categories):
        # Get data for this category
        mask = data[categorical_col] == category
        category_data = data[mask]

        x_cat = x_scaled[mask]
        y_cat = y_scaled[mask]

        if len(x_cat) < min_points_per_category:
            continue

        # Create density estimation for this category using KDE
        try:
            from scipy.stats import gaussian_kde

            # Prepare data for KDE
            positions = np.vstack([x_cat, y_cat])

            # Create KDE with smoothing control
            if smoothing_factor == 1.0:
                kernel = gaussian_kde(positions)
            else:
                custom_bw = lambda kde_obj: kde_obj.scotts_factor() / smoothing_factor
                kernel = gaussian_kde(positions, bw_method=custom_bw)

            # Evaluate on grid
            grid_positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kernel(grid_positions)
            density = density.reshape(xx.shape)

            # Normalize density for consistent visualization across categories
            density = density / density.max()

            # Create filled contours for characteristic zones
            base_color = colors[i]
            light_alpha = 0.3

            # Define threshold level based on smoothing factor
            if smoothing_factor <= 0.5:
                threshold_level = 0.05
            elif smoothing_factor <= 1.0:
                threshold_level = 0.1
            elif smoothing_factor <= 1.5:
                threshold_level = 0.15
            else:
                threshold_level = 0.25

            # Create filled contours
            contour_filled = ax.contourf(xx, yy, density,
                                       levels=[threshold_level, density.max()],
                                       colors=[base_color],
                                       alpha=light_alpha)

            # Add contour lines if delimitation is enabled
            if delimitation:
                contour_lines = ax.contour(xx, yy, density,
                                         levels=[threshold_level],
                                         colors=[base_color],
                                         alpha=0.8,
                                         linewidths=1.5)

        except ImportError:
            # Fallback: if scipy is not available, use simpler approach
            print("Warning: scipy not available. Using simplified zone representation with smoothing adjustment.")
            from matplotlib.patches import Ellipse

            x_mean, y_mean = x_cat.mean(), y_cat.mean()
            x_std, y_std = x_cat.std(), y_cat.std()

            base_multiplier = 4
            if smoothing_factor <= 0.5:
                size_multiplier = base_multiplier * 1.8
            elif smoothing_factor <= 1.0:
                size_multiplier = base_multiplier * (2.0 - smoothing_factor)
            else:
                size_multiplier = base_multiplier / (smoothing_factor * 0.8)

            ellipse = Ellipse((x_mean, y_mean),
                            size_multiplier * x_std, size_multiplier * y_std,
                            alpha=0.3,
                            facecolor=base_color,
                            edgecolor=base_color if delimitation else 'none',
                            linewidth=1.5)
            ax.add_patch(ellipse)

        except Exception as e:
            print(f"Warning: Could not create density estimation for category '{category}': {e}")
            continue

        # Add scatter points if requested
        if show_scatter_points:
            ax.scatter(x_cat, y_cat,
                      c=[base_color],
                      s=50,
                      alpha=0.8,
                      edgecolors='white',
                      linewidth=0.5,
                      label=f'{categorical_col}: {category}',
                      zorder=10)

    # Create legend
    if show_scatter_points:
        ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        legend_elements = []
        for i, category in enumerate(categories):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1,
                                               facecolor=colors[i],
                                               alpha=0.5,
                                               label=f'{categorical_col}: {category}'))

        ax.legend(handles=legend_elements,
                 frameon=False,
                 loc='center left',
                 bbox_to_anchor=(1.02, 0.5))

    # Set axis labels with proper scaling and suffix notation
    def create_axis_label(column_name, suffix):
        """Create axis label with proper suffix notation"""
        if suffix:
            column_upper = column_name.upper()
            # Check if it's a percentage metric
            is_percentage = ('GROWTH' in column_upper or 'CAGR' in column_upper or 'MARGIN' in column_upper)

            if 'SIZE' not in column_upper:
                return f'{column_name} {suffix} €'
            elif is_percentage:
                return f'{column_name} %'
            else:
                return f'{column_name} {suffix}'
        else:
            return column_name

    x_label = create_axis_label(x_axis_col, x_suffix)
    y_label = create_axis_label(y_axis_col, y_suffix)

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Generate insights analysis
    categories_analysis = analyze_categories(data, categorical_col, x_axis_col, y_axis_col, insight_params)
    insights = generate_category_insights(categories_analysis, categorical_col, x_axis_col, y_axis_col, insight_params)

    analysis_json = {
        "categories_analysis": categories_analysis,
        "insights": insights
    }

    if openai is True:
          analysis_json = generate_strategic_insights(
              y_axis_col, categorical_col,x_axis_col,analysis_json,
              API_KEY= API_KEY
          )

    print(f"Cluster chart created with {len(data)} data points across {len(categories)} categories")
    if insight_params:
        print(f"Additional analysis variables: {', '.join(insight_params)}")
    print(f"Generated {len(insights)} insights")

    return fig, analysis_json