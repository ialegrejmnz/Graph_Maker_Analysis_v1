import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import uuid

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
    AVAILABLE_ESTIMATORS,
    generate_comparison_insights,
    generate_strategic_insights
)

from common_functions import (
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
                          percentile_range=(0, 1),openai=False):
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
    if openai is True:
      result_full_json = generate_strategic_insights(
          categorical_column, None,numeric_column,result_full_json,
          API_KEY=API_KEY
      )

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