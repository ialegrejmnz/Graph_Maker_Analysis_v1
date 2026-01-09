import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.patches import Rectangle, Ellipse
from typing import Optional, Dict, Tuple, Union
from dotenv import load_dotenv
import os

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
    generate_multilevel_aggregations,
    add_pareto_insights,
    generate_strategic_insights
)

from common_functions import (
    get_scale_and_format_eur,
    generate_intelligent_ylabel,
    setup_chart_style,
    create_custom_color_palette
)

def _get_custom_scale_and_suffix_standalone(values):
    """
    Standalone version of scale and suffix calculation for consistent oval formatting.

    Parameters:
    -----------
    values : array-like
        Array of numeric values

    Returns:
    --------
    tuple
        (scale_factor, suffix_string)

    Examples:
    ---------
    values >= 1,000,000,000 → (1e9, 'Bn')
    values >= 1,000,000 → (1e6, 'Mn')
    values >= 1,000 → (1e3, 'K')
    values < 1,000 → (1, '')
    """
    if len(values) == 0:
        return 1, ''

    max_val = max(abs(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)))

    if max_val >= 1_000_000_000:  # Billions
        return 1e9, 'Bn'
    elif max_val >= 1_000_000:    # Millions
        return 1e6, 'Mn'
    elif max_val >= 1_000:        # Thousands
        return 1e3, 'K'
    else:                         # Units
        return 1, ''

def _format_oval_value_standalone(value, scale, suffix, estimator=None):
    """
    Standalone version of oval value formatting.

    Parameters:
    -----------
    value : float
        Value to format
    scale : float
        Scale factor (1, 1e3, 1e6, 1e9)
    suffix : str
        Suffix string ('', 'K', 'Mn', 'Bn')
    estimator : str, optional
        Statistical estimator ('count', 'sum', 'mean', etc.)

    Returns:
    --------
    str
        Formatted value string

    Examples:
    ---------
    9340000, 1e6, 'Mn' → '9.3Mn'
    4650, 1e3, 'K' → '4.7K'
    123, 1, '', 'count' → '123'
    45.67, 1, '' → '45.7'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '0'

    # Special formatting for count estimator: integer without decimals or units
    if estimator == 'count':
        return f"{int(value)}"

    scaled_value = value / scale

    if suffix == '':  # No suffix, format based on decimal places
        if scaled_value == int(scaled_value):
            return f"{int(scaled_value)}"
        else:
            return f"{scaled_value:.1f}"  # Changed to 1 decimal place
    else:  # Has suffix, always use 1 decimal place (changed from 2)
            return f"{scaled_value:.1f}{suffix}"

def _create_oval_title_standalone(estimator, variable_name):
    """
    Standalone version of oval title creation.

    Parameters:
    -----------
    estimator : str
        Statistical estimator (e.g., 'sum', 'mean', 'median')
    variable_name : str
        Variable name (e.g., 'Number of employees', 'Revenue EUR')

    Returns:
    --------
    str
        Formatted title (e.g., 'Sum Number of employees', 'Mean Revenue EUR')
    """
    # 🔥 MODIFICACIÓN: Special handling for 'count' estimator
    if estimator.lower() == 'count':
        return "Number of companies"

    return f"{estimator.title()} {variable_name}"

def plot_financial_stacked_barplot(df, numeric_column, categorical_column, stack_column, estimator='mean',
                                 figsize=(12, 7), rotation=45, horizontal_params=None, vertical_params=None,openai=False):
    """
    Creates a stacked bar plot for financial data grouped by category and sub-category with enhanced formatting.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the financial data
    numeric_column : str
        Name of the numeric column to analyze
    categorical_column : str
        Name of the categorical column for grouping (x-axis categories)
    stack_column : str
        Name of the categorical column for stacking (sub-categories)
    estimator : str, default 'mean'
        Statistical estimator to apply from AVAILABLE_ESTIMATORS
    figsize : tuple, default (12, 7)
        Figure size (width, height)
    rotation : int, default 45
        Rotation angle for x-axis labels
    horizontal_params : dict, optional
        Dictionary with up to 2 key-value pairs for horizontal oval metrics.
        Format: {'estimator': 'numeric_column_name', 'estimator2': 'numeric_column_name2'}
    vertical_params : dict, optional
        Dictionary with up to 2 key-value pairs for vertical oval metrics grouped by stack categories.
        Format: {'estimator': 'numeric_column_name', 'estimator2': 'numeric_column_name2'}

    Returns:
    --------
    tuple
        (matplotlib.axes.Axes, dict) - Axes object and enhanced insights JSON
    """

    # Build list of essential columns for main chart (only these are required)
    essential_columns = [numeric_column, categorical_column, stack_column]

    # Build list of additional columns to validate (but not required for main chart)
    additional_columns = []
    if horizontal_params:
        # Validate horizontal_params format first
        if not isinstance(horizontal_params, dict) or len(horizontal_params) > 2:
            raise ValueError("horizontal_params must be a dictionary with maximum 2 key-value pairs")
        additional_columns.extend(horizontal_params.values())

    if vertical_params:
        # Validate vertical_params format first
        if not isinstance(vertical_params, dict) or len(vertical_params) > 2:
            raise ValueError("vertical_params must be a dictionary with maximum 2 key-value pairs")
        additional_columns.extend(vertical_params.values())

    # Validate that all columns exist in DataFrame
    all_columns = essential_columns + additional_columns
    for col in all_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame")

    if estimator not in AVAILABLE_ESTIMATORS:
        raise ValueError(f"Invalid estimator '{estimator}'. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

    # Validate that all numeric columns are actually numeric
    numeric_columns_to_check = [numeric_column] + additional_columns
    for col in numeric_columns_to_check:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")

    # Additional validation for horizontal_params and vertical_params estimators
    if horizontal_params:
        for est, col in horizontal_params.items():
            if est not in AVAILABLE_ESTIMATORS:
                raise ValueError(f"Invalid estimator '{est}' in horizontal_params. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

    if vertical_params:
        for est, col in vertical_params.items():
            if est not in AVAILABLE_ESTIMATORS:
                raise ValueError(f"Invalid estimator '{est}' in vertical_params. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

    # Create base dataset: only remove nulls from essential columns for main chart
    df_base = df[essential_columns].dropna()

    insights = generate_multilevel_aggregations(df_base, categorical_column, stack_column, numeric_column, estimator, values="percentages")
    insights_JSON = add_pareto_insights(insights)
    if openai is True:
      insights_JSON = generate_strategic_insights(
          categorical_column, stack_column,numeric_column,insights_JSON,
          API_KEY=API_KEY
      )

    if df_base.empty:
        raise ValueError("No valid data after removing null values from essential columns")

    # Calculate estimator by category and stack category using base dataset
    pivot_data = df_base.pivot_table(values=numeric_column,
                                    index=categorical_column,
                                    columns=stack_column,
                                    aggfunc=AVAILABLE_ESTIMATORS[estimator],
                                    fill_value=0)

    # Format category names
    formatted_categories = [str(cat).capitalize() for cat in pivot_data.index]

    # Get unique stack categories and create custom color palette
    stack_categories = pivot_data.columns.tolist()
    colors = create_custom_color_palette(len(stack_categories))

    # Configure style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)

    # Setup transparent background and styling
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Get all values for consistent scaling (use absolute values for scale calculation)
    all_values = pivot_data.values.flatten()
    all_values = all_values[~pd.isna(all_values)]
    scale, suffix = get_scale_and_format_eur(all_values, numeric_column)

    # Create stacked bar plot
    bar_width = 0.45
    x_positions = np.arange(len(formatted_categories))

    # Initialize bottom values for stacking
    bottoms = np.zeros(len(formatted_categories))

    # Calculate total values for each category (for top labels) - using original signed values
    category_totals = pivot_data.sum(axis=1).values

    # 🔥 CORRECCIÓN PARA MANEJO DE NEGATIVOS: Plot each stack layer
    bars_list = []
    for i, stack_cat in enumerate(stack_categories):
        original_values = pivot_data[stack_cat].values  # Valores originales (con signo)
        display_values = np.abs(original_values)        # Valores absolutos para altura visual

        bars = ax.bar(x_positions, display_values, bar_width, bottom=bottoms,
                     color=colors[i], edgecolor='white', linewidth=1,
                     label=str(stack_cat).capitalize())
        bars_list.append(bars)

        # 🔥 MODIFICACIÓN: Calculate visual totals (sum of absolute values) for each category
        visual_category_totals = []
        for j in range(len(formatted_categories)):
            visual_total = sum(np.abs(pivot_data.iloc[j].values))  # Suma de valores absolutos
            visual_category_totals.append(visual_total)

        # Add percentage labels inside segments (only if segment is tall enough)
        min_height_for_percentage = max(abs(v) for v in all_values) * 0.06  # 6% of max absolute value

        for j, (bar, original_value, display_value) in enumerate(zip(bars, original_values, display_values)):
            # Show percentage if the absolute value is large enough
            if original_value != 0 and abs(original_value) >= min_height_for_percentage:
                # 🔥 CAMBIO PRINCIPAL: Calculate percentage relative to VISUAL height (absolute values)
                visual_category_total = visual_category_totals[j]  # Total visual (valores absolutos)
                percentage = (abs(original_value) / visual_category_total * 100) if visual_category_total != 0 else 0

                # Position label in the center of the VISUAL segment (display_value)
                center_y = bottoms[j] + display_value/2

                # Format percentage (always positive now since it's relative to visual height)
                percentage_text = f'{percentage:.0f}%'

                # Choose text color based on original value sign (for visual distinction)
                text_color = '#FF6B6B' if original_value < 0 else 'white'  # Rojo para negativos, blanco para positivos

                ax.text(bar.get_x() + bar.get_width()/2, center_y, percentage_text,
                       ha='center', va='center', color=text_color,
                       fontweight='bold', fontsize=10)

        # Update bottom values for next stack using DISPLAY values (absolute)
        bottoms += display_values

    # Add total value labels on top of each bar with SIMPLIFIED formatting
    label_offset = max(abs(v) for v in all_values) * 0.02  # 2% offset above the bar

    for i, (x_pos, total_value) in enumerate(zip(x_positions, category_totals)):
        if total_value != 0:  # 🔥 CORRECCIÓN: Mostrar totales tanto positivos como negativos
            # Use formatting functions for consistency
            total_scale, total_suffix = _get_custom_scale_and_suffix_standalone([abs(total_value)])
            formatted_total = _format_oval_value_standalone(abs(total_value), total_scale, total_suffix)

            # Add sign prefix for negative totals
            if total_value < 0:
                formatted_total = f"-{formatted_total}"

            # Position label above the bar (use sum of display values for positioning)
            visual_bar_height = sum(np.abs(pivot_data.iloc[i].values))
            label_y = visual_bar_height + label_offset

            # Add total label in bold
            ax.text(x_pos, label_y, formatted_total,
                   ha='center', va='bottom', color='black', fontweight='bold', fontsize=11)

    # Generate intelligent Y-axis label
    label, unit, year = generate_intelligent_ylabel(numeric_column, estimator)

    # Add scale suffix to unit for EUR columns if applicable
    if 'EUR' in numeric_column.upper() and suffix and estimator != 'count':
        unit = f'€ {suffix}' if suffix else '€'
    elif estimator == 'count':
        unit = '# firms'

    # Construct final label
    if year and unit:
        ylabel = f'{label} ({year}, {unit})'
    elif unit:
        ylabel = f'{label} ({unit})'
    elif year:
        ylabel = f'{label} ({year})'
    else:
        ylabel = label

    # Configure axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(formatted_categories, fontweight='normal')
    ax.set_xlabel("", fontweight='normal')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')

    # Apply consistent styling
    ax = setup_chart_style(ax)

    # Add legend with different positions based on params
    if horizontal_params or vertical_params:
        # When there are params, keep legend on the left
        ax.legend(loc='lower left', bbox_to_anchor=(-0.30, 0.5), frameon=False, fontsize=10)
    else:
        # When no params, place legend on the right
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)

    # Rotate x-axis labels if needed
    if rotation > 0:
        plt.xticks(rotation=rotation, ha='right')

    # Add light grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # ================================
    # ENHANCED HORIZONTAL OVAL METRICS WITH INSIGHTS
    # ================================
    if horizontal_params:
        # Calculate metrics for horizontal ovals
        oval_data = []
        for est, col in horizontal_params.items():
            # Create dataset specific for this column
            df_for_this_metric = df_base.copy()
            if col in df.columns:
                df_for_this_metric[col] = df.loc[df_base.index, col]
                df_for_this_metric = df_for_this_metric.dropna(subset=[col])

            # Calculate metric for each category
            if not df_for_this_metric.empty:
                category_metrics = df_for_this_metric.groupby(categorical_column)[col].agg(AVAILABLE_ESTIMATORS[est])
                category_metrics = category_metrics.reindex(pivot_data.index, fill_value=0)
            else:
                category_metrics = pd.Series(0, index=pivot_data.index)

            # Add horizontal insights to JSON
            for category, value in category_metrics.items():
                insight_key = f"{est.title()} {col} for {str(category).title()}"
                insights_JSON[insight_key] = value

            # Use improved formatting system
            col_values = category_metrics.values
            col_scale, col_suffix = _get_custom_scale_and_suffix_standalone(col_values)

            oval_data.append({
                'estimator': est,
                'column': col,
                'values': category_metrics.values,
                'scale': col_scale,
                'suffix': col_suffix,
                'formatted_values': [_format_oval_value_standalone(v, col_scale, col_suffix, est) for v in category_metrics.values],
                'title': _create_oval_title_standalone(est, col)
            })

        # Calculate positioning with dynamic spacing
        max_visual_bar_height = max(sum(np.abs(pivot_data.iloc[i].values)) for i in range(len(pivot_data)))
        dynamic_offset = max_visual_bar_height * 0.15
        total_label_space = max_visual_bar_height * 0.05

        oval_base_y = max_visual_bar_height + total_label_space + dynamic_offset
        oval_height = max_visual_bar_height * 0.08
        oval_spacing = max_visual_bar_height * 0.12

        for row_idx, oval_info in enumerate(oval_data):
            oval_y = oval_base_y + (row_idx * oval_spacing)

            # Use custom title from processed data
            title = oval_info['title']
            ax.text(-0.8, oval_y, title, ha='right', va='center',
                   fontsize=10, fontweight='normal', color='black')

            # Add ovals for each category
            for i, (x_pos, formatted_value) in enumerate(zip(x_positions, oval_info['formatted_values'])):
                # Create oval using Ellipse
                from matplotlib.patches import Ellipse
                oval_width = 0.55
                oval = Ellipse((x_pos, oval_y), oval_width, oval_height,
                             facecolor='lightgray', edgecolor='gray',
                             linewidth=0.5, alpha=0.8)
                ax.add_patch(oval)

                # Use formatted value with consistent 2 decimal places
                ax.text(x_pos, oval_y, formatted_value,
                       ha='center', va='center', color='black',
                       fontweight='bold', fontsize=9)

        # Adjust ylim to accommodate ovals with dynamic spacing
        if oval_data:
            top_oval_y = oval_base_y + ((len(oval_data) - 1) * oval_spacing) + oval_height/2
            ax.set_ylim(0, top_oval_y * 1.05)

    # ================================
    # ENHANCED VERTICAL OVAL METRICS WITH INSIGHTS
    # ================================
    if vertical_params:
        # Calculate metrics for vertical ovals
        vertical_oval_data = []
        for est, col in vertical_params.items():
            df_for_this_metric = df_base.copy()
            if col in df.columns:
                df_for_this_metric[col] = df.loc[df_base.index, col]
                df_for_this_metric = df_for_this_metric.dropna(subset=[col])

            if not df_for_this_metric.empty:
                stack_metrics = df_for_this_metric.groupby(stack_column)[col].agg(AVAILABLE_ESTIMATORS[est])
                stack_metrics = stack_metrics.reindex(stack_categories, fill_value=0)
            else:
                stack_metrics = pd.Series(0, index=stack_categories)

            # Add vertical insights to JSON
            for stack_category, value in stack_metrics.items():
                insight_key = f"{est.title()} {col} for {str(stack_category).title()}"
                insights_JSON[insight_key] = value

            # Use improved formatting system
            col_values = stack_metrics.values
            col_scale, col_suffix = _get_custom_scale_and_suffix_standalone(col_values)

            vertical_oval_data.append({
                'estimator': est,
                'column': col,
                'values': stack_metrics.values,
                'categories': stack_metrics.index.tolist(),
                'scale': col_scale,
                'suffix': col_suffix,
                'formatted_values': [_format_oval_value_standalone(v, col_scale, col_suffix, est) for v in stack_metrics.values],  # 🔥 MODIFICACIÓN: Pass estimator for count handling
                'title': _create_oval_title_standalone(est, col)
            })

        # Position vertical ovals on the right side
        max_visual_bar_height = max(sum(np.abs(pivot_data.iloc[i].values)) for i in range(len(pivot_data))) if len(pivot_data) > 0 else 100
        chart_right = len(formatted_categories) - 0.5
        vertical_oval_base_x = chart_right + 0.8
        vertical_oval_width = 0.55
        vertical_oval_spacing = 0.6

        # Calculate vertical positions for ovals
        chart_height = max_visual_bar_height
        vertical_oval_height = chart_height * 0.08
        stack_spacing = chart_height * 0.15

        total_stack_height = (len(stack_categories) - 1) * stack_spacing
        vertical_oval_start_y = (chart_height - total_stack_height) / 2

        for col_idx, oval_info in enumerate(vertical_oval_data):
            oval_x = vertical_oval_base_x + (col_idx * vertical_oval_spacing)

            # Use custom title from processed data
            title = oval_info['title']
            title_y = chart_height * 0.95

            # Split long titles into multiple lines
            max_chars_per_line = 10
            words = title.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= max_chars_per_line or not current_line:
                    current_line.append(word)
                    current_length += len(word) + (1 if current_line else 0)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                lines.append(' '.join(current_line))

            title_text = '\n'.join(lines)
            ax.text(oval_x, title_y, title_text, ha='center', va='bottom',
                   fontsize=10, fontweight='normal', color='black',
                   rotation=0, multialignment='center')

            # Add ovals for each stack category
            for i, (stack_cat, formatted_value) in enumerate(zip(oval_info['categories'], oval_info['formatted_values'])):
                oval_y = vertical_oval_start_y + (i * stack_spacing)

                # Create oval with corresponding stack category color
                from matplotlib.patches import Ellipse
                oval = Ellipse((oval_x, oval_y), vertical_oval_width, vertical_oval_height,
                             facecolor=colors[i], edgecolor='white',
                             linewidth=1, alpha=0.9)
                ax.add_patch(oval)

                # Use formatted value with consistent 2 decimal places
                ax.text(oval_x, oval_y, formatted_value,
                       ha='center', va='center', color='white',
                       fontweight='bold', fontsize=9)

        # Adjust xlim to accommodate vertical ovals
        if vertical_oval_data:
            rightmost_oval_x = vertical_oval_base_x + ((len(vertical_oval_data) - 1) * vertical_oval_spacing) + vertical_oval_width/2
            ax.set_xlim(ax.get_xlim()[0], rightmost_oval_x * 1.05)

    if not horizontal_params and not vertical_params:
        # Add some extra space at the top for the total labels (using visual height)
        max_visual_bar_height = max(sum(np.abs(pivot_data.iloc[i].values)) for i in range(len(pivot_data))) if len(pivot_data) > 0 else 0
        ax.set_ylim(0, max_visual_bar_height * 1.1)

    # Adjust layout
    plt.tight_layout()

    return ax, insights_JSON

def plot_financial_stacked_barplot_100(df, numeric_column, categorical_column, stack_column, estimator='mean',
                                       figsize=(12, 7), rotation=45, horizontal_params=None, vertical_params=None,openai=False):
    """
    Creates a 100% stacked bar plot for financial data grouped by category and sub-category with enhanced formatting.
    Now handles negative values correctly by normalizing bar heights to 100% while preserving original percentages in labels.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the financial data
    numeric_column : str
        Name of the numeric column to analyze
    categorical_column : str
        Name of the categorical column for grouping (x-axis categories)
    stack_column : str
        Name of the categorical column for stacking (sub-categories)
    estimator : str, default 'mean'
        Statistical estimator to apply from AVAILABLE_ESTIMATORS
    figsize : tuple, default (12, 7)
        Figure size (width, height)
    rotation : int, default 45
        Rotation angle for x-axis labels
    horizontal_params : dict, optional
        Dictionary with up to 2 key-value pairs for horizontal oval metrics
    vertical_params : dict, optional
        Dictionary with up to 2 key-value pairs for vertical oval metrics

    Returns:
    --------
    tuple
        (matplotlib.axes.Axes, dict) - Axes object and enhanced insights JSON
    """

    # Build list of essential columns for main chart
    essential_columns = [numeric_column, categorical_column, stack_column]

    # Build list of additional columns to validate
    additional_columns = []
    if horizontal_params:
        if not isinstance(horizontal_params, dict) or len(horizontal_params) > 2:
            raise ValueError("horizontal_params must be a dictionary with maximum 2 key-value pairs")
        additional_columns.extend(horizontal_params.values())

    if vertical_params:
        if not isinstance(vertical_params, dict) or len(vertical_params) > 2:
            raise ValueError("vertical_params must be a dictionary with maximum 2 key-value pairs")
        additional_columns.extend(vertical_params.values())

    # Validate that all columns exist in DataFrame
    all_columns = essential_columns + additional_columns
    for col in all_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame")

    if estimator not in AVAILABLE_ESTIMATORS:
        raise ValueError(f"Invalid estimator '{estimator}'. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

    # Validate that all numeric columns are actually numeric
    numeric_columns_to_check = [numeric_column] + additional_columns
    for col in set(numeric_columns_to_check):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")

    # Additional validation for params estimators
    if horizontal_params:
        for est, col in horizontal_params.items():
            if est not in AVAILABLE_ESTIMATORS:
                raise ValueError(f"Invalid estimator '{est}' in horizontal_params. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

    if vertical_params:
        for est, col in vertical_params.items():
            if est not in AVAILABLE_ESTIMATORS:
                raise ValueError(f"Invalid estimator '{est}' in vertical_params. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

    # Create base dataset
    df_base = df[essential_columns].dropna()

    insights = generate_multilevel_aggregations(df_base, categorical_column, stack_column, numeric_column, estimator, values="percentages")
    insights_JSON = add_pareto_insights(insights)
    if openai is True:
      insights_JSON = generate_strategic_insights(
          categorical_column, stack_column,numeric_column,insights_JSON,
          API_KEY=API_KEY
      )

    if df_base.empty:
        raise ValueError("No valid data after removing null values from essential columns")

    # Calculate estimator by category and stack category
    pivot_data = df_base.pivot_table(values=numeric_column,
                                    index=categorical_column,
                                    columns=stack_column,
                                    aggfunc=AVAILABLE_ESTIMATORS[estimator],
                                    fill_value=0)

    # Convert to percentages for labels (original percentages)
    pivot_data_percentage_original = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
    pivot_data_percentage_original = pivot_data_percentage_original.fillna(0)

    # 🔥 NEW: Calculate normalized percentages for visual heights
    # This ensures all bars have the same visual height (100%) even with negative values
    pivot_data_percentage_normalized = pivot_data_percentage_original.copy()

    # Normalize each row (category) to sum to 100% in absolute terms
    for category in pivot_data_percentage_normalized.index:
        row_values = pivot_data_percentage_normalized.loc[category]

        # Calculate sum of absolute values
        abs_sum = row_values.abs().sum()

        # Only normalize if abs_sum > 0 to avoid division by zero
        if abs_sum > 0:
            # Normalize: each value × (100 / sum_of_absolute_values)
            normalization_factor = 100 / abs_sum
            pivot_data_percentage_normalized.loc[category] = row_values * normalization_factor

    # Format category names
    formatted_categories = [str(cat).capitalize() for cat in pivot_data_percentage_normalized.index]

    # Get unique stack categories and create custom color palette
    stack_categories = pivot_data_percentage_normalized.columns.tolist()
    colors = create_custom_color_palette(len(stack_categories))

    # Configure style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)

    # Setup transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Create stacked bar plot (100% with normalized heights)
    bar_width = 0.45
    x_positions = np.arange(len(formatted_categories))

    # Plot each stack layer using NORMALIZED values for heights (always positive)
    bars_list = []
    bottoms = np.zeros(len(formatted_categories))  # Single bottom array since all go upwards

    for i, stack_cat in enumerate(stack_categories):
        # Use normalized values for visual heights (always positive)
        visual_values = pivot_data_percentage_normalized[stack_cat].values
        # Use absolute values for bar heights (always positive)
        bar_heights = np.abs(visual_values)

        # Keep original values for labels (can be negative)
        original_values = pivot_data_percentage_original[stack_cat].values

        bars = ax.bar(x_positions, bar_heights, bar_width, bottom=bottoms,
                     color=colors[i], edgecolor='white', linewidth=1,
                     label=str(stack_cat).capitalize())
        bars_list.append(bars)

        # 🔥 MODIFICACIÓN: Calculate visual percentages relative to normalized heights
        visual_category_totals_normalized = []
        for j in range(len(formatted_categories)):
            # Sum of absolute normalized values (always 100% for each category)
            visual_total = sum(np.abs(pivot_data_percentage_normalized.iloc[j].values))
            visual_category_totals_normalized.append(visual_total)

        # Add percentage labels inside segments (only if segment is tall enough)
        min_height_for_percentage = 5  # Minimum 5% visual height to show label

        for j, (bar, bar_height, original_value) in enumerate(zip(bars, bar_heights, original_values)):
            if bar_height >= min_height_for_percentage:
                # Position label in the center of the segment (using visual height)
                center_y = bottoms[j] + bar_height/2

                # 🔥 CAMBIO PRINCIPAL: Calculate percentage relative to VISUAL height (normalized values)
                visual_category_total = visual_category_totals_normalized[j]  # Total visual normalizado (siempre 100%)
                visual_percentage = (bar_height / visual_category_total * 100) if visual_category_total != 0 else 0

                # Choose text color based on original value sign (for visual distinction)
                text_color = '#FF6B6B' if original_value < 0 else 'white'  # Rojo para negativos, blanco para positivos

                # Format percentage (always positive now since it's relative to visual height)
                percentage_text = f'{visual_percentage:.0f}%'

                ax.text(bar.get_x() + bar.get_width()/2, center_y, percentage_text,
                       ha='center', va='center', color=text_color, fontweight='normal', fontsize=10)

        # Update bottom values for next stack (using absolute visual heights)
        bottoms += bar_heights

    # 🔥 MEJORA 2: Add total value labels on top of each bar with IMPROVED formatting
    category_totals = pivot_data.sum(axis=1).values

    # Get all original values for consistent scaling
    all_original_values = pivot_data.values.flatten()
    all_original_values = all_original_values[~pd.isna(all_original_values)]
    scale, suffix = get_scale_and_format_eur(all_original_values, numeric_column)

    # Add total value labels on top of each bar with NEW FORMAT
    label_offset = 3  # 3% offset above the 100% bar

    for i, (x_pos, total_value) in enumerate(zip(x_positions, category_totals)):
        if total_value != 0:
            # 🔥 NUEVO FORMATO MEJORADO usando funciones standalone
            total_scale, total_suffix = _get_custom_scale_and_suffix_standalone([abs(total_value)])
            formatted_total = _format_oval_value_standalone(abs(total_value), total_scale, total_suffix, estimator)

            # Add sign prefix for negative totals
            if total_value < 0:
                formatted_total = f"-{formatted_total}"

            # Position label above the 100% bar
            label_y = 100 + label_offset

            # Add total label in bold
            ax.text(x_pos, label_y, formatted_total,
                   ha='center', va='bottom', color='black', fontweight='bold', fontsize=11)

    # Generate intelligent Y-axis label for 100% stacked chart
    label, unit, year = generate_intelligent_ylabel(numeric_column, estimator)

    # For 100% stacked charts, always add "Distribution" and show as percentage
    if label:
        ylabel_base = f'{label} Distribution'
    else:
        ylabel_base = 'Distribution'

    # Add year and percentage unit for 100% stacked
    if year:
        ylabel = f'{ylabel_base} ({year}, %)'
    else:
        ylabel = f'{ylabel_base} (%)'

    # Configure axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(formatted_categories, fontweight='normal')
    ax.set_xlabel("", fontweight='normal')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')

    # Set Y-axis to 0-100% since all bars go upwards and are normalized
    ax.set_ylim(0, 100)

    # Apply consistent styling
    ax = setup_chart_style(ax)

    # Add legend with different positions based on params
    if horizontal_params or vertical_params:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.35, 0.5), frameon=False, fontsize=10)
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)

    # Rotate x-axis labels if needed
    if rotation > 0:
        plt.xticks(rotation=rotation, ha='right')

    # Add light grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # ================================
    # 🔥 MEJORA 3+4: ENHANCED HORIZONTAL OVAL METRICS with DYNAMIC positioning and WIDER ovals
    # ================================
    if horizontal_params:
        oval_data = []
        for est, col in horizontal_params.items():
            df_for_this_metric = df_base.copy()
            if col in df.columns:
                df_for_this_metric[col] = df.loc[df_base.index, col]
                df_for_this_metric = df_for_this_metric.dropna(subset=[col])

            if not df_for_this_metric.empty:
                category_metrics = df_for_this_metric.groupby(categorical_column)[col].agg(AVAILABLE_ESTIMATORS[est])
                category_metrics = category_metrics.reindex(pivot_data.index, fill_value=0)
            else:
                category_metrics = pd.Series(0, index=pivot_data.index)

            # 🔥 MEJORA 6: Add horizontal insights to JSON using STANDALONE functions
            for category, value in category_metrics.items():
                insight_key = f"{est.title()} {col} for {str(category).title()}"
                insights_JSON[insight_key] = value

            # 🔥 MEJORA 6: Use improved formatting system with standalone functions
            col_values = category_metrics.values
            col_scale, col_suffix = _get_custom_scale_and_suffix_standalone(col_values)

            oval_data.append({
                'estimator': est,
                'column': col,
                'values': category_metrics.values,
                'scale': col_scale,
                'suffix': col_suffix,
                'formatted_values': [_format_oval_value_standalone(v, col_scale, col_suffix, est) for v in category_metrics.values],
                'title': _create_oval_title_standalone(est, col)
            })

        # 🔥 MEJORA 3: Position ovals above the chart with DYNAMIC spacing
        chart_height = 100  # Fixed height for 100% stacked chart
        dynamic_offset = chart_height * 0.15  # 15% dynamic offset
        total_label_space = 8  # Space for total labels (3% offset + label height)

        oval_base_y = chart_height + total_label_space + dynamic_offset  # 115 + dynamic offset
        oval_height = 6    # Reduced height for more oval shape
        oval_spacing = 12  # Fixed spacing between oval rows

        for row_idx, oval_info in enumerate(oval_data):
            oval_y = oval_base_y + (row_idx * oval_spacing)

            # 🔥 MEJORA 6: Use custom title from processed data
            title = oval_info['title']
            ax.text(-0.8, oval_y, title, ha='right', va='center',
                   fontsize=10, fontweight='normal', color='black')

            # Add ovals for each category
            for i, (x_pos, formatted_value) in enumerate(zip(x_positions, oval_info['formatted_values'])):
                from matplotlib.patches import Ellipse
                # 🔥 MEJORA 4: WIDER oval to match vertical ovals
                oval_width = 0.55  # Changed from 0.45 to 0.55 (same as vertical)
                oval = Ellipse((x_pos, oval_y), oval_width, oval_height,
                             facecolor='lightgray', edgecolor='gray',
                             linewidth=0.5, alpha=0.8)
                ax.add_patch(oval)

                # 🔥 MEJORA 6+8: Use formatted value with consistent formatting and count handling
                ax.text(x_pos, oval_y, formatted_value,
                       ha='center', va='center', color='black',
                       fontweight='bold', fontsize=9)

        # 🔥 MEJORA 7: Adjust ylim to accommodate ovals with improved calculations
        if oval_data:
            top_oval_y = oval_base_y + ((len(oval_data) - 1) * oval_spacing) + oval_height/2
            ax.set_ylim(0, top_oval_y * 1.05)

    # ================================
    # 🔥 MEJORA 6+8: ENHANCED VERTICAL OVAL METRICS with STANDALONE functions
    # ================================
    if vertical_params:
        vertical_oval_data = []
        for est, col in vertical_params.items():
            df_for_this_metric = df_base.copy()
            if col in df.columns:
                df_for_this_metric[col] = df.loc[df_base.index, col]
                df_for_this_metric = df_for_this_metric.dropna(subset=[col])

            if not df_for_this_metric.empty:
                stack_metrics = df_for_this_metric.groupby(stack_column)[col].agg(AVAILABLE_ESTIMATORS[est])
                stack_metrics = stack_metrics.reindex(stack_categories, fill_value=0)
            else:
                stack_metrics = pd.Series(0, index=stack_categories)

            # 🔥 MEJORA 6: Add vertical insights to JSON using STANDALONE functions
            for stack_category, value in stack_metrics.items():
                insight_key = f"{est.title()} {col} for {str(stack_category).title()}"
                insights_JSON[insight_key] = value

            # 🔥 MEJORA 6: Use improved formatting system with standalone functions
            col_values = stack_metrics.values
            col_scale, col_suffix = _get_custom_scale_and_suffix_standalone(col_values)

            vertical_oval_data.append({
                'estimator': est,
                'column': col,
                'values': stack_metrics.values,
                'categories': stack_metrics.index.tolist(),
                'scale': col_scale,
                'suffix': col_suffix,
                'formatted_values': [_format_oval_value_standalone(v, col_scale, col_suffix, est) for v in stack_metrics.values],  # 🔥 MODIFICACIÓN: Pass estimator for count handling
                'title': _create_oval_title_standalone(est, col)
            })

        # Position vertical ovals on the right side of the chart
        chart_right = len(formatted_categories) - 0.5
        vertical_oval_base_x = chart_right + 0.8
        vertical_oval_width = 0.55   # Wider ovals (more width than height)
        vertical_oval_height = 7     # Less height (shorter ovals)
        vertical_oval_spacing = 0.6

        chart_height = 100  # Fixed height for 100% stacked chart
        stack_spacing = 10   # Reduced spacing between ovals (closer together)
        total_stack_height = (len(stack_categories) - 1) * stack_spacing
        vertical_oval_start_y = (chart_height - total_stack_height) / 2

        for col_idx, oval_info in enumerate(vertical_oval_data):
            oval_x = vertical_oval_base_x + (col_idx * vertical_oval_spacing)

            # 🔥 MEJORA 6: Use custom title from processed data
            title = oval_info['title']
            title_y = chart_height * 0.95

            # Split long titles into multiple lines
            max_chars_per_line = 10
            words = title.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= max_chars_per_line or not current_line:
                    current_line.append(word)
                    current_length += len(word) + (1 if current_line else 0)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                lines.append(' '.join(current_line))

            title_text = '\n'.join(lines)
            ax.text(oval_x, title_y, title_text, ha='center', va='bottom',
                   fontsize=10, fontweight='normal', color='black',
                   rotation=0, multialignment='center')

            # Add ovals for each stack category
            for i, (stack_cat, formatted_value) in enumerate(zip(oval_info['categories'], oval_info['formatted_values'])):
                oval_y = vertical_oval_start_y + (i * stack_spacing)

                from matplotlib.patches import Ellipse
                oval = Ellipse((oval_x, oval_y), vertical_oval_width, vertical_oval_height,
                             facecolor=colors[i], edgecolor='white',
                             linewidth=1, alpha=0.9)
                ax.add_patch(oval)

                # 🔥 MEJORA 6+8: Use formatted value with consistent formatting and count handling
                ax.text(oval_x, oval_y, formatted_value,
                       ha='center', va='center', color='white',
                       fontweight='bold', fontsize=9)

        # 🔥 MEJORA 7: Adjust xlim to accommodate vertical ovals with improved calculations
        if vertical_oval_data:
            rightmost_oval_x = vertical_oval_base_x + ((len(vertical_oval_data) - 1) * vertical_oval_spacing) + vertical_oval_width/2
            ax.set_xlim(ax.get_xlim()[0], rightmost_oval_x * 1.05)

    if not horizontal_params and not vertical_params:
        # Set standard limits for 100% stacked chart without ovals
        ax.set_ylim(0, 100)

    # Adjust layout
    plt.tight_layout()

    return ax, insights_JSON

def plot_financial_mekko_chart_100(df, numeric_column, categorical_column, stack_column,
                                   figsize=(14, 8), rotation=45,horizontal_params=None, vertical_params=None,openai=False):
    """
    Creates a 100% MEKKO Chart (Marimekko) for financial data with rescaled percentages.

    The MEKKO chart displays:
    - Bar widths: Percentage that each main category represents of the total
    - Segment heights: Rescaled percentages (all positive, sum to 100% per category)
    - All segments: Stacked above the baseline (y=0) after rescaling
    - Labels: Show original percentages with their corresponding signs

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the financial data
    numeric_column : str
        Name of the numeric column to analyze (determines both widths and heights)
    categorical_column : str
        Name of the categorical column for main categories (x-axis)
    stack_column : str
        Name of the categorical column for subcategories (stacked segments)
    figsize : tuple, default (14, 8)
        Figure size (width, height)
    rotation : int, default 45
        Rotation angle for x-axis labels
    horizontal_params : dict, optional
        Dictionary with up to 2 key-value pairs for horizontal oval metrics
    vertical_params : dict, optional
        Dictionary with up to 2 key-value pairs for vertical oval metrics

    Returns:
    --------
    tuple
        (matplotlib.axes.Axes, dict) - Axes object and enhanced insights JSON
    """

    # Build list of essential columns for main chart
    essential_columns = [numeric_column, categorical_column, stack_column]

    # Build list of additional columns to validate
    additional_columns = []
    if horizontal_params:
        if not isinstance(horizontal_params, dict) or len(horizontal_params) > 2:
            raise ValueError("horizontal_params must be a dictionary with maximum 2 key-value pairs")
        for est, col in horizontal_params.items():
            if est not in AVAILABLE_ESTIMATORS:
                raise ValueError(f"Invalid estimator '{est}' in horizontal_params. Options: {list(AVAILABLE_ESTIMATORS.keys())}")
            additional_columns.append(col)

    if vertical_params:
        if not isinstance(vertical_params, dict) or len(vertical_params) > 2:
            raise ValueError("vertical_params must be a dictionary with maximum 2 key-value pairs")
        for est, col in vertical_params.items():
            if est not in AVAILABLE_ESTIMATORS:
                raise ValueError(f"Invalid estimator '{est}' in vertical_params. Options: {list(AVAILABLE_ESTIMATORS.keys())}")
            additional_columns.append(col)

    # Validate that all columns exist in DataFrame
    all_columns = essential_columns + additional_columns
    for col in all_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame")

    # Validate that all numeric columns are actually numeric
    numeric_columns_to_check = [numeric_column] + additional_columns
    for col in set(numeric_columns_to_check):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")

    # Create base dataset
    df_base = df[essential_columns].copy()

    # Note: For MEKKO charts, we use 'sum' as the default estimator
    insights = generate_multilevel_aggregations(df_base, categorical_column, stack_column, numeric_column, 'sum', values="percentages")
    insights_JSON = add_pareto_insights(insights)
    if openai is True:
      insights_JSON = generate_strategic_insights(
          categorical_column, stack_column,numeric_column,insights_JSON,
          API_KEY=API_KEY
      )

    # Convert categorical columns to strings and handle NaN/None values
    df_base[categorical_column] = df_base[categorical_column].astype(str).replace(['nan', 'None', ''], 'Unknown')
    df_base[stack_column] = df_base[stack_column].astype(str).replace(['nan', 'None', ''], 'Unknown')

    # For the main MEKKO chart: replace NaN in numeric_column with 0
    df_base[numeric_column] = df_base[numeric_column].fillna(0)

    if df_base.empty:
        raise ValueError("No data available after cleaning essential columns")

    # MEKKO CALCULATIONS
    # 1. Calculate widths: Total ABSOLUTE VALUE by main category / Grand total absolute
    category_totals_abs = df_base.groupby(categorical_column)[numeric_column].apply(lambda x: x.abs().sum())
    grand_total_abs = category_totals_abs.sum()

    if grand_total_abs == 0:
        raise ValueError("Grand total (absolute values) is zero - no valid data to plot")

    width_percentages = (category_totals_abs / grand_total_abs) * 100

    # Also calculate original totals (with signs) for display labels
    category_totals_original = df_base.groupby(categorical_column)[numeric_column].sum()

    # 2. Calculate heights: Subcategory values within each main category
    pivot_data = df_base.pivot_table(
        values=numeric_column,
        index=categorical_column,
        columns=stack_column,
        aggfunc='sum',
        fill_value=0
    )

    # Calculate original percentages (with signs) within each main category
    height_percentages_original = pivot_data.div(category_totals_original, axis=0) * 100
    height_percentages_original = height_percentages_original.fillna(0)

    # RESCALE PERCENTAGES: Convert to absolute values and rescale to sum to 100%
    height_percentages_rescaled = {}

    for category in height_percentages_original.index:
        category_data = height_percentages_original.loc[category]

        # Convert to absolute values
        abs_percentages = category_data.abs()

        # Calculate sum of absolute values
        total_abs = abs_percentages.sum()

        if total_abs > 0:
            # Rescale to sum to 100%
            rescaled = (abs_percentages / total_abs) * 100

            # Round to 1 decimal place and adjust to ensure sum is exactly 100%
            rescaled_rounded = rescaled.round(1)

            # Adjust for rounding errors to ensure sum = 100%
            diff = 100.0 - rescaled_rounded.sum()
            if abs(diff) > 0.01:  # Only adjust if difference is significant
                # Add/subtract the difference to the largest value
                max_idx = rescaled_rounded.abs().idxmax()
                rescaled_rounded[max_idx] += diff
                rescaled_rounded[max_idx] = round(rescaled_rounded[max_idx], 1)

            height_percentages_rescaled[category] = rescaled_rounded
        else:
            # If all values are zero, set all to zero
            height_percentages_rescaled[category] = pd.Series(0, index=category_data.index)

    # Convert back to DataFrame
    height_percentages = pd.DataFrame(height_percentages_rescaled).T
    height_percentages = height_percentages.fillna(0)

    # Get categories and stack categories
    categories = width_percentages.index.tolist()
    stack_categories = height_percentages.columns.tolist()
    formatted_categories = [str(cat).capitalize() for cat in categories]

    # INTELLIGENT COLOR ASSIGNMENT
    # Custom color palette with red tones for 'loss' subcategories
    base_colors = ['#7D4BEB', '#948f9c', '#f54298', '#330599', '#331f5e', '#F1D454', '#F5A25C', '#EB6B63']
    red_tones = ['#DC143C', '#B22222', '#8B0000', '#CD5C5C']  # Different shades of red

    # Identify subcategories with 'loss' in their name
    loss_categories = [cat for cat in stack_categories if 'loss' in str(cat).lower()]
    non_loss_categories = [cat for cat in stack_categories if 'loss' not in str(cat).lower()]

    # Assign colors
    colors = {}

    # Assign red tones to loss categories
    for i, cat in enumerate(loss_categories):
        colors[cat] = red_tones[i % len(red_tones)]

    # Assign regular colors to non-loss categories
    for i, cat in enumerate(non_loss_categories):
        colors[cat] = base_colors[i % len(base_colors)]

    # Create color list in the order of stack_categories
    color_list = [colors[cat] for cat in stack_categories]

    # Configure plot with transparent background
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Create MEKKO rectangles with rescaled percentages (all positive now)
    current_x = 0
    rectangles_info = []

    # Calculate max height for proper axis scaling
    max_height = 100  # Since all categories will sum to 100%

    for i, category in enumerate(categories):
        # Get width for this category
        bar_width = width_percentages[category]

        # Get height data for this category (rescaled - all positive now)
        category_heights = height_percentages.loc[category]

        # 🔥 NEW: Get original percentages with signs for labels
        category_heights_original = height_percentages_original.loc[category]

        # 🔥 MODIFICACIÓN: Calculate visual totals for this category (for consistent percentage calculation)
        visual_category_total = sum(np.abs(category_heights.values))

        # Draw segments stacked from bottom to top
        current_y = 0
        for j, (stack_cat, height) in enumerate(category_heights.items()):
            if height > 0:  # Only draw if height is positive
                # Get color for this stack category
                color = color_list[j]

                # Create rectangle (using rescaled height for visual consistency)
                rect = Rectangle((current_x, current_y), bar_width, height,
                               facecolor=color, edgecolor='white', linewidth=1)
                ax.add_patch(rect)

                # Store rectangle info
                rectangles_info.append({
                    'category': category,
                    'stack_category': stack_cat,
                    'x': current_x,
                    'y': current_y,
                    'width': bar_width,
                    'height': height,
                    'color': color,
                    'type': 'positive'
                })

                # 🔥 MAIN CHANGE: Show percentages relative to visual height instead of original values
                min_height_for_label = 5  # Minimum 5% to show label
                min_width_for_label = 8   # Minimum 8% width to show label

                if height >= min_height_for_label and bar_width >= min_width_for_label:
                    label_x = current_x + bar_width/2
                    label_y = current_y + height/2

                    # 🔥 CAMBIO PRINCIPAL: Calculate percentage relative to VISUAL height (rescaled values)
                    visual_percentage = (height / visual_category_total * 100) if visual_category_total != 0 else 0

                    # Get original percentage for color determination
                    original_percentage = category_heights_original[stack_cat]

                    # Choose text color based on original value sign (for visual distinction)
                    text_color = '#FF6B6B' if original_percentage < 0 else 'white'  # Rojo para negativos, blanco para positivos

                    # Format percentage (always positive now since it's relative to visual height)
                    percentage_text = f'{visual_percentage:.0f}%'

                    ax.text(label_x, label_y, percentage_text,
                           ha='center', va='center', color=text_color,
                           fontweight='normal', fontsize=10)

                current_y += height  # Use rescaled height for positioning

        current_x += bar_width

    # Calculate positioning for labels
    label_padding = 3
    category_label_y = -label_padding

    total_label_padding = 3
    total_label_y = max_height + total_label_padding

    # 🔥 MEJORA 2: Add labels with IMPROVED total formatting using standalone functions
    current_x = 0
    for i, category in enumerate(categories):
        bar_width = width_percentages[category]
        label_x = current_x + bar_width/2

        # Add category label
        ax.text(label_x, category_label_y, formatted_categories[i],
               ha='center', va='top', rotation=rotation, fontweight='normal')

        # 🔥 NUEVO FORMATO MEJORADO para etiquetas totales usando funciones standalone
        original_total = category_totals_original[category]
        if original_total != 0:
            # Use standalone functions for consistent formatting
            total_scale, total_suffix = _get_custom_scale_and_suffix_standalone([abs(original_total)])
            formatted_total = _format_oval_value_standalone(abs(original_total), total_scale, total_suffix, 'sum')

            # Add sign prefix for negative totals
            if original_total < 0:
                formatted_total = f"-{formatted_total}"

            ax.text(label_x, total_label_y, formatted_total,
                   ha='center', va='bottom', color='black',
                   fontweight='bold', fontsize=11)

        current_x += bar_width

    # Generate Y-axis label for MEKKO chart
    label, unit, year = generate_intelligent_ylabel(numeric_column)

    # For MEKKO charts, add "Distribution"
    ylabel_base = f'{label} Distribution'

    if year:
        ylabel = f'{ylabel_base} ({year}, %)'
    else:
        ylabel = f'{ylabel_base} (%)'

    # Generate legend title
    legend_title = str(stack_column).replace('_', ' ').title()

    # Configure axes with reduced spacing
    ax.set_xlim(0, 100)

    # Set Y-axis limits with reduced spacing for closer labels
    y_extra_padding = 2
    y_min = category_label_y - y_extra_padding
    y_max = total_label_y

    ax.set_ylim(y_min, y_max)

    # Remove X-axis title and configure axes
    ax.set_xlabel("", fontweight='normal')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')

    # Remove ticks and configure spines for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_list[i],
                                   label=str(stack_cat).capitalize())
                      for i, stack_cat in enumerate(stack_categories)]

    if horizontal_params or vertical_params:
        legend = ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.35, 0.5),
                          frameon=False, fontsize=10, title=legend_title)
    else:
        legend = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                          frameon=False, fontsize=10, title=legend_title)

    # Style the legend title
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(10)

    # Add light grid
    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # ================================
    # 🔥 MEJORA 3+4+6+8: ENHANCED HORIZONTAL OVAL METRICS with DYNAMIC positioning, WIDER ovals and STANDALONE functions
    # ================================
    if horizontal_params:
        oval_data = []
        for est, col in horizontal_params.items():
            df_for_this_metric = df_base.copy()
            if col in df.columns:
                df_for_this_metric[col] = df.loc[df_base.index, col]
                df_for_this_metric = df_for_this_metric.dropna(subset=[col])

            if not df_for_this_metric.empty:
                category_metrics = df_for_this_metric.groupby(categorical_column)[col].agg(AVAILABLE_ESTIMATORS[est])
                category_metrics = category_metrics.reindex(categories, fill_value=0)
            else:
                category_metrics = pd.Series(0, index=categories)

            # 🔥 MEJORA 6: Add horizontal insights to JSON using STANDALONE functions
            for category, value in category_metrics.items():
                insight_key = f"{est.title()} {col} for {str(category).title()}"
                insights_JSON[insight_key] = value

            # 🔥 MEJORA 6: Use improved formatting system with standalone functions
            col_values = category_metrics.values
            col_scale, col_suffix = _get_custom_scale_and_suffix_standalone(col_values)

            oval_data.append({
                'estimator': est,
                'column': col,
                'values': category_metrics.values,
                'scale': col_scale,
                'suffix': col_suffix,
                'formatted_values': [_format_oval_value_standalone(v, col_scale, col_suffix, est) for v in category_metrics.values],
                'title': _create_oval_title_standalone(est, col)
            })

        # 🔥 MEJORA 3+7: Position ovals above the total value labels with DYNAMIC spacing
        chart_height = max_height  # 100 for MEKKO
        dynamic_offset = 15  # Dynamic offset above total labels
        total_label_space = total_label_padding + 5  # Space for total labels + margin

        oval_base_y = total_label_y + dynamic_offset
        oval_height = 8
        oval_spacing = 12

        for row_idx, oval_info in enumerate(oval_data):
            oval_y = oval_base_y + (row_idx * oval_spacing)

            # 🔥 MEJORA 6: Use custom title from processed data with standalone functions
            title = oval_info['title']
            ax.text(-8, oval_y, title, ha='right', va='center',
                   fontsize=10, fontweight='normal', color='black')

            current_x = 0
            for i, (category, value) in enumerate(zip(categories, oval_info['values'])):
                bar_width = width_percentages[category]
                oval_x = current_x + bar_width/2

                # Get formatted value using standalone function
                formatted_value = oval_info['formatted_values'][i]

                # 🔥 MEJORA 4: WIDER oval to match vertical ovals (adaptive width)
                oval_width = min(bar_width * 0.8, 10)  # Adaptive but wider
                oval = Ellipse((oval_x, oval_y), oval_width, oval_height,
                             facecolor='lightgray', edgecolor='gray',
                             linewidth=0.5, alpha=0.8)
                ax.add_patch(oval)

                # 🔥 MEJORA 6+8: Use formatted value with count handling
                ax.text(oval_x, oval_y, formatted_value,
                       ha='center', va='center', color='black',
                       fontweight='bold', fontsize=9)

                current_x += bar_width

        # 🔥 MEJORA 7: Adjust ylim to accommodate ovals with improved calculations
        if oval_data:
            top_oval_y = oval_base_y + ((len(oval_data) - 1) * oval_spacing) + oval_height/2
            ax.set_ylim(y_min, top_oval_y + 5)

    # ================================
    # 🔥 MEJORA 6+7+8: ENHANCED VERTICAL OVAL METRICS with STANDALONE functions and improved positioning
    # ================================
    if vertical_params:
        vertical_oval_data = []
        for est, col in vertical_params.items():
            df_for_this_metric = df_base.copy()
            if col in df.columns:
                df_for_this_metric[col] = df.loc[df_base.index, col]
                df_for_this_metric = df_for_this_metric.dropna(subset=[col])

            if not df_for_this_metric.empty:
                stack_metrics = df_for_this_metric.groupby(stack_column)[col].agg(AVAILABLE_ESTIMATORS[est])
                stack_metrics = stack_metrics.reindex(stack_categories, fill_value=0)
            else:
                stack_metrics = pd.Series(0, index=stack_categories)

            # 🔥 MEJORA 6: Add vertical insights to JSON using STANDALONE functions
            for stack_category, value in stack_metrics.items():
                insight_key = f"{est.title()} {col} for {str(stack_category).title()}"
                insights_JSON[insight_key] = value

            # 🔥 MEJORA 6: Use improved formatting system with standalone functions
            col_values = stack_metrics.values
            col_scale, col_suffix = _get_custom_scale_and_suffix_standalone(col_values)

            vertical_oval_data.append({
                'estimator': est,
                'column': col,
                'values': stack_metrics.values,
                'scale': col_scale,
                'suffix': col_suffix,
                'formatted_values': [_format_oval_value_standalone(v, col_scale, col_suffix, est) for v in stack_metrics.values],  # 🔥 MODIFICACIÓN: Pass estimator for count handling
                'title': _create_oval_title_standalone(est, col)
            })

        # 🔥 MEJORA 7: Position vertical ovals with improved calculations
        vertical_oval_base_x = 108
        vertical_oval_width = 6
        vertical_oval_spacing = 8

        chart_center = (y_max + y_min) / 2
        vertical_oval_height = 8
        stack_spacing = 15

        total_stack_height = (len(stack_categories) - 1) * stack_spacing
        vertical_oval_start_y = chart_center - (total_stack_height / 2)

        for col_idx, oval_info in enumerate(vertical_oval_data):
            oval_x = vertical_oval_base_x + (col_idx * vertical_oval_spacing)

            # 🔥 MEJORA 6: Use custom title from processed data with standalone functions
            title = oval_info['title']
            title_y = total_label_y + 25

            max_chars_per_line = 10
            words = title.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= max_chars_per_line or not current_line:
                    current_line.append(word)
                    current_length += len(word) + (1 if current_line else 0)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                lines.append(' '.join(current_line))

            title_text = '\n'.join(lines)
            ax.text(oval_x, title_y, title_text, ha='center', va='bottom',
                   fontsize=10, fontweight='normal', color='black',
                   rotation=0, multialignment='center')

            for i, (stack_cat, value) in enumerate(zip(stack_categories, oval_info['values'])):
                oval_y = vertical_oval_start_y + (i * stack_spacing)

                # Get formatted value using standalone function
                formatted_value = oval_info['formatted_values'][i]

                oval = Ellipse((oval_x, oval_y), vertical_oval_width, vertical_oval_height,
                             facecolor=color_list[i], edgecolor='white',
                             linewidth=1, alpha=0.9)
                ax.add_patch(oval)

                # 🔥 MEJORA 6+8: Use formatted value with count handling
                ax.text(oval_x, oval_y, formatted_value,
                       ha='center', va='center', color='white',
                       fontweight='bold', fontsize=9)

        # 🔥 MEJORA 7: Adjust xlim with improved calculations
        if vertical_params:
            rightmost_oval_x = vertical_oval_base_x + ((len(vertical_oval_data) - 1) * vertical_oval_spacing) + vertical_oval_width/2
            ax.set_xlim(0, rightmost_oval_x * 1.05)

    plt.tight_layout()

    return ax, insights_JSON