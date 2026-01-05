import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.ticker import FuncFormatter
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

def get_scale_and_format_eur(values, column_name):
    """
    Determine scale and format for values based on column name.
    This function is used across multiple chart types for consistent scaling.

    Parameters:
    -----------
    values : array-like, Series, or DataFrame
        Values to analyze for scaling
    column_name : str
        Column name to determine formatting

    Returns:
    --------
    tuple: (scale_factor, suffix)
    """
    try:
        column_upper = column_name.upper()

        # Check if it's a percentage metric (Growth, CAGR, Margin)
        is_percentage = ('GROWTH' in column_upper or 'CAGR' in column_upper or
                        'MARGIN' in column_upper)

        # For percentage metrics, no scaling needed
        if is_percentage:
            return 1, ''
        elif 'EUR' in column_upper:
            # Handle different input types
            if hasattr(values, 'values'):  # DataFrame or Series
                values_array = values.values.flatten()
            elif isinstance(values, (list, tuple)):
                values_array = np.array(values)
            else:
                values_array = np.asarray(values).flatten()

            # Filter out NaN and infinite values
            valid_values = []
            for v in values_array:
                try:
                    if np.isfinite(v) and not np.isnan(v):
                        valid_values.append(abs(float(v)))
                except (TypeError, ValueError):
                    continue

            # If no valid values, return default scaling
            if not valid_values:
                return 1, ''

            abs_max = max(valid_values)

            if abs_max >= 1e9:
                scale = 1e9
                suffix = 'Bn'
            elif abs_max >= 1e6:
                scale = 1e6
                suffix = 'Mn'
            elif abs_max >= 1e3:
                scale = 1e3
                suffix = 'K'
            else:
                scale = 1
                suffix = ''
            return scale, suffix
        return 1, ''

    except Exception as e:
        # If anything goes wrong, return safe defaults
        print(f"Warning: Error in scaling function: {e}. Using default scaling.")
        return 1, ''


def format_eur_axis(ax, x_min, x_max, n_ticks=6):
    """
    Format EUR axis with custom notation and exactly n_ticks points.
    Handles negative values correctly.
    """
    def eur_formatter(x, pos):
        # Handle the sign separately
        sign = '-' if x < 0 else ''
        abs_x = abs(x)

        if abs_x >= 1e9:
            return f'{sign} {abs_x/1e9:.1f} Bn €'
        elif abs_x >= 1e6:
            return f'{sign} {abs_x/1e6:.1f} Mn €'
        elif abs_x >= 1e3:
            return f'{sign} {abs_x/1e3:.1f} K €'
        else:
            return f'{sign} {abs_x:.0f}€'

    # Set custom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(eur_formatter))

    # Create exactly n_ticks evenly spaced ticks
    ticks = np.linspace(x_min, x_max, n_ticks)
    ax.set_xticks(ticks)

    return ax


def format_regular_axis(ax, x_min, x_max, n_ticks=6):
    """
    Format regular numeric axis with exactly n_ticks points.
    Handles negative values correctly.
    """
    ticks = np.linspace(x_min, x_max, n_ticks)
    ax.set_xticks(ticks)

    # Format large numbers nicely (handles negative values)
    def regular_formatter(x, pos):
        # Handle the sign separately
        sign = '-' if x < 0 else ''
        abs_x = abs(x)

        if abs_x >= 1e9:
            return f'{sign}{abs_x/1e9:.1f} B'
        elif abs_x >= 1e6:
            return f'{sign}{abs_x/1e6:.1f} M'
        elif abs_x >= 1e3:
            return f'{sign}{abs_x/1e3:.1f} K'
        else:
            return f'{sign}{abs_x:.0f}'

    ax.xaxis.set_major_formatter(FuncFormatter(regular_formatter))
    return ax


def format_category_name(category_name, max_length=10):
    """
    Format category names, breaking them into multiple lines
    if they exceed max_length characters.
    """
    category_str = str(category_name).lower().capitalize()

    if len(category_str) <= max_length:
        return category_str

    # Find the best place to break the string
    words = category_str.split()
    if len(words) == 1:
        # Single long word - break at max_length
        return category_str[:max_length] + '\n' + category_str[max_length:]
    else:
        # Multiple words - try to break at word boundaries
        first_line = ""
        second_line = ""

        for word in words:
            if len(first_line + word) <= max_length and not second_line:
                if first_line:
                    first_line += " " + word
                else:
                    first_line = word
            else:
                if second_line:
                    second_line += " " + word
                else:
                    second_line = word

        return first_line + '\n' + second_line if second_line else first_line


def format_value_consistent(value, scale, suffix, column_name, current_estimator='sum'):
    """
    Format numbers with consistent scale based on column name and estimator.
    Used across multiple chart types for consistent value display.
    """
    column_upper = column_name.upper()

    # Check if it's a percentage metric (Growth, CAGR, Margin)
    is_percentage = ('GROWTH' in column_upper or 'CAGR' in column_upper or
                    'MARGIN' in column_upper)

    # For count estimator, don't show € symbol
    if current_estimator == 'count':
        return f'{value:.0f}'
    elif is_percentage:
        return f'{value:.1f}%'
    elif 'EUR' in column_upper:
        scaled_value = value / scale
        if suffix:
            return f'€{scaled_value:.1f} {suffix}'
        else:
            return f'€{scaled_value:.1f}'
    else:
        return f'{value:.1f}'


def generate_intelligent_ylabel(column_name, estimator='sum'):
    """
    Generate intelligent Y-axis labels based on column patterns and estimator.
    Used across bar charts and stacked charts for consistent labeling.
    """
    column_upper = column_name.upper()

    # Extract year(s) - handle CAGR and Growth cases with two years
    if 'CAGR' in column_upper or 'GROWTH' in column_upper:
        year_match = re.search(r'(20\d{2})-(20\d{2})', column_name)
        if year_match:
            year = f'{year_match.group(1)}-{year_match.group(2)}'
        else:
            year_match = re.search(r'20\d{2}', column_name)
            year = year_match.group() if year_match else ''
    else:
        year_match = re.search(r'20\d{2}', column_name)
        year = year_match.group() if year_match else ''

    # Determine KPI and base unit (order matters - most specific first)
    if 'CAGR' in column_upper:
        if 'REVENUE' in column_upper and 'PER EMPLOYEE' in column_upper:
            kpi = 'CAGR Revenue per employee'
        elif 'REVENUE' in column_upper:
            kpi = 'CAGR Revenue'
        elif 'EBITDA' in column_upper and 'PER EMPLOYEE' in column_upper:
            kpi = 'CAGR EBITDA per employee'
        elif 'EBITDA' in column_upper:
            kpi = 'CAGR EBITDA'
        elif 'NET MARGIN' in column_upper:
            kpi = 'CAGR Net Margin'
        elif 'NET INCOME' in column_upper:
            kpi = 'CAGR Net Income'
        else:
            kpi = 'CAGR'
        unit = '%'
    elif 'GROWTH' in column_upper:
        if 'REVENUE' in column_upper and 'PER EMPLOYEE' in column_upper:
            kpi = 'Revenue per employee Growth'
        elif 'REVENUE' in column_upper:
            kpi = 'Revenue Growth'
        elif 'EBITDA' in column_upper and 'MARGIN' in column_upper:
            kpi = 'EBITDA Margin Growth'
        elif 'EBITDA' in column_upper and 'PER EMPLOYEE' in column_upper:
            kpi = 'EBITDA per employee Growth'
        elif 'EBITDA' in column_upper:
            kpi = 'EBITDA Growth'
        elif 'NET MARGIN' in column_upper:
            kpi = 'Net Margin Growth'
        elif 'NET INCOME' in column_upper:
            kpi = 'Net Income Growth'
        elif 'EMPLOYEES' in column_upper:
            kpi = 'Employees Growth'
        else:
            kpi = 'Growth'
        unit = '%'
    elif 'REVENUE' in column_upper:
        if 'PER EMPLOYEE' in column_upper:
            kpi = 'Revenue per employee'
            unit = '€ / employee' if estimator != 'count' else '#'
        elif 'EUR' in column_upper:
            kpi = 'Revenue'
            # Get scale for unit determination
            unit = '€'  # Will be modified with scale suffix later
        else:
            kpi = 'Revenue'
            unit = '€'
    elif 'EBITDA' in column_upper:
        if 'MARGIN' in column_upper:
            kpi = 'EBITDA Margin'
            unit = '%' if estimator != 'count' else '#'
        elif 'PER EMPLOYEE' in column_upper:
            kpi = 'EBITDA per employee'
            unit = '€ / employee' if estimator != 'count' else '#'
        elif 'EUR' in column_upper:
            kpi = 'EBITDA'
            unit = '€'  # Will be modified with scale suffix later
        else:
            kpi = 'EBITDA'
            unit = '€'
    elif 'NET INCOME' in column_upper:
        kpi = 'Net Income'
        if 'EUR' in column_upper:
            unit = '€'  # Will be modified with scale suffix later
        else:
            unit = '#'
    elif 'NET MARGIN' in column_upper:
        kpi = 'Net Margin'
        unit = '%' if estimator != 'count' else '#'
    elif 'NUMBER OF EMPLOYEES' in column_upper:
        kpi = 'Number of employees'
        unit = '# employees' if estimator != 'count' else '#'
    else:
        kpi = column_name.replace('_', ' ').title()
        unit = '#'

    # Determine estimator name
    estimator_names = {
        'mean': 'Average',
        'sum': '',  # No prefix for sum
        'median': 'Median',
        'std': 'Std Dev',
        'var': 'Variance',
        'min': 'Minimum',
        'max': 'Maximum',
        'count': 'Count'
    }

    # Special handling for different estimators
    if estimator == 'sum':
        estimator_prefix = ''
    elif estimator == 'count':
        if 'Revenue' in kpi or 'EBITDA' in kpi or 'Net Income' in kpi:
            estimator_prefix = 'Firm Count'
            kpi = ''
            unit = '# firms'
        else:
            estimator_prefix = f'{kpi} Count'
            kpi = ''
            unit = '# firms'
    else:
        estimator_prefix = estimator_names[estimator]

    # Construct the label
    parts = []
    if estimator_prefix:
        parts.append(estimator_prefix)
    if kpi:
        parts.append(kpi)

    label = ' '.join(parts)

    return label, unit, year


def setup_chart_style(ax, transparent_bg=True):
    """
    Apply consistent styling to chart axes.
    Used across all chart types for consistent appearance.
    """
    if transparent_bg:
        ax.patch.set_alpha(0)

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Style remaining spines
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('black')

    # Remove Y-axis ticks for cleaner look
    ax.set_yticks([])

    return ax


def save_plot_with_transparency(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Save plot with transparent background.
    """
    fig.savefig(
        filename,
        transparent=True,
        facecolor='none',
        edgecolor='none',
        dpi=dpi,
        bbox_inches=bbox_inches
    )


def create_custom_color_palette(n_colors, special_categories=None):
    """
    Create custom color palette with special handling for certain categories.

    Parameters:
    -----------
    n_colors : int
        Number of colors needed
    special_categories : list, optional
        List of category names that should get special (red) colors

    Returns:
    --------
    list: List of colors
    """
    base_colors = ['#7D4BEB', '#948f9c', '#f54298', '#330599', '#331f5e', '#F1D454', '#F5A25C', '#EB6B63']
    red_tones = ['#DC143C', '#B22222', '#8B0000', '#CD5C5C']

    if special_categories:
        colors = []
        special_count = 0
        regular_count = 0

        for i in range(n_colors):
            if i < len(special_categories) and any('loss' in str(cat).lower() for cat in special_categories):
                colors.append(red_tones[special_count % len(red_tones)])
                special_count += 1
            else:
                colors.append(base_colors[regular_count % len(base_colors)])
                regular_count += 1
        return colors
    else:
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]


def apply_outlier_filtering(df, columns, percentile_range):
    """
    Apply percentile-based outlier filtering to specified columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of column names to filter
    percentile_range : tuple
        (lower_percentile, upper_percentile) for filtering

    Returns:
    --------
    pandas.DataFrame: Filtered dataframe
    """
    if percentile_range is None:
        return df

    filtered_df = df.copy()
    lower_pct, upper_pct = percentile_range

    for column in columns:
        if column in filtered_df.columns:
            values = filtered_df[column].replace([np.inf, -np.inf], np.nan)
            valid_values = values.dropna()

            if len(valid_values) > 0:
                lower_bound = valid_values.quantile(lower_pct)
                upper_bound = valid_values.quantile(upper_pct)

                mask = (values >= lower_bound) & (values <= upper_bound)
                filtered_df.loc[~mask, column] = np.nan

    return filtered_df


def interpolate_missing_values(df_subset):
    """
    Fill missing values in time series rows using linear regression interpolation.
    Only interpolates rows that have at least 2 non-null values.
    """
    df_filled = df_subset.copy()

    # Identify numeric columns (time series columns)
    numeric_cols = df_subset.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return df_filled

    # Create year array for regression
    years = []
    for col in numeric_cols:
        year_match = re.search(r'\b(\d{4})\b', str(col))
        if year_match:
            years.append(int(year_match.group(1)))
        else:
            years.append(numeric_cols.index(col))

    years = np.array(years)

    # Process each row
    for idx, row in df_subset.iterrows():
        values = row[numeric_cols].values

        non_null_mask = ~pd.isna(values)
        non_null_count = np.sum(non_null_mask)

        if non_null_count >= 2 and non_null_count < len(values):
            try:
                train_years = years[non_null_mask].reshape(-1, 1)
                train_values = values[non_null_mask]

                lr = LinearRegression()
                lr.fit(train_years, train_values)

                missing_mask = pd.isna(values)
                if np.any(missing_mask):
                    missing_years = years[missing_mask].reshape(-1, 1)
                    predicted_values = lr.predict(missing_years)

                    missing_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if missing_mask[i]]
                    for i, col in enumerate(missing_cols):
                        df_filled.loc[idx, col] = predicted_values[i]

            except Exception:
                continue

    return df_filled