import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
from typing import Optional, Dict, Tuple, Union

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
    format_value_consistent,
    generate_intelligent_ylabel,
    setup_chart_style,
    create_custom_color_palette,
    apply_outlier_filtering,
    interpolate_missing_values
)

class FinancialTimeSeriesPlotter:
    """
    Enhanced time series plotter for financial data with dynamic metric detection and insights
    """

    def __init__(self, df, percentile_range=(0.005, 0.90)):
        """
        Initialize with financial dataset

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing financial data
        percentile_range : tuple or None
            Tuple of (lower_percentile, upper_percentile) to filter outliers.
            Example: (0.005, 0.90) removes bottom 0.5% and top 10% outliers.
            Set to None to disable outlier filtering.
        """
        self.df = df.copy()
        self.percentile_range = percentile_range
        self.metric_mappings = self._create_metric_mappings()

    def _create_metric_mappings(self):
        """
        Create mappings between metrics and their year columns dynamically

        Returns:
        --------
        dict: {metric_name: {year: column_name, ...}, ...}
        """
        # Define the base metric patterns (semi-static mapping)
        base_patterns = {
            # Basic financial metrics
            'Revenue EUR': 'Revenue EUR',
            'EBITDA EUR': 'EBITDA EUR',
            'Net Income EUR': 'Net Income EUR',

            # Margin metrics
            'EBITDA Margin': 'EBITDA Margin',
            'Net Margin': 'Net Margin',

            # Employee metrics
            'Number of employees': 'Number of employees',
            'Revenue per employee': 'Revenue per employee',
            'EBITDA per employee': 'EBITDA per employee',

            # Growth metrics (year-over-year)
            'Revenue EUR Growth': 'Revenue EUR Growth',
            'EBITDA EUR Growth': 'EBITDA EUR Growth',
            'Net Income EUR Growth': 'Net Income EUR Growth',
            'EBITDA Margin Growth': 'EBITDA Margin Growth',
            'Net Margin Growth': 'Net Margin Growth',
            'Number of employees Growth': 'Number of employees Growth',
            'Revenue per employee Growth': 'Revenue per employee Growth',
            'EBITDA per employee Growth': 'EBITDA per employee Growth',

            # CAGR metrics (compound annual growth rate)
            'CAGR Revenue EUR': 'CAGR Revenue EUR',
            'CAGR EBITDA EUR': 'CAGR EBITDA EUR',
            'CAGR Net Margin': 'CAGR Net Margin',
            'CAGR Revenue per employee': 'CAGR Revenue per employee'
        }

        mappings = {}

        for metric_name, base_pattern in base_patterns.items():
            # Find all columns that match this pattern
            matching_columns = self._find_matching_columns(base_pattern)

            if matching_columns:
                # Extract years from matching columns and create mapping
                year_mapping = {}
                for column in matching_columns:
                    year = self._extract_year_from_column(column, base_pattern)
                    if year is not None:
                        year_mapping[year] = column

                # Only add to mappings if we found at least one valid year
                if year_mapping:
                    mappings[metric_name] = year_mapping

        return mappings

    def _find_matching_columns(self, base_pattern):
        """Find all columns that match the base pattern"""
        matching_columns = []

        for column in self.df.columns:
            column_str = str(column)

            if column_str.startswith(base_pattern):
                remaining = column_str[len(base_pattern):]

                if remaining:
                    if remaining.startswith(' '):
                        year_part = remaining[1:]

                        if self._is_valid_year_pattern(year_part):
                            matching_columns.append(column_str)
                elif base_pattern == column_str:
                    matching_columns.append(column_str)

        return matching_columns

    def _is_valid_year_pattern(self, year_part):
        """Check if the year part matches valid patterns (YYYY or YYYY-YYYY)"""
        single_year_pattern = r'^\d{4}$'
        growth_pattern = r'^\d{4}-\d{4}$'

        return bool(re.match(single_year_pattern, year_part) or
                   re.match(growth_pattern, year_part))

    def _extract_year_from_column(self, column, base_pattern):
        """Extract year from column name"""
        if column == base_pattern:
            return 'base'

        remaining = column[len(base_pattern):]

        if remaining.startswith(' '):
            year_part = remaining[1:]

            if re.match(r'^\d{4}$', year_part):
                return int(year_part)
            elif re.match(r'^\d{4}-\d{4}$', year_part):
                return year_part

        return None

    def _apply_metric_outlier_filtering(self, metric):
        """Apply percentile-based outlier filtering only to columns of the specified metric"""
        if self.percentile_range is None or metric not in self.metric_mappings:
            return self.df

        filtered_df = self.df.copy()
        metric_columns = list(self.metric_mappings[metric].values())

        # Use shared utility function
        filtered_df = apply_outlier_filtering(filtered_df, metric_columns, self.percentile_range)

        return filtered_df

    def _interpolate_missing_values(self, df_subset):
        """Fill missing values in time series rows using linear regression interpolation"""
        return interpolate_missing_values(df_subset)

    def get_available_metrics(self):
        """Get available metrics"""
        return list(self.metric_mappings.keys())

    def get_categorical_columns(self):
        """Get categorical columns"""
        categorical_cols = []
        for col in self.df.columns:
            if (self.df[col].dtype == 'object' or
                (self.df[col].nunique() < 20 and self.df[col].nunique() > 1)):
                categorical_cols.append(col)
        return sorted(categorical_cols)

    def print_mappings(self):
        """Utility function to see the created mappings"""
        print("=== METRIC MAPPINGS ===")
        for metric, year_mapping in self.metric_mappings.items():
            print(f"\n{metric}:")
            for year, column in sorted(year_mapping.items()):
                print(f"  {year} -> {column}")
        print(f"\nTotal metrics found: {len(self.metric_mappings)}")

    def _prepare_data(self, metric):
        """Prepare time series data for plotting"""
        if metric not in self.metric_mappings:
            available = ', '.join(self.get_available_metrics())
            raise ValueError(f"Metric '{metric}' not available. Choose from: {available}")

        # Apply outlier filtering only for this specific metric
        filtered_df = self._apply_metric_outlier_filtering(metric)

        year_columns = self.metric_mappings[metric]
        data_points = []

        # Process each year independently
        for year, column in year_columns.items():
            if column in filtered_df.columns and year != 'base':
                year_data = filtered_df[column].replace([np.inf, -np.inf], np.nan)
                valid_values = year_data.dropna()

                if len(valid_values) > 0:
                    data_points.append({
                        'year': year,
                        'mean': valid_values.mean(),
                        'std': valid_values.std() if len(valid_values) > 1 else 0,
                        'count': len(valid_values)
                    })

        return pd.DataFrame(data_points).sort_values('year')

    def _get_complete_time_series(self, metric):
        """Get individual time series for rows that have data for ALL years"""
        if metric not in self.metric_mappings:
            return []

        filtered_df = self._apply_metric_outlier_filtering(metric)
        year_columns = self.metric_mappings[metric]

        available_columns = [col for year, col in year_columns.items()
                           if col in filtered_df.columns and year != 'base']

        if not available_columns:
            return []

        metric_df = filtered_df[available_columns].copy()
        metric_df = metric_df.replace([np.inf, -np.inf], np.nan)
        complete_rows = metric_df.dropna()

        if complete_rows.empty:
            return []

        complete_series = []
        years = np.array(sorted([year for year, col in year_columns.items()
                               if col in available_columns and year != 'base']))

        for idx, row in complete_rows.iterrows():
            values = []
            for year in years:
                col = year_columns[year]
                if col in available_columns:
                    values.append(row[col])

            if len(values) == len(years):
                complete_series.append((years, np.array(values)))

        return complete_series

    def _prepare_categorical_data(self, metric, categorical_column):
        """Prepare categorical time series data with missing value interpolation"""
        if metric not in self.metric_mappings:
            raise ValueError(f"Metric '{metric}' not available")

        if categorical_column not in self.df.columns:
            raise ValueError(f"Column '{categorical_column}' not found")

        filtered_df = self._apply_metric_outlier_filtering(metric)
        year_columns = self.metric_mappings[metric]

        year_specific_columns = [col for year, col in year_columns.items()
                               if year != 'base' and col in filtered_df.columns]

        if not year_specific_columns:
            return {}

        subset_columns = year_specific_columns + [categorical_column]
        df_subset = filtered_df[subset_columns].copy()
        df_interpolated = self._interpolate_missing_values(df_subset)
        categories = df_interpolated[categorical_column].dropna().unique()

        result = {}
        for category in categories:
            category_df = df_interpolated[df_interpolated[categorical_column] == category]
            data_points = []

            for year, column in year_columns.items():
                if column in category_df.columns and year != 'base':
                    year_data = category_df[column].replace([np.inf, -np.inf], np.nan)
                    valid_values = year_data.dropna()

                    if len(valid_values) > 0:
                        data_points.append({
                            'year': year,
                            'mean': valid_values.mean(),
                            'std': valid_values.std() if len(valid_values) > 1 else 0,
                            'count': len(valid_values)
                        })

            if data_points:
                result[str(category)] = pd.DataFrame(data_points).sort_values('year')

        return result

    def _get_complete_time_series_for_category(self, metric, categorical_column, category):
        """Get individual time series for rows in a specific category with data for ALL years"""
        if metric not in self.metric_mappings:
            return []

        filtered_df = self._apply_metric_outlier_filtering(metric)
        year_columns = self.metric_mappings[metric]

        year_specific_columns = [col for year, col in year_columns.items()
                               if year != 'base' and col in filtered_df.columns]

        if not year_specific_columns:
            return []

        subset_columns = year_specific_columns + [categorical_column]
        df_subset = filtered_df[subset_columns].copy()
        df_interpolated = self._interpolate_missing_values(df_subset)
        category_df = df_interpolated[df_interpolated[categorical_column] == category].copy()

        if category_df.empty:
            return []

        available_columns = [col for year, col in year_columns.items()
                           if col in category_df.columns and year != 'base']

        if not available_columns:
            return []

        metric_df = category_df[available_columns].copy()
        metric_df = metric_df.replace([np.inf, -np.inf], np.nan)
        complete_rows = metric_df.dropna()

        if complete_rows.empty:
            return []

        complete_series = []
        years = np.array(sorted([year for year, col in year_columns.items()
                               if col in available_columns and year != 'base']))

        for idx, row in complete_rows.iterrows():
            values = []
            for year in years:
                col = year_columns[year]
                if col in available_columns:
                    values.append(row[col])

            if len(values) == len(years):
                complete_series.append((years, np.array(values)))

        return complete_series

    def _calculate_cagr_with_custom_logic(self, value1, value2, diff):
        """
        Calculate CAGR using the custom logic provided
        """
        import numpy as np
        import pandas as pd

        def real_power(base, exponent):
            """Safe power calculation handling negative bases"""
            if base > 0:
                return base ** exponent
            elif base < 0:
                # For negative bases, only allow integer exponents in standard math
                # For fractional exponents, we'll use absolute value
                return -(abs(base) ** exponent)
            else:
                return 0

        # Check for null values
        if pd.isna(value1) or pd.isna(value2):
            return np.nan, 'null'

        # Check for division by zero
        if value2 == 0:
            return np.nan, 'zero_division'

        # Determine which formula to use based on signs
        value1_positive = value1 > 0
        value2_positive = value2 > 0

        # Case 1: Both values are positive - Standard formula
        if value1_positive and value2_positive:
            try:
                cagr = real_power(value1 / value2, 1/diff) - 1
                return cagr, 'standard'
            except (ValueError, ZeroDivisionError, OverflowError):
                return np.nan, 'error'

        # Case 2: Both values are negative - Special formula for both negative
        elif not value1_positive and not value2_positive:
            try:
                numerator = value1 - value2 + abs(value2)
                denominator = abs(value2)

                # Check for division by zero
                if denominator == 0:
                    return np.nan, 'zero_division'

                base_ratio = numerator / denominator
                cagr = real_power(base_ratio, 1/diff) - 1
                return cagr, 'both_negative'
            except (ValueError, ZeroDivisionError, OverflowError):
                return np.nan, 'error'

        # Case 3: Sign change (one positive, one negative) - NEW LOGIC
        else:
            try:
                # Identify the negative value
                negative_value = value1 if value1 < 0 else value2

                # Calculate the correction factor: double the absolute value of the negative value
                correction = 2 * abs(negative_value)

                # Apply correction to both values
                corrected_value1 = value1 + correction
                corrected_value2 = value2 + correction

                # Check for division by zero after correction
                if corrected_value2 == 0:
                    return np.nan, 'zero_division'

                # Apply standard formula with corrected values
                cagr = real_power(corrected_value1 / corrected_value2, 1/diff) - 1
                return cagr, 'sign_corrected'
            except (ValueError, ZeroDivisionError, OverflowError):
                return np.nan, 'error'

    def _calculate_insights(self, data, metric_name):
        """Calculate insights for time series data"""
        if data.empty or len(data) < 2:
            return {
                'metric': metric_name,
                'error': 'Insufficient data for insights calculation',
                'n_points': len(data) if not data.empty else 0
            }

        years = data['year'].values
        values = data['mean'].values

        insights = {
            'metric': metric_name,
            'n_points': len(data),
            'first_year': int(years[0]),
            'last_year': int(years[-1]),
            'first_value': values[0],
            'last_value': values[-1]
        }

        # Calculate trend slope using linear regression
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(years, values)
            insights.update({
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_direction': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Flat'
            })
        except Exception as e:
            insights.update({
                'trend_slope': np.nan,
                'trend_error': str(e)
            })

        # Calculate CAGR using custom logic
        diff = years[-1] - years[0]
        if diff > 0:
            cagr_value, cagr_method = self._calculate_cagr_with_custom_logic(values[-1], values[0], diff)
            insights.update({
                'cagr_value': cagr_value,
                'cagr_percentage': cagr_value * 100 if not pd.isna(cagr_value) else np.nan,
                'cagr_method': cagr_method
            })
        else:
            insights.update({
                'cagr_value': np.nan,
                'cagr_method': 'insufficient_timespan'
            })

        return insights

    def _calculate_categorical_insights(self, data_dict, metric_name):
        """Calculate insights for categorical data with comparisons"""
        insights = {
            'metric': metric_name,
            'categories': {},
            'comparisons': []
        }

        # Calculate insights for each category
        category_cagrs = {}
        for category, data in data_dict.items():
            cat_insights = self._calculate_insights(data, f"{metric_name} - {category}")
            insights['categories'][category] = cat_insights

            if not pd.isna(cat_insights.get('cagr_percentage')):
                category_cagrs[category] = cat_insights['cagr_percentage']

        # Generate comparisons between categories
        if len(category_cagrs) >= 2:
            # Sort categories by CAGR (descending)
            sorted_categories = sorted(category_cagrs.items(), key=lambda x: x[1], reverse=True)

            # Generate comparison statements
            for i, (cat1, cagr1) in enumerate(sorted_categories):
                for cat2, cagr2 in sorted_categories[i+1:]:
                    if cagr2 != 0:  # Avoid division by zero
                        if cagr2 > 0:
                            growth_ratio = (cagr1 - cagr2) / cagr2 * 100
                            comparison = f"{cat1} grows with a {cagr1:.1f}% CAGR, {growth_ratio:.0f}% faster than {cat2} with a growth of {cagr2:.1f}% CAGR"
                        else:
                            # When cagr2 is negative, comparison is more complex
                            comparison = f"{cat1} grows with a {cagr1:.1f}% CAGR, while {cat2} declines with {cagr2:.1f}% CAGR"
                    else:
                        comparison = f"{cat1} grows with a {cagr1:.1f}% CAGR, while {cat2} shows no growth (0.0% CAGR)"

                    insights['comparisons'].append(comparison)
                    # Only add the first comparison to avoid too many comparisons
                    break

        return insights

    def _calculate_subplot_layout(self, n_categories, custom_layout=None):
        """Calculate subplot layout based on number of categories"""
        if custom_layout is not None:
            if sum(custom_layout) != n_categories:
                raise ValueError(f"Custom layout sum ({sum(custom_layout)}) must equal number of categories ({n_categories})")

            n_rows = len(custom_layout)
            n_cols = max(custom_layout)
            return n_rows, n_cols, custom_layout

        # Default automatic layout
        if n_categories <= 3:
            return 1, n_categories, [n_categories]
        elif n_categories == 4:
            return 2, 2, [2, 2]
        elif n_categories <= 6:
            return 2, 3, [3, n_categories - 3]
        elif n_categories <= 9:
            return 3, 3, [3, 3, n_categories - 6]
        else:
            cols_per_row = int(np.ceil(np.sqrt(n_categories)))
            rows_needed = int(np.ceil(n_categories / cols_per_row))

            layout = []
            remaining = n_categories
            for i in range(rows_needed):
                if remaining >= cols_per_row:
                    layout.append(cols_per_row)
                    remaining -= cols_per_row
                else:
                    layout.append(remaining)
                    break

            return rows_needed, cols_per_row, layout

    def plot(self, metric, categorical_column=None, figsize=(14, 8),
             show_confidence=False, smooth_line=True, show_points=False,
             subplot_layout=None,openai=False):
        """
        Create time series plot and return insights

        Returns:
        --------
        tuple: (plot_axes, insights_json)
        """
        plt.style.use('default')

        if categorical_column is None:
            ax, insights = self._plot_single(metric, figsize, show_confidence, smooth_line, show_points,openai)
        else:
            ax, insights = self._plot_categorical(metric, categorical_column, figsize,
                                                show_confidence, smooth_line, show_points, subplot_layout,openai)

        return ax, insights

    def _plot_single(self, metric, figsize, show_confidence, smooth_line, show_points,openai=False):
        """Create single time series plot with mean line and individual complete series"""
        data = self._prepare_data(metric)

        if data.empty:
            print(f"No data available for '{metric}'")
            return None, {'error': f"No data available for '{metric}'"}

        # Calculate insights
        insights = self._calculate_insights(data, metric)
        if openai is True:
          insights = generate_strategic_insights(
              None, None,metric,insights,
              API_KEY=API_KEY
          )

        # Create figure with transparent background
        fig, ax = plt.subplots(figsize=figsize, facecolor='none')
        ax.patch.set_alpha(0)

        years = data['year'].values
        means = data['mean'].values

        # Get scaling for values
        scale, suffix = get_scale_and_format_eur(means, metric)
        scaled_means = means / scale

        # Define colors
        line_color = '#7D4BEB'
        individual_color = '#B19FEB'
        point_color = 'black'

        # Get complete individual time series
        complete_series = self._get_complete_time_series(metric)

        # Plot individual complete series first
        for series_years, series_values in complete_series:
            scaled_series = series_values / scale

            if smooth_line and len(series_years) > 1 and SCIPY_AVAILABLE:
                try:
                    if len(series_years) == 2:
                        years_smooth = np.linspace(series_years.min(), series_years.max(), 50)
                        values_smooth = np.interp(years_smooth, series_years, scaled_series)
                    else:
                        years_smooth = np.linspace(series_years.min(), series_years.max(), 100)
                        f = interpolate.interp1d(series_years, scaled_series, kind='cubic',
                                               bounds_error=False, fill_value="extrapolate")
                        values_smooth = f(years_smooth)
                    ax.plot(years_smooth, values_smooth, linewidth=0.5, alpha=0.6, color=individual_color)
                except Exception:
                    ax.plot(series_years, scaled_series, linewidth=0.5, alpha=0.6, color=individual_color)
            else:
                ax.plot(series_years, scaled_series, linewidth=0.5, alpha=0.6, color=individual_color)

        # Main mean line - on top
        if smooth_line and len(years) > 1 and SCIPY_AVAILABLE:
            try:
                if len(years) == 2:
                    years_smooth = np.linspace(years.min(), years.max(), 50)
                    means_smooth = np.interp(years_smooth, years, scaled_means)
                else:
                    years_smooth = np.linspace(years.min(), years.max(), 100)
                    f = interpolate.interp1d(years, scaled_means, kind='cubic',
                                           bounds_error=False, fill_value="extrapolate")
                    means_smooth = f(years_smooth)
                ax.plot(years_smooth, means_smooth, linewidth=3, alpha=0.9, color=line_color, zorder=10)
            except Exception:
                ax.plot(years, scaled_means, linewidth=3, alpha=0.9, color=line_color, zorder=10)
        else:
            ax.plot(years, scaled_means, linewidth=3, alpha=0.9, color=line_color, zorder=10)

        # Points - Only if explicitly specified
        if show_points:
            ax.scatter(years, scaled_means, s=120, alpha=0.8, zorder=15,
                      color=point_color, edgecolors='white', linewidth=2)

        # Legacy confidence bands - only if explicitly requested
        if show_confidence and len(data) > 1:
            scaled_std = data['std'].values / scale
            mask = scaled_std > 0
            if mask.any():
                lower = scaled_means - scaled_std
                upper = scaled_means + scaled_std
                ax.fill_between(years, lower, upper, alpha=0.3, color=individual_color)

        # Apply styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.grid(False)

        # Labels
        ylabel = f'{metric} ({suffix} €)' if suffix and 'EUR' in metric else metric
        if suffix == '':
            ylabel = metric

        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f'{metric} - Time Series', fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(years)
        ax.set_xticklabels([int(y) for y in years], fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=11)

        plt.tight_layout()
        plt.show()

        return ax, insights

    def _plot_categorical(self, metric, categorical_column, figsize,
                     show_confidence, smooth_line, show_points, subplot_layout=None,openai=False):
        """Create categorical time series plots with mean lines and individual complete series"""
        data_dict = self._prepare_categorical_data(metric, categorical_column)

        if not data_dict:
            error_msg = f"No data available for '{metric}' by '{categorical_column}'"
            print(error_msg)
            return None, {'error': error_msg}

        # Calculate insights for categorical data
        insights = self._calculate_categorical_insights(data_dict, metric)
        if openai is True:
          insights = generate_strategic_insights(
              categorical_column, None,metric,insights,
              API_KEY=API_KEY
          )

        n_categories = len(data_dict)
        n_rows, n_cols, layout_list = self._calculate_subplot_layout(n_categories, subplot_layout)

        # Create figure with transparent background
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor='none')

        # Handle subplot indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        colors = plt.cm.Set2(np.linspace(0, 1, n_categories))

        # Plot each category
        category_idx = 0
        for row_idx, cols_in_row in enumerate(layout_list):
            for col_idx in range(cols_in_row):
                if category_idx >= n_categories:
                    break

                category = list(data_dict.keys())[category_idx]
                data = data_dict[category]

                ax = axes[row_idx, col_idx]
                ax.patch.set_alpha(0)

                years = data['year'].values
                means = data['mean'].values

                # Independent scaling for each category
                scale, suffix = get_scale_and_format_eur(means, metric)
                scaled_means = means / scale

                # Color handling
                base_color = colors[category_idx]
                mean_color = tuple([max(0, c * 0.6) for c in base_color[:3]] + [0.9])
                individual_color = tuple(list(base_color[:3]) + [0.25])

                # Get complete individual time series for this category
                complete_series = self._get_complete_time_series_for_category(metric, categorical_column, category)

                # Plot individual series
                for series_years, series_values in complete_series:
                    scaled_series = series_values / scale

                    if smooth_line and len(series_years) > 1 and SCIPY_AVAILABLE:
                        try:
                            if len(series_years) == 2:
                                years_smooth = np.linspace(series_years.min(), series_years.max(), 50)
                                values_smooth = np.interp(years_smooth, series_years, scaled_series)
                            else:
                                years_smooth = np.linspace(series_years.min(), series_years.max(), 50)
                                f = interpolate.interp1d(series_years, scaled_series, kind='cubic',
                                                      bounds_error=False, fill_value="extrapolate")
                                values_smooth = f(years_smooth)
                            ax.plot(years_smooth, values_smooth, linewidth=0.5, color=individual_color)
                        except Exception:
                            ax.plot(series_years, scaled_series, linewidth=0.5, color=individual_color)
                    else:
                        ax.plot(series_years, scaled_series, linewidth=0.5, color=individual_color)

                # Main mean line
                if smooth_line and len(years) > 1 and SCIPY_AVAILABLE:
                    try:
                        if len(years) == 2:
                            years_smooth = np.linspace(years.min(), years.max(), 50)
                            means_smooth = np.interp(years_smooth, years, scaled_means)
                        else:
                            years_smooth = np.linspace(years.min(), years.max(), 50)
                            f = interpolate.interp1d(years, scaled_means, kind='cubic',
                                                  bounds_error=False, fill_value="extrapolate")
                            means_smooth = f(years_smooth)
                        ax.plot(years_smooth, means_smooth, linewidth=2.5, color=mean_color, zorder=10)
                    except Exception:
                        ax.plot(years, scaled_means, linewidth=2.5, color=mean_color, zorder=10)
                else:
                    ax.plot(years, scaled_means, linewidth=2.5, color=mean_color, zorder=10)

                # Points if requested
                if show_points:
                    ax.scatter(years, scaled_means, s=80, alpha=0.8, zorder=15,
                              color='black', edgecolors='white', linewidth=1)

                # Confidence bands if requested
                if show_confidence and len(data) > 1:
                    scaled_std = data['std'].values / scale
                    mask = scaled_std > 0
                    if mask.any():
                        lower = scaled_means - scaled_std
                        upper = scaled_means + scaled_std
                        ax.fill_between(years, lower, upper, alpha=0.25, color=base_color)

                # Styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('black')
                ax.spines['bottom'].set_color('black')
                ax.grid(False)

                # Labels
                ylabel = f'{metric} ({suffix} €)' if suffix and 'EUR' in metric else metric
                if suffix == '':
                    ylabel = metric

                ax.set_title(f'{category}', fontsize=12, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_xticks(years)
                ax.set_xticklabels([int(y) for y in years])
                ax.tick_params(axis='both', which='major', labelsize=9)

                category_idx += 1

        # Hide empty subplots
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx < len(layout_list) and col_idx >= layout_list[row_idx]:
                    axes[row_idx, col_idx].set_visible(False)
                elif row_idx >= len(layout_list):
                    axes[row_idx, col_idx].set_visible(False)

        plt.suptitle(f'{metric} by {categorical_column}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return axes, insights

def plot_timeseries(df, metric, categorical_column=None, figsize=(14, 8),
                   subplot_layout=None, show_points=False, percentile_range=(0.005, 0.90), **kwargs):
    """
    Quick function to create time series plots with insights

    Returns:
    --------
    tuple: (plot_axes, insights_json)
    """
    plotter = FinancialTimeSeriesPlotter(df, percentile_range=percentile_range)
    return plotter.plot(metric, categorical_column, figsize, subplot_layout=subplot_layout,
                       show_points=show_points, **kwargs)

def get_options(df, percentile_range=(0.005, 0.90)):
    """See available metrics and categorical columns"""
    plotter = FinancialTimeSeriesPlotter(df, percentile_range=percentile_range)
    return {
        'metrics': plotter.get_available_metrics(),
        'categorical_columns': plotter.get_categorical_columns()
    }

class StackedSumBarchart:
    """
    Class for processing financial time series data and converting it to long format
    for stacked bar charts with sum aggregation
    """

    def __init__(self, df, percentile_range=(0.005, 0.90)):
        """
        Initialize with financial dataset

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing financial data
        percentile_range : tuple or None
            Tuple of (lower_percentile, upper_percentile) to filter outliers.
            Example: (0.005, 0.90) removes bottom 0.5% and top 10% outliers.
            Set to None to disable outlier filtering.
        """
        self.df = df.copy()
        self.percentile_range = percentile_range
        self.metric_mappings = self._create_metric_mappings()

    def _create_metric_mappings(self):
        """Create mappings between metrics and their year columns dynamically"""
        # Define the base metric patterns (same as time series)
        base_patterns = {
            # Basic financial metrics
            'Revenue EUR': 'Revenue EUR',
            'EBITDA EUR': 'EBITDA EUR',
            'Net Income EUR': 'Net Income EUR',

            # Margin metrics
            'EBITDA Margin': 'EBITDA Margin',
            'Net Margin': 'Net Margin',

            # Employee metrics
            'Number of employees': 'Number of employees',
            'Revenue per employee': 'Revenue per employee',
            'EBITDA per employee': 'EBITDA per employee',

            # Growth metrics (year-over-year)
            'Revenue EUR Growth': 'Revenue EUR Growth',
            'EBITDA EUR Growth': 'EBITDA EUR Growth',
            'Net Income EUR Growth': 'Net Income EUR Growth',
            'EBITDA Margin Growth': 'EBITDA Margin Growth',
            'Net Margin Growth': 'Net Margin Growth',
            'Number of employees Growth': 'Number of employees Growth',
            'Revenue per employee Growth': 'Revenue per employee Growth',
            'EBITDA per employee Growth': 'EBITDA per employee Growth',

            # CAGR metrics (compound annual growth rate)
            'CAGR Revenue EUR': 'CAGR Revenue EUR',
            'CAGR EBITDA EUR': 'CAGR EBITDA EUR',
            'CAGR Net Margin': 'CAGR Net Margin',
            'CAGR Revenue per employee': 'CAGR Revenue per employee'
        }

        mappings = {}

        for metric_name, base_pattern in base_patterns.items():
            matching_columns = self._find_matching_columns(base_pattern)

            if matching_columns:
                year_mapping = {}
                for column in matching_columns:
                    year = self._extract_year_from_column(column, base_pattern)
                    if year is not None:
                        year_mapping[year] = column

                if year_mapping:
                    mappings[metric_name] = year_mapping

        return mappings

    def _find_matching_columns(self, base_pattern):
        """Find all columns that match the base pattern"""
        matching_columns = []

        for column in self.df.columns:
            column_str = str(column)

            if column_str.startswith(base_pattern):
                remaining = column_str[len(base_pattern):]

                if remaining:
                    if remaining.startswith(' '):
                        year_part = remaining[1:]
                        if self._is_valid_year_pattern(year_part):
                            matching_columns.append(column_str)
                elif base_pattern == column_str:
                    matching_columns.append(column_str)

        return matching_columns

    def _is_valid_year_pattern(self, year_part):
        """Check if the year part matches valid patterns (YYYY or YYYY-YYYY)"""
        single_year_pattern = r'^\d{4}$'
        growth_pattern = r'^\d{4}-\d{4}$'

        return bool(re.match(single_year_pattern, year_part) or
                   re.match(growth_pattern, year_part))

    def _extract_year_from_column(self, column, base_pattern):
        """Extract year from column name"""
        if column == base_pattern:
            return 'base'

        remaining = column[len(base_pattern):]

        if remaining.startswith(' '):
            year_part = remaining[1:]

            if re.match(r'^\d{4}$', year_part):
                return int(year_part)
            elif re.match(r'^\d{4}-\d{4}$', year_part):
                return year_part

        return None

    def _calculate_cagr_with_custom_logic(self, value1, value2, diff):
        """
        Calculate CAGR using the custom logic provided
        """
        import numpy as np
        import pandas as pd

        def real_power(base, exponent):
            """Safe power calculation handling negative bases"""
            if base > 0:
                return base ** exponent
            elif base < 0:
                # For negative bases, only allow integer exponents in standard math
                # For fractional exponents, we'll use absolute value
                return -(abs(base) ** exponent)
            else:
                return 0

        # Check for null values
        if pd.isna(value1) or pd.isna(value2):
            return np.nan, 'null'

        # Check for division by zero
        if value2 == 0:
            return np.nan, 'zero_division'

        # Determine which formula to use based on signs
        value1_positive = value1 > 0
        value2_positive = value2 > 0

        # Case 1: Both values are positive - Standard formula
        if value1_positive and value2_positive:
            try:
                cagr = real_power(value1 / value2, 1/diff) - 1
                return cagr, 'standard'
            except (ValueError, ZeroDivisionError, OverflowError):
                return np.nan, 'error'

        # Case 2: Both values are negative - Special formula for both negative
        elif not value1_positive and not value2_positive:
            try:
                numerator = value1 - value2 + abs(value2)
                denominator = abs(value2)

                # Check for division by zero
                if denominator == 0:
                    return np.nan, 'zero_division'

                base_ratio = numerator / denominator
                cagr = real_power(base_ratio, 1/diff) - 1
                return cagr, 'both_negative'
            except (ValueError, ZeroDivisionError, OverflowError):
                return np.nan, 'error'

        # Case 3: Sign change (one positive, one negative) - NEW LOGIC
        else:
            try:
                # Identify the negative value
                negative_value = value1 if value1 < 0 else value2

                # Calculate the correction factor: double the absolute value of the negative value
                correction = 2 * abs(negative_value)

                # Apply correction to both values
                corrected_value1 = value1 + correction
                corrected_value2 = value2 + correction

                # Check for division by zero after correction
                if corrected_value2 == 0:
                    return np.nan, 'zero_division'

                # Apply standard formula with corrected values
                cagr = real_power(corrected_value1 / corrected_value2, 1/diff) - 1
                return cagr, 'sign_corrected'
            except (ValueError, ZeroDivisionError, OverflowError):
                return np.nan, 'error'

    def _calculate_category_cagrs(self, pivot_data):
        """
        Calculate CAGR for each category in the pivot_data

        Parameters:
        -----------
        pivot_data : pandas.DataFrame
            DataFrame with Years as index and categories as columns

        Returns:
        --------
        dict
            Dictionary with category names as keys and CAGR info as values
            Format: {category: {'cagr': value, 'method': method, 'years_span': diff, 'value1': recent, 'value2': oldest}}
        """
        import pandas as pd
        import numpy as np

        cagr_results = {}

        for category in pivot_data.columns:
            category_data = pivot_data[category].copy()

            # Remove zero and NaN values
            non_zero_data = category_data[(category_data != 0) & (~pd.isna(category_data))]

            if len(non_zero_data) < 2:
                # Not enough data points for CAGR calculation
                cagr_results[category] = {
                    'cagr': np.nan,
                    'method': 'insufficient_data',
                    'years_span': 0,
                    'value1': np.nan,
                    'value2': np.nan
                }
                continue

            # Sort by year index to get oldest and most recent
            non_zero_data_sorted = non_zero_data.sort_index()

            # Get oldest and most recent values
            oldest_year = non_zero_data_sorted.index[0]
            recent_year = non_zero_data_sorted.index[-1]

            value2 = non_zero_data_sorted.iloc[0]  # Oldest (value2)
            value1 = non_zero_data_sorted.iloc[-1]  # Most recent (value1)

            # Calculate year difference
            diff = abs(recent_year - oldest_year)

            if diff == 0:
                # Same year, no growth calculation possible
                cagr_results[category] = {
                    'cagr': 0.0,
                    'method': 'same_year',
                    'years_span': 0,
                    'value1': value1,
                    'value2': value2
                }
                continue

            # Calculate CAGR using custom logic
            cagr_value, method = self._calculate_cagr_with_custom_logic(value1, value2, diff)

            cagr_results[category] = {
                'cagr': cagr_value,
                'method': method,
                'years_span': diff,
                'value1': value1,
                'value2': value2,
                'oldest_year': oldest_year,
                'recent_year': recent_year
            }

        return cagr_results

    def _format_cagr_value(self, cagr_value):
        """
        Format CAGR value for display in ovals

        Parameters:
        -----------
        cagr_value : float
            CAGR value (as decimal, e.g., 0.15 for 15%)

        Returns:
        --------
        str
            Formatted CAGR string (e.g., "15.3%" or "N/A")
        """
        import pandas as pd
        import numpy as np

        if pd.isna(cagr_value) or np.isinf(cagr_value):
            return "N/A"

        # Convert to percentage and format
        percentage = cagr_value * 100

        # Format with appropriate decimal places
        if abs(percentage) < 0.1:
            return f"{percentage:.2f}%"
        elif abs(percentage) < 10:
            return f"{percentage:.1f}%"
        else:
            return f"{percentage:.0f}%"

    def _apply_metric_outlier_filtering(self, metric):
        """Apply percentile-based outlier filtering only to columns of the specified metric"""
        if self.percentile_range is None or metric not in self.metric_mappings:
            return self.df

        metric_columns = list(self.metric_mappings[metric].values())
        existing_columns = [col for col in metric_columns if col in self.df.columns]

        # Use shared utility function
        return apply_outlier_filtering(self.df, existing_columns, self.percentile_range)

    def _interpolate_missing_values(self, df_subset):
        """Fill missing values using shared utility function"""
        return interpolate_missing_values(df_subset)

    def _validate_oval_params(self, horizontal_params, vertical_params):
        """Validate horizontal and vertical parameters for oval metrics"""
        # Validate horizontal_params structure and base patterns
        if horizontal_params:
            if not isinstance(horizontal_params, dict) or len(horizontal_params) > 2:
                raise ValueError("horizontal_params must be a dictionary with maximum 2 key-value pairs")

            for est, base_pattern in horizontal_params.items():
                if est not in AVAILABLE_ESTIMATORS:
                    raise ValueError(f"Invalid estimator '{est}' in horizontal_params. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

                if base_pattern not in self.metric_mappings:
                    available_patterns = list(self.metric_mappings.keys())
                    raise ValueError(f"Base pattern '{base_pattern}' not found in available metrics. Options: {available_patterns}")

        # Validate vertical_params structure and columns
        if vertical_params:
            if not isinstance(vertical_params, dict) or len(vertical_params) > 2:
                raise ValueError("vertical_params must be a dictionary with maximum 2 key-value pairs")

            for est, col in vertical_params.items():
                if est not in AVAILABLE_ESTIMATORS:
                    raise ValueError(f"Invalid estimator '{est}' in vertical_params. Options: {list(AVAILABLE_ESTIMATORS.keys())}")

                if col not in self.df.columns:
                    raise ValueError(f"Column '{col}' specified in vertical_params does not exist in DataFrame")

                # Validate that the column is numeric
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    raise ValueError(f"Column '{col}' in vertical_params must be numeric")

    def _calculate_vertical_params(self, filtered_df, vertical_params, categorical_column, final_categories=None):
        """
        Calculate vertical oval metrics from original data (before interpolation)
        Now filters by final_categories to only show ovals for categories that appear in the final chart.

        Parameters:
        -----------
        filtered_df : pandas.DataFrame
            Filtered dataframe before interpolation
        vertical_params : dict
            Vertical parameters configuration
        categorical_column : str
            Name of categorical column
        final_categories : list, optional
            List of categories that appear in the final chart. If provided, only these categories will be included in ovals.
        """
        if not vertical_params:
            return []

        vertical_data = []

        for est, col in vertical_params.items():
            # Create subset with the required columns
            required_columns = [col, categorical_column]
            df_subset = filtered_df[required_columns].dropna()

            if df_subset.empty:
                # If no data, create zero values for all categories
                unique_categories = filtered_df[categorical_column].dropna().unique()
                category_metrics = pd.Series(0, index=unique_categories)
            else:
                # Group by category and apply estimator
                category_metrics = df_subset.groupby(categorical_column)[col].agg(AVAILABLE_ESTIMATORS[est])

            if final_categories is not None:
                # Only keep categories that appear in the final chart
                available_categories = [cat for cat in final_categories if cat in category_metrics.index]
                if available_categories:
                    category_metrics = category_metrics.loc[available_categories]
                else:
                    # If no categories match, create empty series with final_categories as index
                    category_metrics = pd.Series(0, index=final_categories)

            # Get custom scale and suffix for oval formatting
            col_values = category_metrics.values
            col_scale, col_suffix = self._get_custom_scale_and_suffix(col_values)

            # Create custom title
            oval_title = self._create_oval_title(est, col)

            vertical_data.append({
                'estimator': est,
                'column': col,
                'title': oval_title,
                'values': category_metrics.values,
                'categories': category_metrics.index.tolist(),
                'scale': col_scale,
                'suffix': col_suffix,
                # 🔥 NUEVO: Pass estimator to formatting function
                'formatted_values': [self._format_oval_value(v, col_scale, col_suffix, est) for v in category_metrics.values]
            })

        return vertical_data

    def _calculate_horizontal_params(self, result_df, horizontal_params, metric):
        """
        Calculate horizontal oval metrics from processed data (after interpolation)
        Now includes custom titles, formatting, and estimator passing.
        """
        if not horizontal_params:
            return []

        horizontal_data = []

        for est, base_pattern in horizontal_params.items():
            # Group by Year and apply estimator to the current metric
            if result_df.empty:
                year_metrics = pd.Series(dtype=float)
            else:
                year_metrics = result_df.groupby('Year')[metric].agg(AVAILABLE_ESTIMATORS[est])

            # Get custom scale and suffix for oval formatting
            if not year_metrics.empty:
                year_values = year_metrics.values
                col_scale, col_suffix = self._get_custom_scale_and_suffix(year_values)
                # 🔥 NUEVO: Pass estimator to formatting function
                formatted_values = [self._format_oval_value(v, col_scale, col_suffix, est) for v in year_values]
            else:
                year_values = np.array([])
                col_scale, col_suffix = 1, ''
                formatted_values = []

            # Create custom title
            oval_title = self._create_oval_title(est, base_pattern)

            horizontal_data.append({
                'estimator': est,
                'base_pattern': base_pattern,
                'title': oval_title,
                'values': year_values,
                'years': year_metrics.index.tolist() if not year_metrics.empty else [],
                'scale': col_scale,
                'suffix': col_suffix,
                'formatted_values': formatted_values
            })

        return horizontal_data

    def _get_custom_scale_and_suffix(self, values):
        """
        Get scale and suffix for custom oval formatting.

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

    def _format_oval_value(self, value, scale, suffix, estimator='sum'):
        """
        Format a value for display in ovals with special handling for count estimator.

        Parameters:
        -----------
        value : float
            Value to format
        scale : float
            Scale factor (1, 1e3, 1e6, 1e9)
        suffix : str
            Suffix string ('', 'K', 'Mn', 'Bn')
        estimator : str, default 'sum'
            Statistical estimator used ('sum', 'mean', 'median', 'count', etc.)

        Returns:
        --------
        str
            Formatted value string

        Examples:
        ---------
        9340000, 1e6, 'Mn', 'sum' → '9.34Mn'
        4650, 1e3, 'K', 'mean' → '4.65K'
        1500, 1e3, 'K', 'count' → '1500' (always raw integer for count, no scaling)
        45.67, 1, '', 'sum' → '45.67'
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return '0'

        # 🔥 NUEVO: Special handling for count estimator - NEVER apply scaling, always show raw integers
        if estimator.lower() == 'count':
            return f"{int(value)}"  # Always show raw value as integer, no scaling or suffix

        # Original logic for other estimators
        scaled_value = value / scale

        if suffix == '':  # No suffix, format based on decimal places
            if scaled_value == int(scaled_value):
                return f"{int(scaled_value)}"
            else:
                return f"{scaled_value:.2f}"
        else:  # Has suffix, always use 2 decimal places for non-count
            return f"{scaled_value:.2f}{suffix}"

    def _create_oval_title(self, estimator, variable_name):
        """
        Create formatted title for ovals in the format: 'Estimator Variable Name'

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
        return f"{estimator.title()} {variable_name}"

    def get_available_metrics(self):
        """Get available metrics"""
        return list(self.metric_mappings.keys())

    def get_categorical_columns(self):
        """Get categorical columns"""
        categorical_cols = []
        for col in self.df.columns:
            if (self.df[col].dtype == 'object' or
                (self.df[col].nunique() < 20 and self.df[col].nunique() > 1)):
                categorical_cols.append(col)
        return sorted(categorical_cols)

    def process_data(self, metric, categorical_column=None, horizontal_params=None, vertical_params=None):
        """
        Process the financial data and return unpivoted/melted DataFrame along with oval metrics

        Parameters:
        -----------
        metric : str
            Metric to process (e.g., 'Revenue EUR', 'EBITDA EUR')
        categorical_column : str, optional
            Categorical column to include in the result
        horizontal_params : dict, optional
            Dictionary with up to 2 key-value pairs for horizontal oval metrics
            Format: {'estimator': 'base_pattern'}, e.g., {'mean': 'Revenue EUR'}
        vertical_params : dict, optional
            Dictionary with up to 2 key-value pairs for vertical oval metrics
            Format: {'estimator': 'column_name'}, e.g., {'sum': 'EBITDA EUR 2024'}

        Returns:
        --------
        tuple
            If horizontal_params or vertical_params provided: (pandas.DataFrame, horizontal_data, vertical_data)
            Otherwise: pandas.DataFrame

            - pandas.DataFrame: Unpivoted DataFrame with columns ['Year', metric_name, categorical_column (if provided)]
            - horizontal_data: List of dictionaries with horizontal oval metrics data
            - vertical_data: List of dictionaries with vertical oval metrics data
        """
        if metric not in self.metric_mappings:
            available = ', '.join(self.get_available_metrics())
            raise ValueError(f"Metric '{metric}' not available. Choose from: {available}")

        # Validate oval parameters if provided
        self._validate_oval_params(horizontal_params, vertical_params)

        # Step 1: Apply outlier filtering
        filtered_df = self._apply_metric_outlier_filtering(metric)

        # Step 2: Select relevant columns for main processing
        year_columns = self.metric_mappings[metric]

        # Get only the year-specific columns (exclude 'base' columns)
        year_specific_columns = [col for year, col in year_columns.items()
                               if year != 'base' and col in filtered_df.columns]

        if not year_specific_columns:
            raise ValueError(f"No year-specific columns found for metric '{metric}'")

        # Include categorical column if specified
        columns_to_include = year_specific_columns.copy()
        if categorical_column is not None:
            if categorical_column not in filtered_df.columns:
                raise ValueError(f"Categorical column '{categorical_column}' not found in DataFrame")
            columns_to_include.append(categorical_column)

        # Create subset
        df_subset = filtered_df[columns_to_include].copy()

        # Step 3: Apply interpolation
        df_interpolated = self._interpolate_missing_values(df_subset)

        # Step 4: Unpivot/Melt the data
        id_vars = [categorical_column] if categorical_column is not None else []
        value_vars = year_specific_columns

        # Melt the dataframe
        melted_df = pd.melt(df_interpolated,
                           id_vars=id_vars,
                           value_vars=value_vars,
                           var_name='Year_Column',
                           value_name=metric)

        # Step 5: Extract year from column names
        melted_df['Year'] = melted_df['Year_Column'].apply(
            lambda x: self._extract_year_from_column_name(x, metric)
        )

        # Step 6: Clean up and reorder columns
        melted_df = melted_df.drop('Year_Column', axis=1)

        # Reorder columns: Year first, then metric, then categorical (if present)
        if categorical_column is not None:
            melted_df = melted_df[['Year', metric, categorical_column]]
        else:
            melted_df = melted_df[['Year', metric]]

        # Step 7: Remove rows with null values in the metric
        melted_df = melted_df.dropna(subset=[metric])

        # Step 8: Sort by Year and categorical column (if present)
        sort_cols = ['Year']
        if categorical_column is not None:
            sort_cols.append(categorical_column)
        melted_df = melted_df.sort_values(sort_cols).reset_index(drop=True)

        # Get final categories that will appear in the chart
        final_categories = None
        if categorical_column is not None and not melted_df.empty:
            final_categories = sorted(melted_df[categorical_column].unique())

        # Step 9: Calculate vertical oval metrics (BEFORE interpolation, but with final_categories filter)
        vertical_data = []
        if vertical_params and categorical_column:
            vertical_data = self._calculate_vertical_params(
                filtered_df, vertical_params, categorical_column, final_categories
            )

        # Step 10: Calculate horizontal oval metrics (AFTER interpolation and melting)
        horizontal_data = []
        if horizontal_params:
            horizontal_data = self._calculate_horizontal_params(melted_df, horizontal_params, metric)

        # Return based on whether oval parameters were provided
        if horizontal_params or vertical_params:
            return melted_df, horizontal_data, vertical_data
        else:
            return melted_df

    def _extract_year_from_column_name(self, column_name, base_metric):
        """Extract year from a column name given the base metric"""
        base_pattern = base_metric

        # Remove the base pattern to get the year part
        remaining = column_name[len(base_pattern):]

        if remaining.startswith(' '):
            year_part = remaining[1:]

            # Extract single year (YYYY)
            year_match = re.match(r'^(\d{4})$', year_part)
            if year_match:
                return int(year_match.group(1))

            # For growth patterns (YYYY-YYYY), return the latter year
            growth_match = re.match(r'^(\d{4})-(\d{4})$', year_part)
            if growth_match:
                return int(growth_match.group(2))  # Return the end year

        # Fallback: try to find any 4-digit year in the column name
        year_match = re.search(r'\b(\d{4})\b', column_name)
        if year_match:
            return int(year_match.group(1))

        return None

# Convenience functions
def prepare_stacked_data(df, metric, categorical_column=None, percentile_range=(0.005, 0.90)):
      """
      Convenience function to prepare data for stacked bar charts
      """
      processor = StackedSumBarchart(df, percentile_range=percentile_range)
      return processor.process_data(metric, categorical_column)

def get_stacked_options(df, percentile_range=(0.005, 0.90)):
      """
      See available metrics and categorical columns for stacked charts
      """
      processor = StackedSumBarchart(df, percentile_range=percentile_range)
      return {
          'metrics': processor.get_available_metrics(),
          'categorical_columns': processor.get_categorical_columns()
      }

def plot_financial_timeseries_stacked_barplot(df, metric, categorical_column, stack_column,
                                              estimator='sum', figsize=(12, 7), rotation=45,
                                              percentile_range=(0.005, 0.90),
                                              horizontal_params=None, vertical_params=None,
                                              chart_type='regular', CAGR_oval=False, openai=False):
    """
    Creates a stacked bar plot directly from time series financial data with integrated processing.

    This function combines the StackedSumBarchart processing with the plotting functionality
    to create stacked bar charts in a single call. Now supports oval metrics for additional insights
    and CAGR calculations.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the financial time series data
    metric : str
        Base metric name (e.g., 'Revenue EUR', 'EBITDA EUR')
        The function will automatically find all year columns for this metric
    categorical_column : str
        Name of the categorical column for grouping (will become the stack categories)
    stack_column : str
        This parameter is kept for API consistency but will be set to 'Year' internally
        since we're creating time series stacked charts
    estimator : str, default 'sum'
        Statistical estimator to apply ('sum', 'mean', 'median', etc.)
    figsize : tuple, default (12, 7)
        Figure size (width, height)
    rotation : int, default 45
        Rotation angle for x-axis labels
    percentile_range : tuple, default (0.005, 0.90)
        Tuple of (lower_percentile, upper_percentile) to filter outliers
    horizontal_params : dict, optional
        Dictionary with up to 2 key-value pairs for horizontal oval metrics
        Format: {'estimator': 'base_pattern'}, e.g., {'mean': 'Revenue EUR'}
        Shows aggregated values by year
    vertical_params : dict, optional
        Dictionary with up to 2 key-value pairs for vertical oval metrics
        Format: {'estimator': 'column_name'}, e.g., {'sum': 'EBITDA EUR 2024'}
        Shows aggregated values by category
    chart_type : str, default 'regular'
        Type of stacked chart: 'regular', '100%', or 'mekko'
    CAGR_oval : bool, default False
        Whether to display CAGR ovals on the right side of the chart.
        CAGR insights are always calculated and included in the returned insights regardless of this parameter.

    Returns:
    --------
    tuple
        (matplotlib.axes.Axes, dict)
        - Axes object of the created plot
        - Dictionary with insights including CAGR information for each category
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.patches import Ellipse

    # Step 1: Process the data using StackedSumBarchart
    processor = StackedSumBarchart(df, percentile_range=percentile_range)

    # Validate that the metric exists
    available_metrics = processor.get_available_metrics()
    if metric not in available_metrics:
        raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")

    # Process the data to long format (with or without oval metrics)
    if horizontal_params or vertical_params:
        result_df, horizontal_data, vertical_data = processor.process_data(
            metric, categorical_column, horizontal_params, vertical_params
        )
    else:
        result_df = processor.process_data(metric, categorical_column)
        horizontal_data = []
        vertical_data = []

    if result_df.empty:
        raise ValueError(f"No data available after processing metric '{metric}' with category '{categorical_column}'")

    # Generate base insights
    insights_stacked_time_Series = generate_multilevel_aggregations(result_df, categorical_column, 'Year', metric, estimator, values="percentages")
    insights_JSON = add_pareto_insights(insights_stacked_time_Series)

    # Prepare pivot data for CAGR calculation and chart creation
    pivot_data = result_df.pivot_table(
        index='Year',
        columns=categorical_column,
        values=metric,
        aggfunc=AVAILABLE_ESTIMATORS[estimator],
        fill_value=0
    )

    if pivot_data.empty:
        raise ValueError("No data available for plotting after pivot operation")

    # Always calculate CAGR for insights (regardless of CAGR_oval parameter)
    cagr_results = processor._calculate_category_cagrs(pivot_data)

    # Add CAGR information to insights
    for category_name, cagr_info in cagr_results.items():
        key = f"CAGR value for {category_name}"
        insights_JSON[key] = cagr_info['cagr']

    if openai is True:
          insights_JSON = generate_strategic_insights(
              categorical_column, 'Year',metric,insights_JSON,
              API_KEY=API_KEY
          )

    # Prepare CAGR oval data if CAGR_oval=True
    cagr_oval_data = []
    if CAGR_oval and categorical_column:

        # Get final categories for consistency
        final_categories = sorted(result_df[categorical_column].unique())

        # Prepare CAGR data for ovals
        cagr_values = []
        cagr_formatted = []
        cagr_categories = []

        for category in final_categories:
            if category in cagr_results:
                cagr_value = cagr_results[category]['cagr']
                formatted_cagr = processor._format_cagr_value(cagr_value)
            else:
                cagr_value = np.nan
                formatted_cagr = "N/A"

            cagr_values.append(cagr_value)
            cagr_formatted.append(formatted_cagr)
            cagr_categories.append(category)

        cagr_oval_data = [{
            'estimator': 'cagr',
            'title': 'CAGR',
            'values': cagr_values,
            'categories': cagr_categories,
            'formatted_values': cagr_formatted,
            'scale': 1,  # CAGR is already in percentage format
            'suffix': ''
        }]

    if chart_type.lower() in ['100%', '100', 'percentage', 'normalized']:
        # ================================
        # 100% STACKED BAR CHART - INTEGRATED
        # ================================

        fig, ax = plt.subplots(figsize=figsize)

        # Convert to percentages
        pivot_percentage = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
        pivot_percentage = pivot_percentage.fillna(0)

        # Get data for plotting
        x_positions = np.arange(len(pivot_data.index))
        categories = list(pivot_data.columns)
        years = list(pivot_data.index)

        # Create color palette
        colors = create_custom_color_palette(len(categories))

        # Create stacked bars
        bottoms = np.zeros(len(pivot_data.index))

        for i, category in enumerate(categories):
            values = pivot_percentage[category].values
            ax.bar(x_positions, values, bottom=bottoms,
                  label=category, color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)

            # Add percentage labels for segments >= 5%
            for j, (x_pos, value) in enumerate(zip(x_positions, values)):
                if value >= 5:  # Only show labels for segments >= 5%
                    label_y = bottoms[j] + value / 2
                    ax.text(x_pos, label_y, f'{value:.0f}%',
                           ha='center', va='center', color='white', fontweight='bold', fontsize=9)

            bottoms += values

        # Add total value labels on top of each bar (showing original absolute values)
        category_totals = pivot_data.sum(axis=1).values
        all_original_values = pivot_data.values.flatten()
        all_original_values = all_original_values[~pd.isna(all_original_values)]
        scale, suffix = get_scale_and_format_eur(all_original_values, metric)

        label_offset = 3  # 3% offset above the 100% bar
        for i, (x_pos, total_value) in enumerate(zip(x_positions, category_totals)):
            if total_value != 0:
                formatted_total = format_value_consistent(total_value, scale, suffix, metric, estimator)
                label_y = 100 + label_offset
                ax.text(x_pos, label_y, formatted_total,
                       ha='center', va='bottom', color='black', fontweight='bold', fontsize=11)

        # Generate Y-axis label
        label, unit, year = generate_intelligent_ylabel(metric, estimator)
        if label:
            ylabel_base = f'{label} Distribution'
        else:
            ylabel_base = 'Distribution'

        if year:
            ylabel = f'{ylabel_base} ({year}, %)'
        else:
            ylabel = f'{ylabel_base} (%)'

        # Configure axes
        ax.set_xticks(x_positions)
        ax.set_xticklabels(years, fontweight='normal')
        ax.set_xlabel("Year", fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
        ax.set_ylim(0, 100)

        # Apply styling
        ax = setup_chart_style(ax)

        # Add legend with different positions based on params
        has_right_ovals = bool(horizontal_data or vertical_data or cagr_oval_data)
        if has_right_ovals:
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
        # ADD HORIZONTAL OVALS - IMPROVED (100% CHART)
        # ================================
        if horizontal_data:
            chart_height = 100
            total_label_space = 8  # Space for total labels (3% offset + label height)
            dynamic_offset = chart_height * 0.15  # 15% dynamic offset
            oval_base_y = chart_height + total_label_space + dynamic_offset

            # 🔥 MEJORA: Apply same values as first code
            oval_height = 6    # Same as first code
            oval_spacing = 12  # Same as first code

            for row_idx, h_data in enumerate(horizontal_data):
                oval_y = oval_base_y + (row_idx * oval_spacing)

                # Use custom title from processed data with fallback
                title = h_data.get('title', f"{h_data['estimator']} {h_data['base_pattern']}")

                ax.text(-0.8, oval_y, title, ha='right', va='center',
                       fontsize=10, fontweight='normal', color='black')

                # Use formatted_values (now includes count handling)
                formatted_values = h_data.get('formatted_values', [])

                # Add ovals for each year using formatted values
                for i, (x_pos, formatted_value) in enumerate(zip(x_positions, formatted_values)):
                    # 🔥 MEJORA: Apply same width as first code
                    oval_width = 0.55  # Same width as vertical ovals
                    oval = Ellipse((x_pos, oval_y), oval_width, oval_height,
                                 facecolor='lightgray', edgecolor='gray',
                                 linewidth=0.5, alpha=0.8)
                    ax.add_patch(oval)

                    ax.text(x_pos, oval_y, formatted_value,
                           ha='center', va='center', color='black',
                           fontweight='bold', fontsize=9)

            # Adjust ylim to accommodate ovals
            if horizontal_data:
                top_oval_y = oval_base_y + ((len(horizontal_data) - 1) * oval_spacing) + oval_height/2
                ax.set_ylim(0, top_oval_y * 1.05)

        # ================================
        # ADD VERTICAL OVALS - IMPROVED (100% CHART)
        # ================================
        # Combine vertical_data with CAGR ovals
        all_vertical_data = vertical_data + cagr_oval_data

        if all_vertical_data:
            chart_right = len(years) - 0.5
            vertical_oval_base_x = chart_right + 0.8

            # 🔥 MEJORA: Apply same values as first code
            vertical_oval_width = 0.55   # Same as first code
            vertical_oval_height = 7     # Same as first code
            vertical_oval_spacing = 0.6  # Same as first code

            chart_height = 100
            stack_spacing = 10   # Keep spacing between ovals of the same column
            total_stack_height = (len(categories) - 1) * stack_spacing
            vertical_oval_start_y = (chart_height - total_stack_height) / 2

            for col_idx, v_data in enumerate(all_vertical_data):
                oval_x = vertical_oval_base_x + (col_idx * vertical_oval_spacing)

                # Use custom title from processed data with fallback
                title = v_data.get('title', f"{v_data['estimator']} {v_data.get('column', '')}")
                title_y = chart_height * 0.95

                # 🔥 FIXED: Improved title positioning to avoid overlap
                max_chars_per_line = 8  # Reduced for better fit
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
                # 🔥 FIXED: Position titles closer to ovals, not too far above
                title_y_closer = title_y - 5  # Bring titles closer to the ovals
                ax.text(oval_x, title_y_closer, title_text, ha='center', va='bottom',
                       fontsize=9, fontweight='normal', color='black',
                       rotation=0, multialignment='center')

                # Use formatted_values if available
                formatted_values = v_data.get('formatted_values', [])
                categories_list = v_data.get('categories', categories)

                # Add ovals for each category using formatted values
                for i, (category, formatted_value) in enumerate(zip(categories_list, formatted_values)):
                    oval_y = vertical_oval_start_y + (i * stack_spacing)

                    # Use appropriate color (match with bar colors for all ovals including CAGR)
                    color_index = i if i < len(colors) else i % len(colors)
                    oval_color = colors[color_index]
                    text_color = 'white'
                    edge_color = 'white'

                    oval = Ellipse((oval_x, oval_y), vertical_oval_width, vertical_oval_height,
                                 facecolor=oval_color, edgecolor=edge_color,
                                 linewidth=1, alpha=0.9)
                    ax.add_patch(oval)

                    ax.text(oval_x, oval_y, formatted_value,
                           ha='center', va='center', color=text_color,
                           fontweight='bold', fontsize=9)

            # 🔥 MEJORA: Adjust xlim to accommodate vertical ovals
            if all_vertical_data:
                rightmost_oval_x = vertical_oval_base_x + ((len(all_vertical_data) - 1) * vertical_oval_spacing) + vertical_oval_width/2
                ax.set_xlim(ax.get_xlim()[0], rightmost_oval_x * 1.05)

        plt.tight_layout()

    elif chart_type.lower() in ['regular', 'normal', 'standard']:
        # ================================
        # REGULAR STACKED BAR CHART - INTEGRATED
        # ================================

        fig, ax = plt.subplots(figsize=figsize)

        # Get data for plotting
        x_positions = np.arange(len(pivot_data.index))
        categories = list(pivot_data.columns)
        years = list(pivot_data.index)

        # Create color palette
        colors = create_custom_color_palette(len(categories))

        # Create stacked bars
        bottoms = np.zeros(len(pivot_data.index))

        for i, category in enumerate(categories):
            values = pivot_data[category].values
            ax.bar(x_positions, values, bottom=bottoms,
                  label=category, color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
            bottoms += values

        # Get scale for formatting
        all_values = pivot_data.values.flatten()
        all_values = all_values[~pd.isna(all_values)]
        scale, suffix = get_scale_and_format_eur(all_values, metric)

        # Add total value labels on top of each bar
        category_totals = pivot_data.sum(axis=1).values
        for i, (x_pos, total_value) in enumerate(zip(x_positions, category_totals)):
            if total_value != 0:
                formatted_total = format_value_consistent(total_value, scale, suffix, metric, estimator)
                ax.text(x_pos, max(bottoms), formatted_total,
                       ha='center', va='bottom', color='black', fontweight='bold', fontsize=11)

        # Generate Y-axis label
        label, unit, year = generate_intelligent_ylabel(metric, estimator)
        if year:
            ylabel = f'{label} ({year})'
        else:
            ylabel = label

        # Configure axes
        ax.set_xticks(x_positions)
        ax.set_xticklabels(years, fontweight='normal')
        ax.set_xlabel("Year", fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')

        # Apply styling
        ax = setup_chart_style(ax)

        # Add legend
        has_right_ovals = bool(horizontal_data or vertical_data or cagr_oval_data)
        if has_right_ovals:
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
        # ADD HORIZONTAL OVALS - IMPROVED (REGULAR CHART)
        # ================================
        if horizontal_data:
            chart_top = max(category_totals) if category_totals.size > 0 else 0
            oval_base_y = chart_top * 1.15

            # 🔥 MEJORA: Apply proportional values for regular chart
            oval_height = chart_top * 0.05  # Proportional to chart height
            oval_spacing = chart_top * 0.08  # Proportional spacing

            for row_idx, h_data in enumerate(horizontal_data):
                oval_y = oval_base_y + (row_idx * oval_spacing)
                title = h_data.get('title', f"{h_data['estimator']} {h_data['base_pattern']}")

                ax.text(-0.8, oval_y, title, ha='right', va='center',
                       fontsize=10, fontweight='normal', color='black')

                # Use formatted_values (now includes count handling)
                formatted_values = h_data.get('formatted_values', [])

                for i, (x_pos, formatted_value) in enumerate(zip(x_positions, formatted_values)):
                    # 🔥 MEJORA: Apply same width as first code
                    oval_width = 0.55  # Same width as vertical ovals
                    oval = Ellipse((x_pos, oval_y), oval_width, oval_height,
                                 facecolor='lightgray', edgecolor='gray',
                                 linewidth=0.5, alpha=0.8)
                    ax.add_patch(oval)

                    ax.text(x_pos, oval_y, formatted_value,
                           ha='center', va='center', color='black',
                           fontweight='bold', fontsize=9)

            # Adjust ylim
            if horizontal_data:
                top_oval_y = oval_base_y + ((len(horizontal_data) - 1) * oval_spacing) + oval_height/2
                ax.set_ylim(0, top_oval_y * 1.05)

        # ================================
        # ADD VERTICAL OVALS - IMPROVED (REGULAR CHART)
        # ================================
        # Combine vertical_data with CAGR ovals
        all_vertical_data = vertical_data + cagr_oval_data

        if all_vertical_data:
            chart_right = len(years) - 0.5
            vertical_oval_base_x = chart_right + 0.8

            # 🔥 MEJORA: Apply same values as first code
            vertical_oval_width = 0.55   # Same as first code
            vertical_oval_height = max(category_totals) * 0.07 if category_totals.size > 0 else 7  # Proportionally adjusted
            vertical_oval_spacing = 0.6  # Same as first code

            chart_height = max(category_totals) if category_totals.size > 0 else 100
            stack_spacing = chart_height * 0.10  # Adjusted for regular chart
            total_stack_height = (len(categories) - 1) * stack_spacing
            vertical_oval_start_y = (chart_height - total_stack_height) / 2

            for col_idx, v_data in enumerate(all_vertical_data):
                oval_x = vertical_oval_base_x + (col_idx * vertical_oval_spacing)
                title = v_data['title']

                # Add title above
                title_y = chart_height * 0.95

                # Split long titles into multiple lines (same as first code)
                max_chars_per_line = 8  # Reduced for better fit
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
                # 🔥 FIXED: Position titles closer to ovals, not too far above
                title_y_closer = title_y - 5  # Bring titles closer to the ovals
                ax.text(oval_x, title_y_closer, title_text, ha='center', va='bottom',
                       fontsize=9, fontweight='normal', color='black',
                       multialignment='center')

                # Add ovals using formatted values (now includes count handling)
                for i, (category, formatted_value) in enumerate(zip(v_data['categories'], v_data['formatted_values'])):
                    oval_y = vertical_oval_start_y + (i * stack_spacing)

                    # Use appropriate color
                    color_index = i if i < len(colors) else i % len(colors)
                    oval_color = colors[color_index]
                    text_color = 'white'
                    edge_color = 'white'

                    oval = Ellipse((oval_x, oval_y), vertical_oval_width, vertical_oval_height,
                                 facecolor=oval_color, edgecolor=edge_color,
                                 linewidth=1, alpha=0.9)
                    ax.add_patch(oval)

                    ax.text(oval_x, oval_y, formatted_value,
                           ha='center', va='center', color=text_color,
                           fontweight='bold', fontsize=9)

            # Adjust xlim to accommodate vertical ovals
            if all_vertical_data:
                rightmost_oval_x = vertical_oval_base_x + ((len(all_vertical_data) - 1) * vertical_oval_spacing) + vertical_oval_width/2
                ax.set_xlim(ax.get_xlim()[0], rightmost_oval_x * 1.05)

        plt.tight_layout()

    else:
        raise ValueError("chart_type must be 'regular', '100%', or 'mekko'")

    if CAGR_oval:
        print(f"✅ {chart_type.capitalize()} stacked bar chart created successfully with CAGR ovals!")
    else:
        print(f"✅ {chart_type.capitalize()} stacked bar chart created successfully!")

    return ax, insights_JSON