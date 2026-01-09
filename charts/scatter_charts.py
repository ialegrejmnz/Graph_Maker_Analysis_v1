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
    generate_strategic_insights
)

from common_functions import (
    get_scale_and_format_eur,
    format_value_consistent
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
    show_r2: bool = True,  # Mostrar R² en la leyenda
    openai=False
):
    """
    Creates a scatter plot or bubble chart with linear and non-linear regression options.
    Returns both the figure and a comprehensive insights analysis.

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
    If categorical_col is provided:
        tuple: (matplotlib.figure.Figure, dict) - Figure and comprehensive analysis
    If categorical_col is None:
        tuple: (matplotlib.figure.Figure, dict) - Figure and basic analysis
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

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

    # Check for company names column (needed for top companies analysis)
    has_company_names = 'Company name Latin alphabet' in data.columns

    # Validate company names column if provided
    if names:
        if not isinstance(names, dict):
            raise ValueError("names must be a dictionary with format {'column_name': N}")

        if not has_company_names:
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
    if names and has_company_names:
        essential_cols.append('Company name Latin alphabet')

    data = data.dropna(subset=essential_cols)

    if len(data) == 0:
        raise ValueError("No data remaining after filtering and removing NaN values")

    # Store original data for analysis
    original_data = data.copy()

    # ============= INSIGHTS ANALYSIS FUNCTIONS =============

    def calculate_linear_slope(x_data, y_data):
        """Calculate linear regression slope"""
        if len(x_data) < 2:
            return None
        try:
            x_reg = x_data.values.reshape(-1, 1)
            y_reg = y_data.values
            model = LinearRegression()
            model.fit(x_reg, y_reg)
            return model.coef_[0]
        except:
            return None

    def generate_slope_insight(slope, category_name=None):
        """Generate slope insight text"""
        if slope is None:
            return None

        if category_name:
            base_text = f"By performing a linear regression between {x_axis_col} and {y_axis_col} for {category_name}, a slope of {slope:.3f} is obtained, which means that when {x_axis_col} increases by one unit, {y_axis_col} tends to "
        else:
            base_text = f"By performing a linear regression between {x_axis_col} and {y_axis_col}, a slope of {slope:.3f} is obtained, which means that when {x_axis_col} increases by one unit, {y_axis_col} tends to "

        if slope > 0:
            return base_text + f"increase by {slope:.3f} units"
        else:
            return base_text + f"decrease by {abs(slope):.3f} units"

        if category_name:
            return base_text + " in this category."
        else:
            return base_text + "."

    def get_top_companies_text(data, column, n=5, category_name=None):
        """Generate top companies text with values"""
        if not has_company_names:
            if category_name:
                return f"Company names not available for {column} analysis in {category_name} category"
            else:
                return f"Company names not available for {column} analysis"

        # Get top companies
        top_companies = data.nlargest(n, column)
        if len(top_companies) == 0:
            if category_name:
                return f"No companies available for {column} analysis in {category_name} category"
            else:
                return f"No companies available for {column} analysis"

        # Get scale and format for the column
        scale, suffix = get_scale_and_format_eur(data[column], column)

        # Format companies with values
        company_texts = []
        actual_count = min(len(top_companies), n)
        for _, row in top_companies.iterrows():
            company_name = row['Company name Latin alphabet']
            value = row[column]
            formatted_value = format_value_consistent(value, scale, suffix, column, 'mean')
            company_texts.append(f"{company_name} with {formatted_value}")

        # Create final text with proper number handling
        companies_list = ", ".join(company_texts)
        if actual_count == 1:
            number_word = "company"
        else:
            number_words = {2: "two", 3: "three", 4: "four", 5: "five"}
            number_word = number_words.get(actual_count, str(actual_count)) + " companies"

        if category_name:
            return f"The {number_word} with the highest {column} in {category_name} category {'is' if actual_count == 1 else 'are'}: {companies_list}"
        else:
            return f"The {number_word} with the highest {column} {'is' if actual_count == 1 else 'are'}: {companies_list}"

    def analyze_categories(data, categorical_col, x_col, y_col, size_col=None):
        """Analyze data by categories"""
        analysis = {}
        total_count = len(data)

        # Total analysis
        total_slope = calculate_linear_slope(data[x_col], data[y_col])
        total_analysis = {
            "company_count": total_count,
            "company_percentage": 100.0,
            f"{x_col}_mean": data[x_col].mean(),
            f"{y_col}_mean": data[y_col].mean(),
            "slope": total_slope
        }
        if size_col:
            total_analysis[f"{size_col}_mean"] = data[size_col].mean()

        analysis["Total"] = total_analysis

        # Category analysis
        categories = data[categorical_col].unique()
        for category in categories:
            category_data = data[data[categorical_col] == category]
            category_count = len(category_data)
            category_percentage = (category_count / total_count * 100) if total_count > 0 else 0

            # Calculate slope for category
            category_slope = calculate_linear_slope(category_data[x_col], category_data[y_col])

            category_analysis = {
                "company_count": category_count,
                "company_percentage": category_percentage,
                f"{x_col}_mean": category_data[x_col].mean(),
                f"{y_col}_mean": category_data[y_col].mean(),
                "slope": category_slope
            }

            if size_col:
                category_analysis[f"{size_col}_mean"] = category_data[size_col].mean()

            analysis[category] = category_analysis

        return analysis

    def generate_category_comparisons(categories_analysis, x_col, y_col, size_col=None):
        """Generate comparative insights between categories"""
        comparisons = []

        # Get categories (excluding Total)
        category_data = {name: data for name, data in categories_analysis.items()
                        if name != "Total"}
        total_data = categories_analysis.get("Total", {})

        if len(category_data) == 0:
            return comparisons

        category_names = list(category_data.keys())

        # Helper function for safe ratios
        def get_ratio(val1, val2):
            if abs(val2) < 0.001:
                return None
            return abs(val1 / val2)

        # 1. COMPARISONS BETWEEN CATEGORIES
        if len(category_names) >= 2:
            for i, cat1 in enumerate(category_names):
                for cat2 in category_names[i+1:]:  # Avoid duplicates
                    data1 = category_data[cat1]
                    data2 = category_data[cat2]

                    # Compare x_col means
                    ratio = get_ratio(data1[f"{x_col}_mean"], data2[f"{x_col}_mean"])
                    if ratio and ratio > 1.2:
                        comparisons.append(f"The mean of {x_col} in {cat1} category is {ratio:.1f}x higher than in {cat2} category")

                    # Compare y_col means
                    ratio = get_ratio(data1[f"{y_col}_mean"], data2[f"{y_col}_mean"])
                    if ratio and ratio > 1.2:
                        comparisons.append(f"The mean of {y_col} in {cat1} category is {ratio:.1f}x higher than in {cat2} category")

                    # Compare size_col means if exists
                    if size_col:
                        ratio = get_ratio(data1[f"{size_col}_mean"], data2[f"{size_col}_mean"])
                        if ratio and ratio > 1.2:
                            comparisons.append(f"The mean of {size_col} in {cat1} category is {ratio:.1f}x higher than in {cat2} category")

                    # Compare company counts
                    ratio = get_ratio(data1["company_count"], data2["company_count"])
                    if ratio and ratio > 1.2:
                        comparisons.append(f"The number of companies in {cat1} category is {ratio:.1f}x higher than in {cat2} category ({data1['company_count']} vs {data2['company_count']})")

        # 2. COMPARISONS BETWEEN CATEGORIES AND TOTAL
        for category_name, category_info in category_data.items():
            # Compare x_col with total
            ratio = get_ratio(category_info[f"{x_col}_mean"], total_data[f"{x_col}_mean"])
            if ratio and ratio > 1.2:
                comparisons.append(f"The mean of {x_col} in {category_name} category is {ratio:.1f}x higher than the mean of {x_col} in all the data")
            elif ratio and ratio < 0.8:
                inverse_ratio = get_ratio(total_data[f"{x_col}_mean"], category_info[f"{x_col}_mean"])
                if inverse_ratio:
                    comparisons.append(f"The mean of {x_col} in all the data is {inverse_ratio:.1f}x higher than the mean of {x_col} in {category_name} category")

            # Compare y_col with total
            ratio = get_ratio(category_info[f"{y_col}_mean"], total_data[f"{y_col}_mean"])
            if ratio and ratio > 1.2:
                comparisons.append(f"The mean of {y_col} in {category_name} category is {ratio:.1f}x higher than the mean of {y_col} in all the data")
            elif ratio and ratio < 0.8:
                inverse_ratio = get_ratio(total_data[f"{y_col}_mean"], category_info[f"{y_col}_mean"])
                if inverse_ratio:
                    comparisons.append(f"The mean of {y_col} in all the data is {inverse_ratio:.1f}x higher than the mean of {y_col} in {category_name} category")

            # Compare size_col with total if exists
            if size_col:
                ratio = get_ratio(category_info[f"{size_col}_mean"], total_data[f"{size_col}_mean"])
                if ratio and ratio > 1.2:
                    comparisons.append(f"The mean of {size_col} in {category_name} category is {ratio:.1f}x higher than the mean of {size_col} in all the data")
                elif ratio and ratio < 0.8:
                    inverse_ratio = get_ratio(total_data[f"{size_col}_mean"], category_info[f"{size_col}_mean"])
                    if inverse_ratio:
                        comparisons.append(f"The mean of {size_col} in all the data is {inverse_ratio:.1f}x higher than the mean of {size_col} in {category_name} category")

        # Limit comparisons to avoid overwhelm
        if len(comparisons) > 12:
            comparisons = comparisons[:12]

        return comparisons

    # ============= GENERATE INSIGHTS =============

    if categorical_col:
        # WITH CATEGORICAL COLUMN
        categories_analysis = analyze_categories(original_data, categorical_col, x_axis_col, y_axis_col, size_col)
        comparisons = generate_category_comparisons(categories_analysis, x_axis_col, y_axis_col, size_col)

        # Generate regression insights for each category
        regression_insights = []
        for category_name, category_info in categories_analysis.items():
            if category_name != "Total" and category_info["slope"] is not None:
                slope_insight = generate_slope_insight(category_info["slope"], category_name)
                if slope_insight:
                    regression_insights.append(slope_insight)

        # Generate top companies text BY CATEGORY
        top_companies = {}
        if has_company_names:
            # Top companies for each category
            categories = original_data[categorical_col].unique()
            for category in categories:
                category_data = original_data[original_data[categorical_col] == category]
                top_companies[f"top_x_axis_{category}"] = get_top_companies_text(category_data, x_axis_col, category_name=category)
                top_companies[f"top_y_axis_{category}"] = get_top_companies_text(category_data, y_axis_col, category_name=category)
        else:
            # If no company names available, still show structure by category
            categories = original_data[categorical_col].unique()
            for category in categories:
                top_companies[f"top_x_axis_{category}"] = f"Company names not available for {x_axis_col} analysis in {category} category"
                top_companies[f"top_y_axis_{category}"] = f"Company names not available for {y_axis_col} analysis in {category} category"

        # Build analysis JSON
        analysis_json = {
            "categories_analysis": categories_analysis,
            "comparisons": comparisons,
            "regression_insights": regression_insights,
            "top_companies": top_companies
        }
    else:
        # WITHOUT CATEGORICAL COLUMN
        total_count = len(original_data)
        total_slope = calculate_linear_slope(original_data[x_axis_col], original_data[y_axis_col])

        overall_analysis = {
            "company_count": total_count,
            f"{x_axis_col}_mean": original_data[x_axis_col].mean(),
            f"{y_axis_col}_mean": original_data[y_axis_col].mean(),
            "slope": total_slope
        }

        if size_col:
            overall_analysis[f"{size_col}_mean"] = original_data[size_col].mean()

        # Generate regression insight
        regression_insight = generate_slope_insight(total_slope)

        # Generate top companies text
        top_companies = {}
        if has_company_names:
            top_companies["top_x_axis"] = get_top_companies_text(original_data, x_axis_col)
            top_companies["top_y_axis"] = get_top_companies_text(original_data, y_axis_col)
        else:
            top_companies["top_x_axis"] = f"Company names not available for {x_axis_col} analysis"
            top_companies["top_y_axis"] = f"Company names not available for {y_axis_col} analysis"

        # Build analysis JSON
        analysis_json = {
            "overall_analysis": overall_analysis,
            "regression_insight": regression_insight if regression_insight else "Unable to calculate regression slope",
            "top_companies": top_companies
        }

    if openai is True:
          analysis_json = generate_strategic_insights(
              y_axis_col, categorical_col,x_axis_col,analysis_json,
              API_KEY=API_KEY
          )
    # ============= PLOTTING (ORIGINAL CODE) =============

    # Determine which companies should have names shown
    companies_to_name = set()
    if names:
        for column, n_companies in names.items():
            if n_companies > 0:
                top_companies_df = data.nlargest(n_companies, column)['Company name Latin alphabet'].tolist()
                companies_to_name.update(top_companies_df)
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
            if names and companies_to_name and has_company_names:
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
        if names and companies_to_name and has_company_names:
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

    # Return figure and analysis
    return fig, analysis_json