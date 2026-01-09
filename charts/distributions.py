import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    generate_strategic_insights
)

from common_functions import (
    get_scale_and_format_eur,
    format_eur_axis,
    format_regular_axis,
    format_category_name,
    format_value_consistent,
    apply_outlier_filtering,
)

def plot_distributions_by_category(df, categorical_column, numeric_column, figsize=(8, 10),
                                   bandwidth=1.0, remove_outliers=True, percentile_range=(0.05, 0.95),
                                   transparent_bg=True, show_global_mean=True, n_ticks=6,
                                   diff_from_mean=False, openai=False):
    """
    Creates a distribution plot with KDE curves and quantile bands for each category.
    Now returns both the plot and comprehensive analysis dictionary.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    categorical_column : str
        Name of the categorical column
    numeric_column : str
        Name of the numeric column to plot on x-axis
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 10)
    bandwidth : float, optional
        Bandwidth adjustment for KDE. Default is 1.0
    remove_outliers : bool, optional
        Whether to remove outliers using percentile method. Default is True
    percentile_range : tuple, optional
        Range of percentiles to keep (lower, upper). Default is (0.05, 0.95)
    transparent_bg : bool, optional
        Whether to set transparent background for saving. Default is True
    show_global_mean : bool, optional
        Whether to show global mean as vertical dashed line. Default is True
    n_ticks : int, optional
        Number of ticks to show on x-axis. Default is 6
    diff_from_mean : bool, optional
        Whether to show percentage difference from global mean for each category. Default is False

    Returns:
    --------
    tuple: (fig, axs, analysis_dict)
        - fig: matplotlib figure object
        - axs: matplotlib axes objects
        - analysis_dict: comprehensive analysis dictionary
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Define color palette for quantiles based on main purple color
    main_purple = '#7D4BEB'
    mid_purple = '#A284F0'    # Lighter version
    light_purple = '#C7B8F5'  # Even lighter version
    colors = [light_purple, mid_purple, main_purple, mid_purple, light_purple]

    darkgrey = '#525252'

    # Validate input columns
    if categorical_column not in df.columns:
        raise ValueError(f"Column '{categorical_column}' not found in DataFrame")
    if numeric_column not in df.columns:
        raise ValueError(f"Column '{numeric_column}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        raise ValueError(f"Column '{numeric_column}' must be numeric")

    # Create copy and clean data
    df_clean = df[[categorical_column, numeric_column]].copy().dropna()

    if len(df_clean) == 0:
        raise ValueError("No data remaining after removing NaN values")

    # Filter outliers if requested
    df_plot = df_clean.copy()
    if remove_outliers:
        df_plot = apply_outlier_filtering(df_plot, [numeric_column], percentile_range)

    global_mean = df_plot[numeric_column].mean()
    global_median = df_plot[numeric_column].median()
    total_companies = len(df_plot)

    # Get unique categories sorted by mean of numeric column
    categories = df_plot.groupby(categorical_column)[numeric_column].mean().sort_values().index.tolist()
    n_categories = len(categories)

    if n_categories == 0:
        raise ValueError("No categories found in data")

    # Initialize analysis dictionary
    analysis_dict = {
        "global_analysis": {
            "global_mean": global_mean,
            "global_median": global_median,
            "total_companies": total_companies,
        },
        "categories_analysis": {},
        "insights": []
    }

    # Analyze each category
    for category in categories:
        subset = df_plot[df_plot[categorical_column] == category].copy()

        if len(subset) == 0:
            continue

        # Basic statistics
        cat_mean = subset[numeric_column].mean()
        cat_median = subset[numeric_column].median()
        cat_total_companies = len(subset)

        # Distance to global mean
        distance_absolute = (cat_mean - global_mean)

        # Calculate 80% concentration
        subset_sorted = subset.sort_values(numeric_column, ascending=False).copy()
        subset_sorted['cumsum'] = subset_sorted[numeric_column].cumsum()
        total_value = subset_sorted[numeric_column].sum()
        target_value = total_value * 0.8

        # Find how many companies represent 80% or more
        companies_80_percent = 0
        if total_value > 0:
            companies_80_percent = len(subset_sorted[subset_sorted['cumsum'] <= target_value]) + 1
            companies_80_percent = min(companies_80_percent, cat_total_companies)

        concentration_ratio = companies_80_percent / cat_total_companies if cat_total_companies > 0 else 0

        # Tercile analysis
        cat_min = subset[numeric_column].min()
        cat_max = subset[numeric_column].max()
        cat_range = cat_max - cat_min

        if cat_range > 0:
            tercile_1_cutoff = cat_min + (cat_range / 3)
            tercile_2_cutoff = cat_min + (2 * cat_range / 3)

            companies_tercile_1 = len(subset[subset[numeric_column] <= tercile_1_cutoff])
            companies_tercile_2 = len(subset[(subset[numeric_column] > tercile_1_cutoff) &
                                           (subset[numeric_column] <= tercile_2_cutoff)])
            companies_tercile_3 = len(subset[subset[numeric_column] > tercile_2_cutoff])
        else:
            # If all values are the same
            tercile_1_cutoff = cat_min
            tercile_2_cutoff = cat_min
            companies_tercile_1 = cat_total_companies
            companies_tercile_2 = 0
            companies_tercile_3 = 0

        # Store category analysis
        analysis_dict["categories_analysis"][category] = {
            "mean": cat_mean,
            "median": cat_median,
            "distance_to_global_mean": distance_absolute,
            "companies_for_80_percent": companies_80_percent,
            "total_companies": cat_total_companies,
            "concentration_ratio": concentration_ratio,
            "tercile_analysis": {
                "min_value": cat_min,
                "max_value": cat_max,
                "tercile_1_cutoff": tercile_1_cutoff,
                "tercile_2_cutoff": tercile_2_cutoff,
                "companies_tercile_1": companies_tercile_1,
                "companies_tercile_2": companies_tercile_2,
                "companies_tercile_3": companies_tercile_3
            }
        }

    # Generate insights
    insights = generate_distribution_insights(analysis_dict, numeric_column, categorical_column)
    analysis_dict["insights"] = insights
    if openai is True:
      insights = generate_strategic_insights(
          categorical_column, None,numeric_column,insights,
          API_KEY=API_KEY
      )

    # CREATE THE PLOT
    # Create subplots with transparent background if requested
    fig, axs = plt.subplots(nrows=n_categories, ncols=1, figsize=figsize)

    # Set transparent background
    if transparent_bg:
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0.0)

    # Handle case where there's only one category
    if n_categories == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    # Get overall data range for consistent x-axis
    x_min = df_plot[numeric_column].min()
    x_max = df_plot[numeric_column].max()
    x_range = x_max - x_min
    x_padding = x_range * 0.05  # 5% padding

    # Calculate the left margin adjustment based on whether diff_from_mean is enabled
    base_left_margin = x_padding * 2
    diff_margin = x_padding * 4 if diff_from_mean else 0
    total_left_margin = base_left_margin + diff_margin

    # Iterate over categories
    for i, category in enumerate(categories):

        # Set transparent background for each subplot if requested
        if transparent_bg:
            axs[i].patch.set_facecolor('none')
            axs[i].patch.set_alpha(0.0)

        # Subset data for current category
        subset = df_plot[df_plot[categorical_column] == category]

        if len(subset) == 0:
            continue

        # Plot KDE distribution
        sns.kdeplot(
            data=subset[numeric_column],
            fill=True,
            bw_adjust=bandwidth,
            ax=axs[i],
            color='grey',
            edgecolor='lightgrey'
        )

        # Set axis limits FIRST to ensure consistency
        axs[i].set_xlim(x_min - x_padding, x_max + x_padding)

        # Get y-axis limits after plotting
        y_max = axs[i].get_ylim()[1]
        axs[i].set_ylim(0, y_max)

        # Format category name (handle long names by splitting them)
        formatted_category = format_category_name(category)

        # Display category name on the left
        axs[i].text(
            x_min - total_left_margin,
            y_max * 0.5,
            formatted_category,
            ha='right',
            va='center',
            fontsize=10,
            fontweight='semibold',
            color=darkgrey
        )

        # Display percentage difference from mean if requested
        if diff_from_mean:
            category_mean = subset[numeric_column].mean()
            # Calculate percentage difference: (category_mean - global_mean) / global_mean * 100
            pct_diff = ((category_mean - global_mean) / global_mean) * 100 if global_mean != 0 else 0

            # Format the percentage with sign
            sign = '+' if pct_diff >= 0 else ''
            diff_text = f"({sign}{pct_diff:.1f}%)"

            # Position it between the category name and the plot
            axs[i].text(
                x_min - base_left_margin,
                y_max * 0.5,
                diff_text,
                ha='right',
                va='center',
                fontsize=9,
                color=darkgrey
            )

        # Calculate quantiles
        quantiles = np.nanpercentile(subset[numeric_column], [2.5, 10, 25, 75, 90, 97.5])

        # Fill areas between quantiles
        for j in range(len(quantiles) - 1):
            axs[i].fill_between(
                [quantiles[j], quantiles[j+1]],
                0,
                y_max * 0.2,  # 20% of max height
                color=colors[j],
                alpha=0.8
            )

        # Add mean value point for this category
        mean_value = subset[numeric_column].mean()
        axs[i].scatter([mean_value], [y_max * 0.1], color='black', s=20, zorder=5)

        # Add global mean vertical line (dashed black line) - AFTER setting limits
        if show_global_mean:
            axs[i].axvline(
                x=global_mean,
                color='black',
                linestyle='--',
                linewidth=1.5,
                alpha=0.8,
                ymin=0,  # Start from bottom
                ymax=1,  # Go to top (normalized coordinates)
                zorder=4
            )

        # Remove y-axis label and ticks
        axs[i].set_ylabel('')
        axs[i].set_yticks([])

        # Only show x-axis for the last subplot
        if i < len(categories) - 1:
            axs[i].set_xticks([])
            axs[i].set_xlabel('')
        else:
            # Check if numeric column contains 'EUR' for special formatting
            if 'eur' in numeric_column.lower():
                # Apply custom EUR formatting to x-axis
                axs[i] = format_eur_axis(axs[i], x_min - x_padding, x_max + x_padding, n_ticks)
            else:
                # Apply regular numeric formatting
                axs[i] = format_regular_axis(axs[i], x_min - x_padding, x_max + x_padding, n_ticks)
            axs[i].set_xlabel(numeric_column.replace('_', ' ').title(), fontsize=12)

        # Set thin black axis lines (only bottom axis, no left axis)
        for spine in axs[i].spines.values():
            spine.set_visible(False)

        # Show only bottom spine
        axs[i].spines['bottom'].set_visible(True)
        axs[i].spines['bottom'].set_color('black')
        axs[i].spines['bottom'].set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()

    return fig, axs, analysis_dict

def generate_distribution_insights(analysis_dict, numeric_column, categorical_column):
    """
    Generate insights from the distribution analysis.

    Parameters:
    -----------
    analysis_dict : dict
        Analysis dictionary with category statistics
    numeric_column : str
        Name of the numeric column
    categorical_column : str
        Name of the categorical column

    Returns:
    --------
    list: List of generated insights
    """
    insights = []
    categories_analysis = analysis_dict["categories_analysis"]
    global_mean = analysis_dict["global_analysis"]["global_mean"]

    if len(categories_analysis) == 0:
        return insights

    # Get formatting info for the numeric column
    scale, suffix = get_scale_and_format_eur(
        [data["mean"] for data in categories_analysis.values()],
        numeric_column
    )

    def format_value_for_insight(value):
        """Format value for insight text"""
        return format_value_consistent(value, scale, suffix, numeric_column, 'mean')

    # 1. TERCILE INSIGHTS for each category
    for category, data in categories_analysis.items():
        tercile = data["tercile_analysis"]

        if tercile["companies_tercile_1"] > 0 or tercile["companies_tercile_2"] > 0 or tercile["companies_tercile_3"] > 0:
            cutoff_1_formatted = format_value_for_insight(tercile["tercile_1_cutoff"])
            cutoff_2_formatted = format_value_for_insight(tercile["tercile_2_cutoff"])

            insight = (f"In {category}, we observe that there are "
                      f"{tercile['companies_tercile_1']} companies below {cutoff_1_formatted}, "
                      f"{tercile['companies_tercile_2']} companies between {cutoff_1_formatted} and {cutoff_2_formatted}, "
                      f"and {tercile['companies_tercile_3']} companies above {cutoff_2_formatted}.")
            insights.append(insight)

    # 2. CONCENTRATION INSIGHTS
    # Find category with highest and lowest concentration
    concentration_data = {cat: data["concentration_ratio"] for cat, data in categories_analysis.items()}

    if len(concentration_data) > 1:
        highest_concentration_cat = max(concentration_data.keys(), key=lambda k: concentration_data[k])
        lowest_concentration_cat = min(concentration_data.keys(), key=lambda k: concentration_data[k])

        if highest_concentration_cat != lowest_concentration_cat:
            highest_conc = categories_analysis[highest_concentration_cat]
            lowest_conc = categories_analysis[lowest_concentration_cat]

            insights.append(f"{highest_concentration_cat} shows the highest concentration: "
                          f"only {highest_conc['concentration_ratio']:.1%} of companies "
                          f"({highest_conc['companies_for_80_percent']} out of {highest_conc['total_companies']}) "
                          f"account for 80% of the total {numeric_column.replace('_', ' ').lower()}.")

            insights.append(f"{lowest_concentration_cat} shows the most distributed pattern: "
                          f"{lowest_conc['concentration_ratio']:.1%} of companies "
                          f"({lowest_conc['companies_for_80_percent']} out of {lowest_conc['total_companies']}) "
                          f"are needed to reach 80% of the total {numeric_column.replace('_', ' ').lower()}.")

    # 3. MEAN COMPARISON INSIGHTS
    if len(categories_analysis) > 1:
        mean_values = {cat: data["mean"] for cat, data in categories_analysis.items()}
        highest_mean_cat = max(mean_values.keys(), key=lambda k: mean_values[k])
        lowest_mean_cat = min(mean_values.keys(), key=lambda k: mean_values[k])

        if highest_mean_cat != lowest_mean_cat:
            highest_mean = mean_values[highest_mean_cat]
            lowest_mean = mean_values[lowest_mean_cat]

            if lowest_mean != 0:
                ratio = highest_mean / lowest_mean
                if ratio > 1.5:
                    insights.append(f"The mean {numeric_column.replace('_', ' ').lower()} in {highest_mean_cat} "
                                  f"is {ratio:.1f}x higher than in {lowest_mean_cat} "
                                  f"({format_value_for_insight(highest_mean)} vs {format_value_for_insight(lowest_mean)}).")

    # 5. COMPANY COUNT INSIGHTS
    if len(categories_analysis) > 1:
        company_counts = {cat: data["total_companies"] for cat, data in categories_analysis.items()}
        largest_cat = max(company_counts.keys(), key=lambda k: company_counts[k])
        smallest_cat = min(company_counts.keys(), key=lambda k: company_counts[k])

        if largest_cat != smallest_cat:
            largest_count = company_counts[largest_cat]
            smallest_count = company_counts[smallest_cat]

            if smallest_count > 0 and largest_count / smallest_count > 2:
                insights.append(f"{largest_cat} contains the most companies ({largest_count}), "
                              f"while {smallest_cat} contains the fewest ({smallest_count}).")

    # Limit to avoid overwhelming the user
    if len(insights) > 10:
        insights = insights[:10]

    return insights

def create_custom_kde_plot(data, numeric_column, categorical_column, percentile_range=(0, 1),
                          figsize=(12, 6), alpha=0.4, common_norm=False, n_ticks=5,
                          custom_colors=None,openai=False):
    """
    Create a customized KDE plot with transparent background, hidden Y-axis, and comprehensive analysis.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to plot
    numeric_column : str
        Name of the numeric column for X-axis
    categorical_column : str
        Name of the categorical column for hue grouping
    percentile_range : tuple, optional
        Tuple with two elements (lower_percentile, upper_percentile) to filter outliers.
        Default is (0, 1) which includes all data.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 6)
    alpha : float, optional
        Transparency level for fill. Default is 0.4
    common_norm : bool, optional
        Whether to normalize across all groups. Default is False
    n_ticks : int, optional
        Number of ticks to show on x-axis. Default is 5
    custom_colors : list, optional
        List of hex color codes to use for the plot. If None, uses seaborn default colors.
        If insufficient colors provided, random colors will be added.
        If too many colors provided, only the first n will be used.

    Returns:
    --------
    tuple: (fig, ax, analysis_dict)
        - fig: matplotlib figure object
        - ax: matplotlib axes object
        - analysis_dict: comprehensive analysis dictionary
    """
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Validate inputs
    if not isinstance(percentile_range, (list, tuple)) or len(percentile_range) != 2:
        raise ValueError("percentile_range must be a list or tuple with exactly 2 elements (lower_pct, upper_pct)")

    if numeric_column not in data.columns:
        raise ValueError(f"Column '{numeric_column}' not found in data")

    if categorical_column not in data.columns:
        raise ValueError(f"Column '{categorical_column}' not found in data")

    lower_pct, upper_pct = percentile_range

    if not (0 <= lower_pct <= 1) or not (0 <= upper_pct <= 1):
        raise ValueError("Percentile values must be between 0 and 1")

    if lower_pct >= upper_pct:
        raise ValueError("Lower percentile must be less than upper percentile")

    # Apply percentile-based filtering using the existing function
    filtered_data = apply_outlier_filtering(data, [numeric_column], percentile_range)

    # Remove rows where the numeric column is NaN after filtering
    filtered_data = filtered_data.dropna(subset=[numeric_column])

    if len(filtered_data) == 0:
        print(f"Warning: No data found after applying percentile range [{lower_pct}, {upper_pct}]")
        return None, None, None

    # Calculate global statistics
    global_mean = filtered_data[numeric_column].mean()
    global_median = filtered_data[numeric_column].median()
    total_companies = len(filtered_data)

    # Calculate the actual range after filtering for axis limits
    min_val = filtered_data[numeric_column].min()
    max_val = filtered_data[numeric_column].max()

    # Get unique categories sorted by mean of numeric column
    categories = filtered_data.groupby(categorical_column)[numeric_column].mean().sort_values().index.tolist()
    n_categories = len(categories)

    if n_categories == 0:
        raise ValueError("No categories found in data")

    # Initialize analysis dictionary
    analysis_dict = {
        "global_analysis": {
            "global_mean": global_mean,
            "global_median": global_median,
            "total_companies": total_companies,
        },
        "categories_analysis": {},
        "insights": []
    }

    # Analyze each category
    for category in categories:
        subset = filtered_data[filtered_data[categorical_column] == category].copy()

        if len(subset) == 0:
            continue

        # Basic statistics
        cat_mean = subset[numeric_column].mean()
        cat_median = subset[numeric_column].median()
        cat_total_companies = len(subset)

        # Distance to global mean
        distance_absolute = (cat_mean - global_mean)

        # Calculate 80% concentration
        subset_sorted = subset.sort_values(numeric_column, ascending=False).copy()
        subset_sorted['cumsum'] = subset_sorted[numeric_column].cumsum()
        total_value = subset_sorted[numeric_column].sum()
        target_value = total_value * 0.8

        # Find how many companies represent 80% or more
        companies_80_percent = 0
        if total_value > 0:
            companies_80_percent = len(subset_sorted[subset_sorted['cumsum'] <= target_value]) + 1
            companies_80_percent = min(companies_80_percent, cat_total_companies)

        concentration_ratio = companies_80_percent / cat_total_companies if cat_total_companies > 0 else 0

        # Tercile analysis
        cat_min = subset[numeric_column].min()
        cat_max = subset[numeric_column].max()
        cat_range = cat_max - cat_min

        if cat_range > 0:
            tercile_1_cutoff = cat_min + (cat_range / 3)
            tercile_2_cutoff = cat_min + (2 * cat_range / 3)

            companies_tercile_1 = len(subset[subset[numeric_column] <= tercile_1_cutoff])
            companies_tercile_2 = len(subset[(subset[numeric_column] > tercile_1_cutoff) &
                                           (subset[numeric_column] <= tercile_2_cutoff)])
            companies_tercile_3 = len(subset[subset[numeric_column] > tercile_2_cutoff])
        else:
            # If all values are the same
            tercile_1_cutoff = cat_min
            tercile_2_cutoff = cat_min
            companies_tercile_1 = cat_total_companies
            companies_tercile_2 = 0
            companies_tercile_3 = 0

        # Store category analysis
        analysis_dict["categories_analysis"][category] = {
            "mean": cat_mean,
            "median": cat_median,
            "distance_to_global_mean": distance_absolute,
            "companies_for_80_percent": companies_80_percent,
            "total_companies": cat_total_companies,
            "concentration_ratio": concentration_ratio,
            "tercile_analysis": {
                "min_value": cat_min,
                "max_value": cat_max,
                "tercile_1_cutoff": tercile_1_cutoff,
                "tercile_2_cutoff": tercile_2_cutoff,
                "companies_tercile_1": companies_tercile_1,
                "companies_tercile_2": companies_tercile_2,
                "companies_tercile_3": companies_tercile_3
            }
        }

    # Generate insights using the same function as the original
    insights = generate_distribution_insights(analysis_dict, numeric_column, categorical_column)
    analysis_dict["insights"] = insights
    if openai is True:
      insights = generate_strategic_insights(
          categorical_column, None,numeric_column,insights,
          API_KEY=API_KEY
      )

    # Handle custom colors
    if custom_colors is not None:
        # Validate that all provided colors are valid hex codes
        valid_colors = []
        for color in custom_colors:
            if isinstance(color, str) and color.startswith('#') and len(color) == 7:
                try:
                    # Test if it's a valid hex color
                    int(color[1:], 16)
                    valid_colors.append(color)
                except ValueError:
                    print(f"Warning: '{color}' is not a valid hex color. Skipping.")
            else:
                print(f"Warning: '{color}' is not a valid hex color format. Skipping.")

        # If we have fewer colors than categories, add random colors
        if len(valid_colors) < n_categories:
            needed_colors = n_categories - len(valid_colors)
            print(f"Adding {needed_colors} random colors to complete the palette.")

            for _ in range(needed_colors):
                # Generate random hex color
                random_color = f"#{random.randint(0, 0xFFFFFF):06x}"
                valid_colors.append(random_color)

        # If we have more colors than needed, use only the first n
        elif len(valid_colors) > n_categories:
            valid_colors = valid_colors[:n_categories]
            print(f"Using only the first {n_categories} colors from the provided list.")

        # Set the custom palette
        palette = valid_colors
    else:
        # Use seaborn default palette
        palette = None

    # Create single figure with transparent background
    fig, ax = plt.subplots(figsize=figsize)
    ax.patch.set_alpha(0)      # Make plot area background transparent
    fig.patch.set_alpha(0)     # Make figure background transparent

    # Create single KDE plot with custom colors if provided
    sns.kdeplot(
        data=filtered_data,
        x=numeric_column,
        hue=categorical_column,
        fill=True,
        common_norm=common_norm,
        alpha=alpha,
        ax=ax,
        palette=palette
    )

    # Customize legend - remove the frame/box
    legend = ax.get_legend()
    if legend:
        legend.set_frame_on(False)

    # Hide Y-axis and customize appearance
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set exact limits for X-axis based on filtered data
    ax.set_xlim(min_val, max_val)

    # Apply formatting based on column name (using existing functions)
    if 'eur' in numeric_column.lower():
        # Apply custom EUR formatting to x-axis
        ax = format_eur_axis(ax, min_val, max_val, n_ticks)
    else:
        # Apply regular numeric formatting
        ax = format_regular_axis(ax, min_val, max_val, n_ticks)

    # Set X-axis label (no title as requested)
    ax.set_xlabel(numeric_column.replace('_', ' ').title())

    # Remove any title that might have been set
    ax.set_title('')

    plt.tight_layout()
    return fig, ax, analysis_dict