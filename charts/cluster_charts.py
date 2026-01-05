import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
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

def create_cluster_chart(
    df: pd.DataFrame,
    x_axis_col: str,
    y_axis_col: str,
    categorical_col: str,
    percentiles: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (12, 8),
    delimitation: bool = True,
    show_scatter_points: bool = True,
    smoothing_factor: float = 1.0
):
    """
    Creates a cluster chart showing characteristic zones for each category with colored areas
    and optional contour lines, with controllable smoothing to prevent overfitting.

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
        - 0.3-0.7: Very smooth, broad zones (low overfitting)
        - 0.8-1.2: Balanced smoothing (recommended)
        - 1.3-3.0: Tight zones, follows data closely (higher overfitting risk)

    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object

    Examples:
    ---------
    # Basic cluster chart with default smoothing
    fig = create_cluster_chart(df, 'Revenue EUR 2024', 'EBITDA EUR 2024', 'Industry')

    # Broad, smooth zones (less overfitting)
    fig = create_cluster_chart(df, 'Revenue EUR 2024', 'EBITDA EUR 2024', 'Industry',
                              smoothing_factor=0.5)

    # Tight zones that follow data closely (more overfitting risk)
    fig = create_cluster_chart(df, 'Revenue EUR 2024', 'EBITDA EUR 2024', 'Industry',
                              smoothing_factor=2.0)

    # Very smooth zones for small datasets or noisy data
    fig = create_cluster_chart(df, 'Revenue EUR 2024', 'EBITDA EUR 2024', 'Industry',
                              smoothing_factor=0.3)
    """

    # Validation
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
    essential_cols = [x_axis_col, y_axis_col, categorical_col]
    data = data.dropna(subset=essential_cols)

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
            # Apply smoothing factor directly through bw_method parameter
            # Lower smoothing_factor = more smoothing (larger bandwidth)
            # Higher smoothing_factor = less smoothing (smaller bandwidth)

            if smoothing_factor == 1.0:
                # Use default bandwidth calculation
                kernel = gaussian_kde(positions)
            else:
                # Use custom bandwidth method based on smoothing factor
                # For smoothing_factor < 1.0: increase bandwidth (more smoothing)
                # For smoothing_factor > 1.0: decrease bandwidth (less smoothing)
                custom_bw = lambda kde_obj: kde_obj.scotts_factor() / smoothing_factor
                kernel = gaussian_kde(positions, bw_method=custom_bw)

            # Evaluate on grid
            grid_positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kernel(grid_positions)
            density = density.reshape(xx.shape)

            # Normalize density for consistent visualization across categories
            density = density / density.max()

            # Create filled contours for characteristic zones (light colors)
            base_color = colors[i]
            light_alpha = 0.3  # Light transparency for zones

            # Define threshold level based on smoothing factor
            # This determines the minimum density to include in the zone
            if smoothing_factor <= 0.5:
                # Very smooth: include more area (lower threshold)
                threshold_level = 0.05
            elif smoothing_factor <= 1.0:
                # Moderate smoothing
                threshold_level = 0.1
            elif smoothing_factor <= 1.5:
                # Default/tight
                threshold_level = 0.15
            else:
                # Very tight: include less area (higher threshold)
                threshold_level = 0.25

            # Create filled contours from threshold to maximum
            # This fills the entire zone without holes
            contour_filled = ax.contourf(xx, yy, density,
                                       levels=[threshold_level, density.max()],
                                       colors=[base_color],
                                       alpha=light_alpha)

            # Add contour lines if delimitation is enabled (only outer boundary)
            if delimitation:
                contour_lines = ax.contour(xx, yy, density,
                                         levels=[threshold_level],
                                         colors=[base_color],
                                         alpha=0.8,
                                         linewidths=1.5)

        except ImportError:
            # Fallback: if scipy is not available, use simpler approach with smoothing adjustment
            print("Warning: scipy not available. Using simplified zone representation with smoothing adjustment.")

            # Simple approach: draw ellipse around data points with smoothing control
            from matplotlib.patches import Ellipse

            # Calculate mean and standard deviation
            x_mean, y_mean = x_cat.mean(), y_cat.mean()
            x_std, y_std = x_cat.std(), y_cat.std()

            # Adjust ellipse size based on smoothing factor
            # Lower smoothing_factor = larger ellipse (more smoothing)
            # Higher smoothing_factor = smaller ellipse (less smoothing)
            base_multiplier = 4  # Base multiplier for 2 standard deviations

            if smoothing_factor <= 0.5:
                size_multiplier = base_multiplier * 1.8  # Much larger ellipse
            elif smoothing_factor <= 1.0:
                size_multiplier = base_multiplier * (2.0 - smoothing_factor)  # Moderate adjustment
            else:
                size_multiplier = base_multiplier / (smoothing_factor * 0.8)  # Smaller ellipse

            # Create ellipse representing characteristic zone
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
                      zorder=10)  # Ensure points are on top

    # Create legend
    if show_scatter_points:
        # Legend will be automatically created from scatter plot labels
        ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        # Create custom legend for zones only
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

    # Set axis labels with proper scaling
    x_label = f'{x_axis_col}'
    if x_suffix:
        x_label += f' ({x_suffix})'

    y_label = f'{y_axis_col}'
    if y_suffix:
        y_label += f' ({y_suffix})'

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Create descriptive title with smoothing info
    title_parts = ["Cluster Chart"]

    # Add smoothing info to title
    if smoothing_factor < 0.7:
        title_parts.append("(Smooth Zones)")
    elif smoothing_factor > 1.3:
        title_parts.append("(Tight Zones)")

    if not show_scatter_points:
        title_parts.append("- Zones Only")

    if delimitation:
        title_parts.append("with Contour Lines")

    title_parts.append(f"by {categorical_col}")

    # Set axis limits with some padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    return fig