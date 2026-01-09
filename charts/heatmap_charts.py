import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def create_custom_colormap_with_purple():
    """
    Creates a custom colormap that goes from light purple to dark purple

    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap with purple gradient (light to dark)
    """
    import matplotlib.colors as mcolors
    import numpy as np

    # Purple gradient from light to dark
    purple_colors = [
        '#f3e5ff',  # Very light purple (low values)
        '#e6ccff',  # Light purple
        '#d9b3ff',  # Medium light purple
        '#cc99ff',  # Medium purple
        '#b380ff',  # Medium purple
        '#9966ff',  # Medium dark purple
        '#7D4BEB',  # Your purple (medium-high values)
        '#6633cc',  # Dark purple
        '#4c0080',  # Very dark purple
        '#330066'   # Very dark purple (high values)
    ]

    # Create custom colormap
    n_bins = 256
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_purple', purple_colors, N=n_bins)

    return custom_cmap

def create_divergent_purple_colormap():
    """
    Creates a divergent colormap for negative and positive values:
    - Negative values: Dark red to light purple
    - Positive values: Light purple to dark purple

    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
        Custom divergent colormap
    """
    import matplotlib.colors as mcolors
    import numpy as np

    # Divergent colors: Red (negative) -> Light Purple (zero) -> Dark Purple (positive)
    divergent_colors = [
        '#8B0000',  # Dark red (very negative)
        '#CD5C5C',  # Medium red (negative)
        '#F08080',  # Light red (slightly negative)
        '#E6E6FA',  # Very light purple (near zero)
        '#D8BFD8',  # Light purple (zero/transition)
        '#DDA0DD',  # Medium light purple (slightly positive)
        '#BA55D3',  # Medium purple (positive)
        '#9932CC',  # Dark orchid (positive)
        '#7D4BEB',  # Your purple (high positive)
        '#4B0082',  # Indigo (very positive)
        '#2E0054'   # Very dark purple (extremely positive)
    ]

    # Create custom colormap
    n_bins = 256
    divergent_cmap = mcolors.LinearSegmentedColormap.from_list('divergent_purple', divergent_colors, N=n_bins)

    return divergent_cmap

def create_gaussian_density_heatmap(x_data, y_data, color_data, grid_resolution=200,
                                   bandwidth='auto', kernel='gaussian',
                                   influence_decay=2.0, normalize_fields=True):
    """
    Creates a smooth heatmap using Gaussian density fields where each data point
    generates a smooth "field of influence" that decays gradually.

    This creates a truly continuous surface where colors change smoothly between points,
    like heat diffusion or density fields.

    Parameters:
    -----------
    x_data, y_data : array-like
        Normalized coordinates (0-100) of data points
    color_data : array-like
        Real color values at data points (not normalized)
    grid_resolution : int, default 200
        Resolution of output grid (higher = smoother but slower)
    bandwidth : float or 'auto', default 'auto'
        Standard deviation of Gaussian kernels (influence radius)
        - 'auto': Calculated as percentage of data range
        - float: Direct value (typical range: 2.0-20.0)
    kernel : str, default 'gaussian'
        Type of kernel function: 'gaussian', 'exponential', 'linear'
    influence_decay : float, default 2.0
        Controls how quickly influence decays with distance
        - Lower values (1.0): Wider influence, more blending
        - Higher values (3.0): Sharper influence, more defined regions
    normalize_fields : bool, default True
        Whether to normalize overlapping influences

    Returns:
    --------
    tuple: (X_grid, Y_grid, Z_grid) - meshgrid coordinates and smooth color field
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    # Convert to numpy arrays
    x_points = np.asarray(x_data)
    y_points = np.asarray(y_data)
    color_values = np.asarray(color_data)

    # Create output grid
    x_grid = np.linspace(0, 100, grid_resolution)
    y_grid = np.linspace(0, 100, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Flatten grid for distance calculations
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    data_points = np.column_stack([x_points, y_points])

    # Auto-calculate bandwidth if needed
    if bandwidth == 'auto':
        # Use percentage of data range as bandwidth
        x_range = np.ptp(x_points)  # Peak-to-peak (max - min)
        y_range = np.ptp(y_points)
        avg_range = (x_range + y_range) / 2
        bandwidth = avg_range * 0.15  # 15% of average data range
        print(f"Auto-calculated bandwidth: {bandwidth:.2f}")

    # Calculate distances from each grid point to all data points
    distances = cdist(grid_points, data_points)

    # Create influence weights based on kernel type
    if kernel == 'gaussian':
        # Gaussian kernel: exp(-0.5 * (d/bandwidth)^2)
        weights = np.exp(-0.5 * (distances / bandwidth) ** influence_decay)
    elif kernel == 'exponential':
        # Exponential kernel: exp(-d/bandwidth)
        weights = np.exp(-distances / bandwidth * influence_decay)
    elif kernel == 'linear':
        # Linear decay with cutoff
        cutoff_distance = bandwidth * 3  # 3x bandwidth cutoff
        weights = np.maximum(0, 1 - (distances / cutoff_distance) ** influence_decay)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    # Calculate weighted color values for each grid point
    if normalize_fields:
        # Normalize weights so they sum to 1 for each grid point
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        # Avoid division by zero
        weight_sums = np.where(weight_sums == 0, 1, weight_sums)
        normalized_weights = weights / weight_sums

        # Calculate weighted average of colors
        Z_values = np.sum(normalized_weights * color_values, axis=1)
    else:
        # Use raw weighted sum (can create higher values where fields overlap)
        Z_values = np.sum(weights * color_values, axis=1)

    # Reshape back to grid shape
    Z_grid = Z_values.reshape(X_grid.shape)

    print(f"Gaussian density heatmap created: {len(x_points)} points -> {grid_resolution}x{grid_resolution} smooth field")
    print(f"Kernel: {kernel}, Bandwidth: {bandwidth:.2f}, Decay: {influence_decay:.1f}")

    return X_grid, Y_grid, Z_grid

def validate_insight_squares(insight_squares, df):
    """
    Validates the insight_squares parameter structure and content.
    New format: {'area_name': [x_left, x_right, y_bottom, y_top], 'param': 'column_name'}

    Parameters:
    -----------
    insight_squares : dict
        Dictionary with area definitions and param column
    df : pandas.DataFrame
        DataFrame to validate column existence

    Returns:
    --------
    dict: Validated insight_squares

    Raises:
    -------
    ValueError: If validation fails
    """
    if not isinstance(insight_squares, dict):
        raise ValueError("insight_squares must be a dictionary")

    if len(insight_squares) > 4:
        raise ValueError("insight_squares can have maximum 4 keys")

    if 'param' not in insight_squares:
        raise ValueError("insight_squares must contain 'param' key")

    param_col = insight_squares['param']
    if not isinstance(param_col, str):
        raise ValueError("'param' value must be a string (column name)")

    if param_col not in df.columns:
        raise ValueError(f"Column '{param_col}' not found in DataFrame")

    if not pd.api.types.is_numeric_dtype(df[param_col]):
        raise ValueError(f"Column '{param_col}' must be numeric")

    # Validate area definitions
    area_keys = [k for k in insight_squares.keys() if k != 'param']
    if len(area_keys) > 3:
        raise ValueError("Maximum 3 area definitions allowed (excluding 'param')")

    for area_name, coords in insight_squares.items():
        if area_name == 'param':
            continue

        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError(f"Area '{area_name}' must have exactly 4 values [x_left, x_right, y_bottom, y_top]")

        if not all(isinstance(val, (int, float)) for val in coords):
            raise ValueError(f"Area '{area_name}' coordinates must be numeric")

        x_left, x_right, y_bottom, y_top = coords

        # Validate coordinate logic
        if x_left >= x_right:
            raise ValueError(f"Area '{area_name}': x_left ({x_left}) must be less than x_right ({x_right})")

        if y_bottom >= y_top:
            raise ValueError(f"Area '{area_name}': y_bottom ({y_bottom}) must be less than y_top ({y_top})")

        # Validate bounds (0-100)
        if not all(0 <= coord <= 100 for coord in coords):
            raise ValueError(f"Area '{area_name}' coordinates must be between 0 and 100")

    return insight_squares

def validate_rectangle_bounds(x_left, x_right, y_bottom, y_top):
    """
    Ensures rectangle coordinates stay within graph bounds (0-100) and are valid.

    Parameters:
    -----------
    x_left, x_right, y_bottom, y_top : float
        Rectangle coordinates

    Returns:
    --------
    tuple: (x_left, x_right, y_bottom, y_top) - validated coordinates
    """
    # Ensure coordinates are within bounds
    x_left = max(0, min(x_left, 100))
    x_right = max(0, min(x_right, 100))
    y_bottom = max(0, min(y_bottom, 100))
    y_top = max(0, min(y_top, 100))

    # Ensure logical ordering
    if x_left >= x_right:
        x_left, x_right = min(x_left, x_right), max(x_left, x_right)

    if y_bottom >= y_top:
        y_bottom, y_top = min(y_bottom, y_top), max(y_bottom, y_top)

    return x_left, x_right, y_bottom, y_top

def analyze_areas(data, insight_squares):
    """
    Analyzes data within defined rectangular areas.
    New format: {'area_name': [x_left, x_right, y_bottom, y_top], 'param': 'column_name'}

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with normalized columns and original data
    insight_squares : dict
        Area definitions and param column

    Returns:
    --------
    dict: Analysis results for each area including total dataset analysis
    """
    param_col = insight_squares['param']
    analysis = {}

    # Calculate total dataset statistics
    total_count = len(data)
    total_analysis = {
        "company_count": total_count,
        "company_percentage": 100.0,  # Total is always 100%
        f"{data.columns[0]} Mean": data.iloc[:, 0].mean(),  # Original X column
        f"{data.columns[1]} Mean": data.iloc[:, 1].mean(),  # Original Y column
        f"{param_col} Mean": data[param_col].mean(),
        "coordinates": "entire_dataset"
    }
    analysis["Total"] = total_analysis

    # Analyze each defined area
    for area_name, coords in insight_squares.items():
        if area_name == 'param':
            continue

        x_left, x_right, y_bottom, y_top = coords
        x_left, x_right, y_bottom, y_top = validate_rectangle_bounds(x_left, x_right, y_bottom, y_top)

        # Filter data within rectangle bounds using normalized coordinates
        mask = ((data['x_normalized'] >= x_left) &
                (data['x_normalized'] <= x_right) &
                (data['y_normalized'] >= y_bottom) &
                (data['y_normalized'] <= y_top))

        area_data = data[mask]
        area_count = len(area_data)
        area_percentage = (area_count / total_count * 100) if total_count > 0 else 0

        if area_count == 0:
            analysis[area_name] = {
                "company_count": 0,
                "company_percentage": 0.0,
                f"{data.columns[0]} Mean": None,
                f"{data.columns[1]} Mean": None,
                f"{param_col} Mean": None,
                "coordinates": {
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_bottom": y_bottom,
                    "y_top": y_top
                }
            }
        else:
            # Calculate means using original (non-normalized) values
            analysis[area_name] = {
                "company_count": area_count,
                "company_percentage": area_percentage,
                f"{data.columns[0]} Mean": area_data.iloc[:, 0].mean(),  # Original X column
                f"{data.columns[1]} Mean": area_data.iloc[:, 1].mean(),  # Original Y column
                f"{param_col} Mean": area_data[param_col].mean(),
                "coordinates": {
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_bottom": y_bottom,
                    "y_top": y_top
                }
            }

    return analysis

def generate_insights(areas_analysis, param_col, x_col, y_col):
    """
    Generates comparative insights between areas and between each area and total dataset.

    Parameters:
    -----------
    areas_analysis : dict
        Analysis results for each area including "Total"
    param_col : str
        Name of the parameter column (color variable)
    x_col : str
        Name of the X-axis column
    y_col : str
        Name of the Y-axis column

    Returns:
    --------
    list: Generated comparative insights
    """
    insights = []

    # Get total data for comparison
    total_data = areas_analysis.get("Total", {})

    # Get area data (excluding Total)
    area_data = {name: data for name, data in areas_analysis.items()
                 if name != "Total" and data["company_count"] > 0 and data[f"{param_col} Mean"] is not None}

    if len(area_data) == 0:
        insights.append("No areas contain data for analysis")
        return insights

    area_names = list(area_data.keys())

    # Helper function to get safe ratio
    def get_ratio(val1, val2):
        if abs(val2) < 0.001:
            return None
        return abs(val1 / val2)

    # 1. COMPARISONS BETWEEN AREAS (if more than 1 area)
    if len(area_names) >= 2:
        # Compare param_col between areas
        param_values = {name: data[f"{param_col} Mean"] for name, data in area_data.items()}

        # Find highest and lowest for param_col
        highest_param_area = max(param_values.keys(), key=lambda k: param_values[k])
        lowest_param_area = min(param_values.keys(), key=lambda k: param_values[k])

        if highest_param_area != lowest_param_area:
            ratio = get_ratio(param_values[highest_param_area], param_values[lowest_param_area])
            if ratio and ratio > 1.2:
                insights.append(f"The mean of {param_col} in {highest_param_area} area is {ratio:.1f}x higher than the mean of {param_col} in {lowest_param_area} area")

        # Compare x_col between areas
        x_values = {name: data[f"{x_col} Mean"] for name, data in area_data.items()}
        highest_x_area = max(x_values.keys(), key=lambda k: x_values[k])
        lowest_x_area = min(x_values.keys(), key=lambda k: x_values[k])

        if highest_x_area != lowest_x_area:
            ratio = get_ratio(x_values[highest_x_area], x_values[lowest_x_area])
            if ratio and ratio > 1.2:
                insights.append(f"The mean of {x_col} in {highest_x_area} area is {ratio:.1f}x higher than the mean of {x_col} in {lowest_x_area} area")

        # Compare y_col between areas
        y_values = {name: data[f"{y_col} Mean"] for name, data in area_data.items()}
        highest_y_area = max(y_values.keys(), key=lambda k: y_values[k])
        lowest_y_area = min(y_values.keys(), key=lambda k: y_values[k])

        if highest_y_area != lowest_y_area:
            ratio = get_ratio(y_values[highest_y_area], y_values[lowest_y_area])
            if ratio and ratio > 1.2:
                insights.append(f"The mean of {y_col} in {highest_y_area} area is {ratio:.1f}x higher than the mean of {y_col} in {lowest_y_area} area")

        # Compare company counts between areas
        company_counts = {name: data["company_count"] for name, data in area_data.items()}
        most_populated = max(company_counts.keys(), key=lambda k: company_counts[k])
        least_populated = min(company_counts.keys(), key=lambda k: company_counts[k])

        if most_populated != least_populated:
            ratio = get_ratio(company_counts[most_populated], company_counts[least_populated])
            if ratio and ratio > 1.2:
                insights.append(f"The number of companies in {most_populated} area is {ratio:.1f}x higher than the number of companies in {least_populated} area ({company_counts[most_populated]} vs {company_counts[least_populated]})")

    # 2. COMPARISONS BETWEEN EACH AREA AND TOTAL
    if total_data and f"{param_col} Mean" in total_data:
        total_param = total_data[f"{param_col} Mean"]
        total_x = total_data[f"{x_col} Mean"]
        total_y = total_data[f"{y_col} Mean"]

        for area_name, area_info in area_data.items():
            # Compare param_col with total
            area_param = area_info[f"{param_col} Mean"]
            ratio = get_ratio(area_param, total_param)
            if ratio and ratio > 1.2:
                insights.append(f"The mean of {param_col} in {area_name} area is {ratio:.1f}x higher than the mean of {param_col} in all the data")
            elif ratio and ratio < 0.8:
                inverse_ratio = get_ratio(total_param, area_param)
                if inverse_ratio:
                    insights.append(f"The mean of {param_col} in all the data is {inverse_ratio:.1f}x higher than the mean of {param_col} in {area_name} area")

            # Compare x_col with total
            area_x = area_info[f"{x_col} Mean"]
            ratio = get_ratio(area_x, total_x)
            if ratio and ratio > 1.2:
                insights.append(f"The mean of {x_col} in {area_name} area is {ratio:.1f}x higher than the mean of {x_col} in all the data")
            elif ratio and ratio < 0.8:
                inverse_ratio = get_ratio(total_x, area_x)
                if inverse_ratio:
                    insights.append(f"The mean of {x_col} in all the data is {inverse_ratio:.1f}x higher than the mean of {x_col} in {area_name} area")

            # Compare y_col with total
            area_y = area_info[f"{y_col} Mean"]
            ratio = get_ratio(area_y, total_y)
            if ratio and ratio > 1.2:
                insights.append(f"The mean of {y_col} in {area_name} area is {ratio:.1f}x higher than the mean of {y_col} in all the data")
            elif ratio and ratio < 0.8:
                inverse_ratio = get_ratio(total_y, area_y)
                if inverse_ratio:
                    insights.append(f"The mean of {y_col} in all the data is {inverse_ratio:.1f}x higher than the mean of {y_col} in {area_name} area")

    # Limit insights to avoid overwhelm
    if len(insights) > 12:
        insights = insights[:12]

    return insights

def create_smooth_density_heatmap(df, x_col, y_col, color_col,
                                  figsize=(12, 8), percentile_range=None,
                                  cmap='viridis', show_scatter_points=True,
                                  # Insight squares parameter (NEW FORMAT)
                                  insight_squares=None,
                                  # Density field parameters
                                  grid_resolution=200,
                                  bandwidth='auto',
                                  kernel='gaussian',
                                  influence_decay=2.0,
                                  normalize_fields=True,
                                  # Visual parameters
                                  interpolation='bilinear',
                                  transparent_bg=True,
                                  scatter_alpha=0.7,
                                  scatter_size=30,
                                  openai=False):
    """
    Creates a smooth continuous heatmap using Gaussian density fields.
    Each data point generates a smooth "field of influence" that decays gradually,
    creating truly continuous color transitions without abrupt changes.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with the data
    x_col : str
        Name of the numeric column for X-axis (will be normalized 0-100)
    y_col : str
        Name of the numeric column for Y-axis (will be normalized 0-100)
    color_col : str
        Name of the numeric column that determines the color (keeps REAL values)
    figsize : tuple, default (12, 8)
        Figure size (width, height)
    percentile_range : tuple, optional
        Tuple (lower_percentile, upper_percentile) to filter outliers
    cmap : str, default 'viridis'
        Matplotlib colormap. Use 'custom_purple', 'divergent_purple', or 'auto'
    show_scatter_points : bool, default True
        Whether to overlay original scatter points on the heatmap
    insight_squares : dict, optional
        Dictionary defining analysis areas and parameter column
        NEW FORMAT: {'Area1':[x_left, x_right, y_bottom, y_top], 'Area2':[x_left, x_right, y_bottom, y_top], 'param':'column_name'}
        Example: {'High':[70, 90, 60, 80], 'Low':[10, 30, 20, 40], 'param':'EBITDA EUR 2024'}

    # DENSITY FIELD PARAMETERS:
    grid_resolution : int, default 200
        Grid resolution (higher = smoother but slower)
        Range: 100-500 (200-300 recommended)
    bandwidth : float or 'auto', default 'auto'
        Influence radius of each point
        - 'auto': Auto-calculated (15% of data range)
        - 2.0-5.0: Narrow influence (sharp transitions)
        - 5.0-15.0: Medium influence (balanced)
        - 15.0-30.0: Wide influence (very smooth)
    kernel : str, default 'gaussian'
        Type of influence decay: 'gaussian', 'exponential', 'linear'
    influence_decay : float, default 2.0
        Speed of influence decay
        - 1.0-1.5: Slow decay (wide, smooth influence)
        - 2.0-2.5: Medium decay (balanced)
        - 3.0+: Fast decay (sharp, localized influence)
    normalize_fields : bool, default True
        Whether to normalize overlapping influences

    # VISUAL PARAMETERS:
    interpolation : str, default 'bilinear'
        Final rendering interpolation
    transparent_bg : bool, default True
        Transparent background
    scatter_alpha : float, default 0.7
        Transparency of scatter points (0-1)
    scatter_size : int, default 30
        Size of scatter points

    Returns:
    --------
    If insight_squares is None:
        matplotlib.figure.Figure: Matplotlib figure object
    If insight_squares is provided:
        tuple: (matplotlib.figure.Figure, dict) - Figure and analysis JSON
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    # Validate insight_squares if provided
    if insight_squares is not None:
        insight_squares = validate_insight_squares(insight_squares, df)

    # Column validation - include param column if insight_squares provided
    required_cols = [x_col, y_col, color_col]
    if insight_squares is not None:
        required_cols.append(insight_squares['param'])

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")

    # Create copy and clean data
    data = df[required_cols].copy().dropna()

    if len(data) == 0:
        raise ValueError("No data remaining after removing NaN values")

    # Apply outlier filter if specified
    if percentile_range is not None:
        lower_pct, upper_pct = percentile_range

        for col in required_cols:
            q_lower = data[col].quantile(lower_pct)
            q_upper = data[col].quantile(upper_pct)
            mask = (data[col] >= q_lower) & (data[col] <= q_upper)
            data = data[mask]

        if len(data) == 0:
            raise ValueError("No data remaining after outlier filtering")

        print(f"Outlier filtering applied. Remaining data points: {len(data)}")

    # Normalize X and Y axes from 0 to 100 (for positioning) and add to DataFrame
    def normalize_to_100(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(50, index=series.index)
        return ((series - min_val) / (max_val - min_val)) * 100

    data['x_normalized'] = normalize_to_100(data[x_col])
    data['y_normalized'] = normalize_to_100(data[y_col])

    # Keep color values REAL (no normalization)
    color_real = data[color_col].copy()

    # Get scaling and formatting for the color variable
    scale, suffix = get_scale_and_format_eur(color_real.values, color_col)

    # CREATE SMOOTH DENSITY FIELD
    X_grid, Y_grid, Z_grid = create_gaussian_density_heatmap(
        data['x_normalized'].values, data['y_normalized'].values, color_real.values,
        grid_resolution=grid_resolution,
        bandwidth=bandwidth,
        kernel=kernel,
        influence_decay=influence_decay,
        normalize_fields=normalize_fields
    )

    # Create the figure
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)

    # Set transparent background if requested
    if transparent_bg:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    # Check for custom colormaps
    if cmap == 'custom_purple' or cmap == 'purple':
        cmap = create_custom_colormap_with_purple()
    elif cmap == 'divergent_purple' or (cmap == 'auto' and color_real.min() < 0):
        cmap = create_divergent_purple_colormap()

    # Determine vmin and vmax for color scaling
    color_min = color_real.min()
    color_max = color_real.max()

    # For divergent colormaps, center around zero if there are negative values
    if color_min < 0 and color_max > 0 and (isinstance(cmap, type(create_divergent_purple_colormap())) or
                                           (isinstance(cmap, str) and 'divergent' in cmap)):
        abs_max = max(abs(color_min), abs(color_max))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = color_min, color_max

    # Create the smooth density heatmap
    im = ax.imshow(Z_grid, extent=[0, 100, 0, 100], origin='lower',
                   cmap=cmap, interpolation=interpolation, aspect='equal',
                   vmin=vmin, vmax=vmax)

    # Optionally overlay original scatter points
    if show_scatter_points:
        scatter = ax.scatter(data['x_normalized'], data['y_normalized'], c=color_real,
                            cmap=cmap, s=scatter_size, alpha=scatter_alpha,
                            edgecolors='white', linewidth=0.5,
                            vmin=vmin, vmax=vmax, zorder=10)

    # Draw insight rectangles if provided
    if insight_squares is not None:
        from matplotlib.patches import Rectangle

        for area_name, coords in insight_squares.items():
            if area_name == 'param':
                continue

            x_left, x_right, y_bottom, y_top = coords
            x_left, x_right, y_bottom, y_top = validate_rectangle_bounds(x_left, x_right, y_bottom, y_top)

            # Draw rectangle
            width = x_right - x_left
            height = y_top - y_bottom

            rect = Rectangle((x_left, y_bottom), width, height,
                           linewidth=1, edgecolor='black',
                           facecolor='none', alpha=1.0, zorder=15)
            ax.add_patch(rect)

            # Add label
            ax.text(x_left + width/2, y_top + 1, area_name,
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color='black', zorder=20)

    # Configure axes (X and Y remain normalized 0-100 for positioning)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Axis labels
    x_label = x_col.replace('_', ' ').title() + ' Normalized '
    y_label = y_col.replace('_', ' ').title() + ' Normalized'

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Title
    title = f'Smooth Density Heatmap: {color_col.replace("_", " ").title()}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Configure colorbar with REAL values and proper formatting
    cbar = plt.colorbar(im, ax=ax)

    # Create simple column name title for colorbar
    color_label = color_col.replace('_', ' ').title()

    # Add scale suffix with proper unit formatting
    if suffix and 'EUR' in color_col.upper():
        color_label += f' ({suffix} €)'
    elif suffix and ('GROWTH' in color_col.upper() or 'CAGR' in color_col.upper() or 'MARGIN' in color_col.upper()):
        color_label += f' ({suffix} %)'
    elif suffix:
        color_label += f' ({suffix})'

    cbar.set_label(color_label, rotation=270, labelpad=20, fontweight='bold')

    # Format colorbar ticks to show proper scaling
    def format_colorbar_ticks(x, pos):
        """Custom formatter for colorbar ticks"""
        if 'EUR' in color_col.upper():
            return format_eur_axis_single(x, scale, suffix)
        elif ('GROWTH' in color_col.upper() or 'CAGR' in color_col.upper() or 'MARGIN' in color_col.upper()):
            return f'{x:.1f}%'
        else:
            if suffix == 'Bn':
                return f'{x/1e9:.1f}'
            elif suffix == 'Mn':
                return f'{x/1e6:.1f}'
            elif suffix == 'K':
                return f'{x/1e3:.1f}'
            else:
                return f'{x:.1f}'

    def format_eur_axis_single(x, scale, suffix):
        """Format single EUR value for colorbar"""
        if x < 0:
            sign = '-'
            x = abs(x)
        else:
            sign = ''

        if suffix == 'Bn':
            return f'{sign}{x/1e9:.1f}'
        elif suffix == 'Mn':
            return f'{sign}{x/1e6:.1f}'
        elif suffix == 'K':
            return f'{sign}{x/1e3:.1f}'
        else:
            return f'{sign}{x:.0f}'

    # Apply custom formatter to colorbar
    from matplotlib.ticker import FuncFormatter
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_colorbar_ticks))

    # Axis styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Optional grid (subtle)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Configure ticks to show as percentages (only for X and Y axes)
    ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_yticks(np.linspace(0, 100, 11))
    ax.set_xticklabels([f'{int(x)}%' for x in np.linspace(0, 100, 11)])
    ax.set_yticklabels([f'{int(y)}%' for y in np.linspace(0, 100, 11)])

    plt.tight_layout()

    # Additional information text
    info_text = f'Data points: {len(data)}'
    if percentile_range:
        info_text += f' | Outlier filter: {percentile_range[0]:.1%}-{percentile_range[1]:.1%}'

    # Add density field info
    density_info = []
    density_info.append(f'Kernel: {kernel}')
    if bandwidth == 'auto':
        actual_bandwidth = (np.ptp(data['x_normalized']) + np.ptp(data['y_normalized'])) / 2 * 0.15
        density_info.append(f'Bandwidth: auto ({actual_bandwidth:.1f})')
    else:
        density_info.append(f'Bandwidth: {bandwidth}')
    density_info.append(f'Resolution: {grid_resolution}²')

    info_text += f' | {", ".join(density_info)}'

    # Add color range info
    try:
        formatted_min = format_value_consistent(color_min, scale, suffix, color_col, 'mean')
        formatted_max = format_value_consistent(color_max, scale, suffix, color_col, 'mean')
        info_text += f' | Color range: {formatted_min} to {formatted_max}'
    except:
        info_text += f' | Color range: {color_min:.2f} to {color_max:.2f}'

    print(info_text)

    # Generate analysis JSON
    analysis_json = {}

    if insight_squares is not None:
        # Perform areas analysis
        areas_analysis = analyze_areas(data, insight_squares)
        insights = generate_insights(areas_analysis, insight_squares['param'], x_col, y_col)

        analysis_json = {
            "areas_analysis": areas_analysis,
            "insights": insights
        }
        if openai is True:
          analysis_json = generate_strategic_insights(
              y_col, color_col,x_col,analysis_json,
              API_KEY=API_KEY
          )

        return fig, analysis_json
    else:
        # Basic analysis for entire dataset
        analysis_json = {
            "overall_analysis": {
                "company_count": len(data),
                f"{x_col} Mean": data[x_col].mean(),
                f"{y_col} Mean": data[y_col].mean()
            },
            "insights": [
                f"Total dataset contains {len(data)} companies",
                f"Average {x_col.replace('_', ' ').title()}: {data[x_col].mean():.2f}",
                f"Average {y_col.replace('_', ' ').title()}: {data[y_col].mean():.2f}"
            ]
        }
        if openai is True:
          analysis_json = generate_strategic_insights(
              y_col, color_col,x_col,analysis_json,
              API_KEY=API_KEY
          )

        return fig, analysis_json