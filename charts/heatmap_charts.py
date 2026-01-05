import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import scipy for smooth interpolation
try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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

        if not isinstance(coords, list) or len(coords) != 3:
            raise ValueError(f"Area '{area_name}' must have exactly 3 values [x, y, width]")

        if not all(isinstance(val, (int, float)) for val in coords):
            raise ValueError(f"Area '{area_name}' coordinates must be numeric")

        x, y, width = coords
        if width <= 0:
            raise ValueError(f"Area '{area_name}' width must be positive")

        if x < 0 or y < 0 or x > 100 or y > 100:
            raise ValueError(f"Area '{area_name}' coordinates must be between 0 and 100")

    return insight_squares


def validate_square_bounds(x_start, y_start, width):
    """
    Ensures square coordinates stay within graph bounds (0-100).

    Parameters:
    -----------
    x_start, y_start : float
        Starting coordinates
    width : float
        Square width

    Returns:
    --------
    tuple: (x_start, y_start, x_end, y_end) - validated coordinates
    """
    # Ensure starting points are within bounds
    x_start = max(0, min(x_start, 100))
    y_start = max(0, min(y_start, 100))

    # Calculate end points ensuring they don't exceed bounds
    x_end = min(x_start + width, 100)
    y_end = min(y_start + width, 100)

    return x_start, y_start, x_end, y_end


def analyze_areas(data, insight_squares):
    """
    Analyzes data within defined square areas.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with normalized columns and original data
    insight_squares : dict
        Area definitions and param column

    Returns:
    --------
    dict: Analysis results for each area
    """
    param_col = insight_squares['param']
    analysis = {}

    for area_name, coords in insight_squares.items():
        if area_name == 'param':
            continue

        x_start, y_start, width = coords
        x_start, y_start, x_end, y_end = validate_square_bounds(x_start, y_start, width)

        # Filter data within square bounds using normalized coordinates
        mask = ((data['x_normalized'] >= x_start) &
                (data['x_normalized'] <= x_end) &
                (data['y_normalized'] >= y_start) &
                (data['y_normalized'] <= y_end))

        area_data = data[mask]

        if len(area_data) == 0:
            analysis[area_name] = {
                "company_count": 0,
                f"{data.columns[0]} Mean": None,
                f"{data.columns[1]} Mean": None,
                f"{param_col} Mean": None,
                "coordinates": {
                    "x_start": x_start,
                    "y_start": y_start,
                    "x_end": x_end,
                    "y_end": y_end
                }
            }
        else:
            # Calculate means using original (non-normalized) values
            analysis[area_name] = {
                "company_count": len(area_data),
                f"{data.columns[0]} Mean": area_data.iloc[:, 0].mean(),  # Original X column
                f"{data.columns[1]} Mean": area_data.iloc[:, 1].mean(),  # Original Y column
                f"{param_col} Mean": area_data[param_col].mean(),
                "coordinates": {
                    "x_start": x_start,
                    "y_start": y_start,
                    "x_end": x_end,
                    "y_end": y_end
                }
            }

    return analysis


def generate_insights(areas_analysis, param_col, x_col, y_col):
    """
    Generates comprehensive automatic insights from areas analysis including X, Y and param variables.

    Parameters:
    -----------
    areas_analysis : dict
        Analysis results for each area
    param_col : str
        Name of the parameter column (color variable)
    x_col : str
        Name of the X-axis column
    y_col : str
        Name of the Y-axis column

    Returns:
    --------
    list: Generated comprehensive insights
    """
    insights = []
    area_names = list(areas_analysis.keys())

    # Skip if less than 2 areas or any area has no companies
    valid_areas = {name: data for name, data in areas_analysis.items()
                   if data["company_count"] > 0 and data[f"{param_col} Mean"] is not None}

    if len(valid_areas) < 2:
        insights.append(f"Insufficient data for comparative analysis across areas")
        return insights

    # === PARAM VARIABLE INSIGHTS ===
    param_values = {name: data[f"{param_col} Mean"] for name, data in valid_areas.items()}
    highest_param_area = max(param_values.keys(), key=lambda k: param_values[k])
    lowest_param_area = min(param_values.keys(), key=lambda k: param_values[k])

    if highest_param_area != lowest_param_area:
        highest_val = param_values[highest_param_area]
        lowest_val = param_values[lowest_param_area]

        if abs(lowest_val) > 0.001:  # Avoid division by very small numbers
            ratio = abs(highest_val / lowest_val)
            if ratio > 1.5:  # Only report significant differences
                insights.append(f"{param_col}: {highest_param_area} area shows {ratio:.1f}x higher values than {lowest_param_area} area")
        else:
            insights.append(f"{param_col}: {highest_param_area} area shows positive values while {lowest_param_area} area shows near-zero values")

    # === X-AXIS VARIABLE INSIGHTS ===
    x_values = {name: data[f"{x_col} Mean"] for name, data in valid_areas.items()}
    highest_x_area = max(x_values.keys(), key=lambda k: x_values[k])
    lowest_x_area = min(x_values.keys(), key=lambda k: x_values[k])

    if highest_x_area != lowest_x_area:
        highest_x_val = x_values[highest_x_area]
        lowest_x_val = x_values[lowest_x_area]

        if abs(lowest_x_val) > 0.001:
            x_ratio = abs(highest_x_val / lowest_x_val)
            if x_ratio > 1.5:
                insights.append(f"{x_col}: {highest_x_area} area excels with {x_ratio:.1f}x higher values than {lowest_x_area} area")

    # === Y-AXIS VARIABLE INSIGHTS ===
    y_values = {name: data[f"{y_col} Mean"] for name, data in valid_areas.items()}
    highest_y_area = max(y_values.keys(), key=lambda k: y_values[k])
    lowest_y_area = min(y_values.keys(), key=lambda k: y_values[k])

    if highest_y_area != lowest_y_area:
        highest_y_val = y_values[highest_y_area]
        lowest_y_val = y_values[lowest_y_area]

        if abs(lowest_y_val) > 0.001:
            y_ratio = abs(highest_y_val / lowest_y_val)
            if y_ratio > 1.5:
                insights.append(f"{y_col}: {highest_y_area} area leads with {y_ratio:.1f}x higher values than {lowest_y_area} area")

    # === COMPANY DISTRIBUTION INSIGHTS ===
    company_counts = {name: data["company_count"] for name, data in valid_areas.items()}
    most_populated = max(company_counts.keys(), key=lambda k: company_counts[k])
    least_populated = min(company_counts.keys(), key=lambda k: company_counts[k])

    if most_populated != least_populated:
        pop_ratio = company_counts[most_populated] / company_counts[least_populated]
        if pop_ratio > 1.5:
            insights.append(f"Company Distribution: {most_populated} area concentrates {company_counts[most_populated]} companies vs {company_counts[least_populated]} in {least_populated} area ({pop_ratio:.1f}x more)")

    # === MULTI-VARIABLE CORRELATION INSIGHTS ===
    # Check if the same area dominates multiple variables
    dominance_count = {}
    for area in valid_areas.keys():
        dominance_count[area] = 0
        if area == highest_param_area:
            dominance_count[area] += 1
        if area == highest_x_area:
            dominance_count[area] += 1
        if area == highest_y_area:
            dominance_count[area] += 1

    # Find areas that dominate multiple metrics
    multi_dominant = {area: count for area, count in dominance_count.items() if count >= 2}

    if multi_dominant:
        for area, count in multi_dominant.items():
            if count == 3:
                insights.append(f"Strategic Positioning: {area} area demonstrates superior performance across all three dimensions ({param_col}, {x_col}, {y_col})")
            elif count == 2:
                leading_vars = []
                if area == highest_param_area:
                    leading_vars.append(param_col)
                if area == highest_x_area:
                    leading_vars.append(x_col)
                if area == highest_y_area:
                    leading_vars.append(y_col)
                insights.append(f"{area} area shows dual leadership in {' and '.join(leading_vars)}")

    # === PERFORMANCE SPREAD INSIGHTS ===
    # Calculate coefficient of variation (relative dispersion) for each variable
    def calc_coefficient_variation(values_dict):
        values = list(values_dict.values())
        if len(values) < 2:
            return 0
        mean_val = sum(values) / len(values)
        if abs(mean_val) < 0.001:
            return 0
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        return std_dev / abs(mean_val)

    param_cv = calc_coefficient_variation(param_values)
    x_cv = calc_coefficient_variation(x_values)
    y_cv = calc_coefficient_variation(y_values)

    # Identify which variable shows most variation across areas
    cv_vars = [(param_cv, param_col), (x_cv, x_col), (y_cv, y_col)]
    cv_vars.sort(reverse=True, key=lambda x: x[0])

    if cv_vars[0][0] > 0.3:  # Significant variation threshold
        most_varied = cv_vars[0][1]
        insights.append(f"Variability Analysis: {most_varied} shows the highest variation across areas, indicating strong regional differences")

    # === BALANCE VS SPECIALIZATION INSIGHTS ===
    # Check if any area is consistently average (balanced) vs specialized
    area_rankings = {}

    for area in valid_areas.keys():
        rankings = []

        # Rank in param_col
        param_rank = sorted(param_values.items(), key=lambda x: x[1], reverse=True)
        param_position = next(i for i, (name, _) in enumerate(param_rank) if name == area) + 1
        rankings.append(param_position)

        # Rank in x_col
        x_rank = sorted(x_values.items(), key=lambda x: x[1], reverse=True)
        x_position = next(i for i, (name, _) in enumerate(x_rank) if name == area) + 1
        rankings.append(x_position)

        # Rank in y_col
        y_rank = sorted(y_values.items(), key=lambda x: x[1], reverse=True)
        y_position = next(i for i, (name, _) in enumerate(y_rank) if name == area) + 1
        rankings.append(y_position)

        area_rankings[area] = {
            'ranks': rankings,
            'avg_rank': sum(rankings) / len(rankings),
            'rank_variance': sum((r - sum(rankings)/len(rankings))**2 for r in rankings) / len(rankings)
        }

    # Find most balanced area (consistently average ranks, low variance)
    most_balanced = min(area_rankings.items(), key=lambda x: x[1]['rank_variance'])
    if most_balanced[1]['rank_variance'] < 0.5 and len(valid_areas) >= 3:
        insights.append(f"Balanced Performance: {most_balanced[0]} area shows consistent performance across all metrics (balanced strategy)")

    # === STRATEGIC RECOMMENDATIONS ===
    if len(insights) >= 3:
        # Identify the overall best performing area based on combined metrics
        area_scores = {}
        for area in valid_areas.keys():
            score = 0
            if area == highest_param_area:
                score += 3
            if area == highest_x_area:
                score += 2
            if area == highest_y_area:
                score += 2
            if area == most_populated:
                score += 1
            area_scores[area] = score

        best_overall = max(area_scores.items(), key=lambda x: x[1])
        if best_overall[1] >= 4:  # High combined score
            insights.append(f"Strategic Focus: {best_overall[0]} area represents the optimal positioning with superior performance across multiple key metrics")

    # Ensure we don't return too many insights (keep most relevant)
    if len(insights) > 8:
        insights = insights[:8]

    return insights


def create_smooth_density_heatmap(df, x_col, y_col, color_col,
                                  figsize=(12, 8), percentile_range=None,
                                  cmap='viridis', show_scatter_points=True,
                                  # Insight squares parameter
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
                                  scatter_size=30):
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
        Example: {'Low':[20,40,10], 'High':[80,90,10], 'param':'EBITDA EUR 2024'}

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
        insights = generate_insights(areas_analysis, insight_squares['param'],x_col, y_col)

        analysis_json = {
            "areas_analysis": areas_analysis,
            "insights": insights
        }

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

        return fig, analysis_json