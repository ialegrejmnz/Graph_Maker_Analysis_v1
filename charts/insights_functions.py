import pandas as pd
from typing import Dict, Union, Literal
import json
import numpy as np

# Diccionario de estimadores disponibles (mismo que en la función de gráficos)
AVAILABLE_ESTIMATORS = {
    'sum': 'sum',
    'mean': 'mean',
    'median': 'median',
    'count': 'count',
    'std': 'std',
    'var': 'var',
    'min': 'min',
    'max': 'max',
    'first': 'first',
    'last': 'last'
}

def generate_multilevel_aggregations(
    df: pd.DataFrame,
    categorical_col1: str,
    categorical_col2: str,
    numeric_col: str,
    estimator: str = 'sum',
    values: Literal["percentages", "absolute", "both"] = "both",
    precision: int = 2
) -> Dict[str, Union[float, int]]:
    """
    Generate comprehensive multilevel aggregations (rollups/cubes) for two categorical columns
    and one numeric column, returning absolute values and/or percentage breakdowns.

    This function creates all possible aggregation levels using the specified estimator:
    - Individual combinations (cat1 + cat2)
    - Category totals (cat1 totals, cat2 totals)
    - Grand total

    And calculates:
    - Percentage of each combination relative to grand total
    - Composition percentages within each category

    Args:
        df (pd.DataFrame): Input DataFrame containing the data
        categorical_col1 (str): Name of the first categorical column
        categorical_col2 (str): Name of the second categorical column
        numeric_col (str): Name of the numeric column to aggregate
        estimator (str): Statistical estimator to apply. Options: 'sum', 'mean', 'median',
                        'count', 'std', 'var', 'min', 'max', 'first', 'last'. Defaults to 'sum'.
        values (Literal["percentages", "absolute", "both"]): Type of values to include:
            - "percentages": Only percentages and grand total
            - "absolute": Only absolute values
            - "both": Both absolute values and percentages. Defaults to "both".
        precision (int, optional): Number of decimal places for percentages. Defaults to 2.

    Returns:
        Dict[str, Union[float, int]]: JSON-serializable dictionary with descriptive keys
                                    and corresponding values (absolutes and/or percentages)

    Raises:
        ValueError: If specified columns don't exist in DataFrame, invalid values parameter,
                   or invalid estimator
        TypeError: If numeric column contains non-numeric data

    Note:
        - Division by zero scenarios return 0.0 for percentages when denominator is zero.
        - For non-sum estimators, percentages represent relative relationships but may not
          sum to 100% (e.g., max values don't aggregate linearly).
        - Negative values are preserved to maintain data integrity and provide real insights.
        - Now uses pivot table logic internally for better performance.
    """

    # Input validation
    _validate_inputs(df, categorical_col1, categorical_col2, numeric_col, values, estimator)

    # Create working DataFrame with only required columns
    work_df = df[[categorical_col1, categorical_col2, numeric_col]].copy()

    # Generate all aggregation levels using pivot table logic (more efficient)
    aggregations = _create_rollup_table(work_df, categorical_col1, categorical_col2, numeric_col, estimator)

    # Calculate percentages and build result dictionary
    result = _build_result_dictionary(
        aggregations, categorical_col1, categorical_col2, numeric_col, estimator, values, precision
    )

    return result


def _validate_inputs(df: pd.DataFrame, cat_col1: str, cat_col2: str, num_col: str,
                    values: str, estimator: str) -> None:
    """Validate input parameters."""
    required_columns = [cat_col1, cat_col2, num_col]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

    if not pd.api.types.is_numeric_dtype(df[num_col]):
        raise TypeError(f"Column '{num_col}' must contain numeric data")

    valid_values = ["percentages", "absolute", "both"]
    if values not in valid_values:
        raise ValueError(f"Parameter 'values' must be one of {valid_values}, got '{values}'")

    # Define available estimators
    AVAILABLE_ESTIMATORS = {
        'sum': 'sum',
        'mean': 'mean',
        'median': 'median',
        'count': 'count',
        'std': 'std',
        'var': 'var',
        'min': 'min',
        'max': 'max',
        'first': 'first',
        'last': 'last'
    }

    if estimator not in AVAILABLE_ESTIMATORS:
        raise ValueError(f"Invalid estimator '{estimator}'. Options: {list(AVAILABLE_ESTIMATORS.keys())}")


def _create_rollup_table(df: pd.DataFrame, cat_col1: str, cat_col2: str,
                        num_col: str, estimator: str) -> pd.DataFrame:
    """
    Create comprehensive rollup table using PIVOT TABLE LOGIC for better performance.
    Maintains the same output format as the original function.
    """

    # Create working copy - NO conversion to absolute values (preserving real insights)
    work_df = df.copy()

    # 🚀 NEW: Use pivot table for efficient aggregation
    try:
        pivot = pd.pivot_table(
            work_df,
            values=num_col,
            index=cat_col1,
            columns=cat_col2,
            aggfunc=estimator,
            fill_value=0,  # Fill missing combinations with 0
            margins=True,  # Automatically adds totals
            margins_name='ALL'
        )
    except Exception as e:
        # Fallback to original groupby method if pivot table fails
        print(f"Warning: Pivot table failed ({e}), falling back to groupby method")
        return _create_rollup_table_fallback(work_df, cat_col1, cat_col2, num_col, estimator)

    # 🔄 Convert pivot table back to the original DataFrame format
    # This maintains compatibility with the rest of the code
    rollup_rows = []

    # Iterate through the pivot table to extract all combinations
    for row_idx in pivot.index:
        for col_idx in pivot.columns:
            value = pivot.loc[row_idx, col_idx]

            # Skip NaN values that might occur in sparse data
            if pd.isna(value):
                continue

            rollup_rows.append({
                cat_col1: row_idx,
                cat_col2: col_idx,
                num_col: value
            })

    # Convert back to DataFrame with the expected structure
    rollup_table = pd.DataFrame(rollup_rows)

    return rollup_table


def _create_rollup_table_fallback(df: pd.DataFrame, cat_col1: str, cat_col2: str,
                                 num_col: str, estimator: str) -> pd.DataFrame:
    """Fallback method using the original groupby approach."""

    work_df = df.copy()

    # Map estimator strings to pandas functions
    AVAILABLE_ESTIMATORS = {
        'sum': 'sum',
        'mean': 'mean',
        'median': 'median',
        'count': 'count',
        'std': 'std',
        'var': 'var',
        'min': 'min',
        'max': 'max',
        'first': 'first',
        'last': 'last'
    }

    agg_func = AVAILABLE_ESTIMATORS[estimator]

    # Individual combinations (most granular level)
    individual = work_df.groupby([cat_col1, cat_col2])[num_col].agg(agg_func).reset_index()

    # Category 1 totals (cat_col1 + 'ALL')
    cat1_totals = work_df.groupby(cat_col1)[num_col].agg(agg_func).reset_index()
    cat1_totals[cat_col2] = 'ALL'

    # Category 2 totals ('ALL' + cat_col2)
    cat2_totals = work_df.groupby(cat_col2)[num_col].agg(agg_func).reset_index()
    cat2_totals[cat_col1] = 'ALL'

    # Grand total ('ALL' + 'ALL')
    if estimator == 'count':
        grand_total_value = work_df[num_col].count()
    else:
        grand_total_value = work_df[num_col].agg(agg_func)

    grand_total = pd.DataFrame({
        cat_col1: ['ALL'],
        cat_col2: ['ALL'],
        num_col: [grand_total_value]
    })

    # Combine all levels
    rollup_table = pd.concat([individual, cat1_totals, cat2_totals, grand_total],
                            ignore_index=True)

    return rollup_table


def _build_result_dictionary(
    aggregations: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
    num_col: str,
    estimator: str,
    values: str,
    precision: int
) -> Dict[str, Union[float, int]]:
    """Build the final result dictionary with descriptive keys."""

    result = {}

    # Get grand total for percentage calculations
    grand_total = aggregations[
        (aggregations[cat_col1] == 'ALL') & (aggregations[cat_col2] == 'ALL')
    ][num_col].iloc[0]

    # Create descriptive labels based on estimator
    estimator_label = _get_estimator_label(estimator)

    # Always add grand total (needed for percentages and useful for absolute)
    result[f'{estimator_label} {num_col.lower()} total'] = _convert_to_python_type(grand_total)

    for _, row in aggregations.iterrows():
        val1, val2, amount = row[cat_col1], row[cat_col2], row[num_col]

        if val1 == 'ALL' and val2 == 'ALL':
            # Grand total - already added
            continue

        elif val1 == 'ALL':
            # Category 2 total
            if values in ["absolute", "both"]:
                result[f'{estimator_label} {num_col.lower()} for {val2}'] = _convert_to_python_type(amount)
            if values in ["percentages", "both"]:
                percentage = _safe_division(amount, grand_total) * 100
                result[f'{estimator_label} {num_col.lower()} percentage related to {val2}'] = \
                    round(float(percentage), precision)

        elif val2 == 'ALL':
            # Category 1 total
            if values in ["absolute", "both"]:
                result[f'{estimator_label} {num_col.lower()} for {val1}'] = _convert_to_python_type(amount)
            if values in ["percentages", "both"]:
                percentage = _safe_division(amount, grand_total) * 100
                result[f'{estimator_label} {num_col.lower()} percentage related to {val1}'] = \
                    round(float(percentage), precision)

        else:
            # Individual combination
            if values in ["absolute", "both"]:
                result[f'{estimator_label} {num_col.lower()} for {val1} and {val2}'] = \
                    _convert_to_python_type(amount)

            if values in ["percentages", "both"]:
                # Percentage of total
                percentage = _safe_division(amount, grand_total) * 100
                result[f'{estimator_label} {num_col.lower()} percentage related to {val1} and {val2}'] = \
                    round(float(percentage), precision)

                # 🚀 IMPROVED: More efficient percentage calculations using pre-computed totals
                # What percentage of cat_col1 total does this combination represent?
                cat1_total_row = aggregations[
                    (aggregations[cat_col1] == val1) & (aggregations[cat_col2] == 'ALL')
                ]

                if not cat1_total_row.empty:
                    cat1_total = cat1_total_row[num_col].iloc[0]
                    if cat1_total != 0:  # Avoid division by zero
                        cat1_percentage = _safe_division(amount, cat1_total) * 100
                        result[f'Percentage of {val2} in {val1} ({estimator})'] = \
                            round(float(cat1_percentage), precision)

                # What percentage of cat_col2 total does this combination represent?
                cat2_total_row = aggregations[
                    (aggregations[cat_col1] == 'ALL') & (aggregations[cat_col2] == val2)
                ]

                if not cat2_total_row.empty:
                    cat2_total = cat2_total_row[num_col].iloc[0]
                    if cat2_total != 0:  # Avoid division by zero
                        cat2_percentage = _safe_division(amount, cat2_total) * 100
                        result[f'Percentage of {val1} in {val2} ({estimator})'] = \
                            round(float(cat2_percentage), precision)

    return result


def _get_estimator_label(estimator: str) -> str:
    """Get descriptive label for estimator."""
    labels = {
        'sum': 'Total',
        'mean': 'Average',
        'median': 'Median',
        'count': 'Count of',
        'std': 'Standard deviation of',
        'var': 'Variance of',
        'min': 'Minimum',
        'max': 'Maximum',
        'first': 'First',
        'last': 'Last'
    }
    return labels.get(estimator, estimator.title())


def _safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """Safely perform division, returning 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _convert_to_python_type(value: Union[int, float, np.number]) -> Union[int, float]:
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    else:
        return value

from typing import Dict, List, Tuple, Union
import re


def add_pareto_insights(
    aggregations_dict: Dict[str, Union[float, int]],
    concentration_threshold: float = 80.0,
    dominance_multiplier: float = 1.5,
    underperformer_threshold: float = 5.0,
    top_n: int = 3
) -> Dict[str, Union[float, int, str]]:
    """
    Add Pareto analysis insights to existing aggregations dictionary.

    This function analyzes the existing aggregations and adds insights about:
    1. Top contributors and concentration patterns
    2. Dominance analysis (categories exceeding fair share)
    3. Underperformer detection

    Args:
        aggregations_dict: Output from generate_multilevel_aggregations function
        concentration_threshold: Percentage threshold for concentration analysis (default: 80%)
        dominance_multiplier: Multiplier vs fair share to consider dominant (default: 1.5x)
        underperformer_threshold: Percentage below which to flag as underperformer (default: 5%)
        top_n: Number of top contributors to highlight (default: 3)

    Returns:
        Dict with original data plus Pareto insights with keys like:
        - "Top 2 contributors in [category]: [details]"
        - "Concentration analysis for [category]: [X]% in top [N]"
        - "Dominant subcategories in [category]: [details]"
        - "Underperformers in [category]: [details]"
    """

    # Create copy to avoid modifying original
    result = aggregations_dict.copy()

    # Parse the existing data structure
    parsed_data = _parse_aggregations_structure(aggregations_dict)

    # Add insights for each main category
    for category_name, subcategory_data in parsed_data['subcategories'].items():
        if subcategory_data:  # Only if there are subcategories

            # 1. Top Contributors & Concentration Analysis
            _add_top_contributors_insights(result, category_name, subcategory_data,
                                         concentration_threshold, top_n)

            # 2. Dominance Analysis
            _add_dominance_insights(result, category_name, subcategory_data,
                                  dominance_multiplier)

            # 3. Underperformer Detection
            _add_underperformer_insights(result, category_name, subcategory_data,
                                       underperformer_threshold)

    return result


def _parse_aggregations_structure(aggregations_dict: Dict) -> Dict:
    """Parse the aggregations dictionary to extract category structure."""

    parsed = {
        'total_key': None,
        'categories': {},  # cat1 or cat2 totals
        'subcategories': {}  # combinations grouped by main category
    }

    # Patterns to match different key types
    total_pattern = r'^(\w+) total$'
    category_total_pattern = r'^(\w+) total for (\w+) (.+)$'
    composition_pattern = r'^Percentage of (\w+) (.+) in (\w+) (.+)$'

    for key, value in aggregations_dict.items():
        # Find total key
        if re.match(total_pattern, key):
            parsed['total_key'] = key

        # Find composition percentages (these tell us the subcategory breakdowns)
        elif match := re.match(composition_pattern, key):
            subcategory_type, subcategory_value = match.group(1), match.group(2)
            main_category_type, main_category_value = match.group(3), match.group(4)

            # Group by main category
            main_key = f"{main_category_type} {main_category_value}"
            if main_key not in parsed['subcategories']:
                parsed['subcategories'][main_key] = []

            parsed['subcategories'][main_key].append({
                'subcategory': f"{subcategory_type} {subcategory_value}",
                'percentage': value,
                'main_category': main_key
            })

    return parsed


def _add_top_contributors_insights(result: Dict, category_name: str,
                                 subcategory_data: List[Dict],
                                 concentration_threshold: float, top_n: int):
    """Add top contributors and concentration insights."""

    # Sort by percentage descending
    sorted_subs = sorted(subcategory_data, key=lambda x: x['percentage'], reverse=True)

    # Top N contributors
    top_contributors = sorted_subs[:top_n]
    if len(top_contributors) >= 2:
        top_names = [f"{sub['subcategory']} ({sub['percentage']}%)"
                    for sub in top_contributors]
        top_total = sum(sub['percentage'] for sub in top_contributors)

        result[f"Top {len(top_contributors)} contributors in {category_name}"] = \
            f"{', '.join(top_names)} - Total: {top_total:.1f}%"

    # Concentration analysis - how many needed for threshold%
    cumulative = 0
    count_for_threshold = 0
    for sub in sorted_subs:
        cumulative += sub['percentage']
        count_for_threshold += 1
        if cumulative >= concentration_threshold:
            break

    total_subcategories = len(subcategory_data)
    result[f"Concentration analysis for {category_name}"] = \
        f"{concentration_threshold}% concentrated in {count_for_threshold} out of {total_subcategories} subcategories"


def _add_dominance_insights(result: Dict, category_name: str,
                          subcategory_data: List[Dict], dominance_multiplier: float):
    """Add dominance analysis insights."""

    total_subcategories = len(subcategory_data)
    fair_share = 100.0 / total_subcategories
    dominance_threshold = fair_share * dominance_multiplier

    dominant_subs = [sub for sub in subcategory_data
                     if sub['percentage'] > dominance_threshold]

    if dominant_subs:
        # Sort by percentage for better readability
        dominant_subs.sort(key=lambda x: x['percentage'], reverse=True)

        dominant_details = []
        for sub in dominant_subs:
            excess = sub['percentage'] / fair_share
            dominant_details.append(
                f"{sub['subcategory']} ({sub['percentage']}% vs {fair_share:.1f}% expected, {excess:.1f}x)"
            )

        result[f"Dominant subcategories in {category_name}"] = \
            f"{len(dominant_subs)} above {dominance_multiplier}x fair share: {', '.join(dominant_details)}"


def _add_underperformer_insights(result: Dict, category_name: str,
                               subcategory_data: List[Dict],
                               underperformer_threshold: float):
    """Add underperformer detection insights."""

    underperformers = [sub for sub in subcategory_data
                      if sub['percentage'] < underperformer_threshold]

    if underperformers:
        underperformer_names = [f"{sub['subcategory']} ({sub['percentage']}%)"
                               for sub in underperformers]

        result[f"Underperformers in {category_name}"] = \
            f"{len(underperformers)} below {underperformer_threshold}%: {', '.join(underperformer_names)}"

import numpy as np
import re
from typing import Dict, Union

def generate_comparison_insights(data_json: Dict[str, Union[float, np.float64]]) -> Dict[str, float]:
    """
    Generates comparative insights from financial barplot JSON data.

    Parameters:
    -----------
    data_json : dict
        JSON dictionary generated by plot_financial_barplot function containing
        financial metrics by category and overall metric

    Returns:
    --------
    dict
        Enhanced dictionary with original data plus comparative insights including:
        - Differences between each category and overall mean
        - Ratios between categories (using highest value as denominator)
        - Percentage differences
    """

    # Create a copy to avoid modifying original data
    enhanced_json = data_json.copy()

    # Convert numpy types to float for consistency
    for key, value in enhanced_json.items():
        if isinstance(value, (np.float64, np.int64)):
            enhanced_json[key] = float(value)

    # Extract metric information from keys
    overall_key = None
    category_data = {}

    # Find overall metric and extract base information
    for key, value in enhanced_json.items():
        if "Overall" in key:
            overall_key = key
            overall_value = value

            # Extract metric name from overall key
            # Pattern: "Estimator Overall MetricName"
            match = re.search(r"(\w+)\s+Overall\s+(.+)", key)
            if match:
                estimator = match.group(1)
                metric_name = match.group(2)
        else:
            # Extract category information
            # Pattern: "Estimator MetricName for CategoryColumn CategoryValue"
            match = re.search(r"(\w+)\s+(.+?)\s+for\s+(.+?)\s+(.+)", key)
            if match:
                est = match.group(1)
                metric = match.group(2)
                category_column = match.group(3)
                category_value = match.group(4)

                category_data[category_value] = {
                    'value': value,
                    'estimator': est,
                    'metric': metric,
                    'category_column': category_column
                }

    # Generate insights only if we have both overall and category data
    if overall_key and category_data:

        # 1. Generate differences between each category and overall mean
        for category, info in category_data.items():
            difference = info['value'] - overall_value
            diff_key = f"{info['metric']} difference between {category} and the Overall {info['metric']} {info['estimator']}"
            enhanced_json[diff_key] = round(difference, 2)

        # 2. Generate percentage differences between each category and overall mean
        if overall_value != 0:  # Avoid division by zero
            for category, info in category_data.items():
                pct_difference = ((info['value'] - overall_value) / abs(overall_value)) * 100
                pct_key = f"{info['metric']} percentage difference between {category} and the Overall {info['metric']} {info['estimator']}"
                enhanced_json[pct_key] = round(pct_difference, 2)

        # 3. Find the category with highest absolute value for ratio comparisons
        max_category = max(category_data.items(), key=lambda x: abs(x[1]['value']))
        max_category_name = max_category[0]
        max_category_value = max_category[1]['value']

        # 4. Generate ratios between other categories and the highest value category
        if max_category_value != 0:  # Avoid division by zero
            for category, info in category_data.items():
                if category != max_category_name:
                    ratio = info['value'] / max_category_value
                    ratio_key = f"{info['metric']} ratio between {category} and {max_category_name}"
                    enhanced_json[ratio_key] = round(ratio, 4)

        # 5. Generate pairwise comparisons (differences) between categories
        categories = list(category_data.keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                difference = category_data[cat1]['value'] - category_data[cat2]['value']
                comparison_key = f"{category_data[cat1]['metric']} difference between {cat1} and {cat2}"
                enhanced_json[comparison_key] = round(difference, 2)

        # 6. Generate relative performance insights (above/below average indicators)
        above_average = []
        below_average = []

        for category, info in category_data.items():
            if info['value'] > overall_value:
                above_average.append(category)
            elif info['value'] < overall_value:
                below_average.append(category)

        # Add summary insights
        if above_average:
            enhanced_json[f"Categories above {metric_name} average"] = above_average
        if below_average:
            enhanced_json[f"Categories below {metric_name} average"] = below_average

        # 7. Generate ranking information
        sorted_categories = sorted(category_data.items(), key=lambda x: x[1]['value'], reverse=True)
        ranking = {cat: idx + 1 for idx, (cat, _) in enumerate(sorted_categories)}

        for category, rank in ranking.items():
            rank_key = f"{metric_name} ranking for {category}"
            enhanced_json[rank_key] = rank

        # 8. Generate spread insights
        values = [info['value'] for info in category_data.values()]
        min_value = min(values)
        max_value = max(values)
        spread = max_value - min_value

        enhanced_json[f"{metric_name} spread (max - min)"] = round(spread, 2)

    return enhanced_json