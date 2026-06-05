def _mean_score(scores: list[int]) -> float:
    """Return the arithmetic mean for a non-empty score list."""
    if not scores:
        raise ValueError(
            "At least one completed run is required"
            " to calculate score metrics."
        )
    return sum(scores) / len(scores)


def _median_sorted(values: list[int]) -> float:
    """Return the median for an already sorted non-empty list."""
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[midpoint])
    return (values[midpoint - 1] + values[midpoint]) / 2


def _quartiles_sorted(values: list[int]) -> tuple[float, float, float]:
    """Return Q1, median, and Q3 using Tukey's hinges."""
    median = _median_sorted(values)
    midpoint = len(values) // 2
    lower_half = values[:midpoint]
    upper_half = values[midpoint + (len(values) % 2) :]
    q1 = _median_sorted(lower_half) if lower_half else median
    q3 = _median_sorted(upper_half) if upper_half else median
    return q1, median, q3


def _score_summary_lines(scores: list[int]) -> list[str]:
    """Return human-readable summary statistics for completed test runs."""
    if not scores:
        raise ValueError(
            "At least one completed run is required"
            " to calculate score metrics."
        )
    sorted_scores = sorted(scores)
    n_scores = len(sorted_scores)
    mean_score = sum(sorted_scores) / n_scores
    q1, median_score, q3 = _quartiles_sorted(sorted_scores)
    minimum_score = sorted_scores[0]
    maximum_score = sorted_scores[-1]
    iqr = q3 - q1
    low_outlier_limit = q1 - 1.5 * iqr
    high_outlier_limit = q3 + 1.5 * iqr
    outliers = [
        score
        for score in sorted_scores
        if score < low_outlier_limit or score > high_outlier_limit
    ]
    top_count = max(1, (n_scores + 9) // 10)
    top_scores = sorted_scores[-top_count:]
    return [
        f"Runs: {n_scores}",
        f"Scores: {scores}",
        f"Mean Score: {mean_score:.2f}",
        f"Median Score: {median_score:.2f}",
        f"Min Score: {minimum_score}",
        f"Q1 Score: {q1:.2f}",
        f"Q3 Score: {q3:.2f}",
        f"Max Score: {maximum_score}",
        f"IQR: {iqr:.2f}",
        f"Outlier Bounds: < {low_outlier_limit:.2f}"
        f" or > {high_outlier_limit:.2f}",
        f"Outliers ({len(outliers)}): {outliers}",
        f"Top 10% Count: {top_count}",
        f"Top 10% Scores: {top_scores}",
    ]
