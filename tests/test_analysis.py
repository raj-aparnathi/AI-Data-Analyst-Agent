"""
tests/test_analysis.py — M4 Analysis + Visualization Tests
=============================================================
Comprehensive pytest test suite covering:
- Statistical operations (mean, median, mode, min, max, count, variance, std, quartiles)
- GroupBy + aggregation
- Sorting
- Filtering
- Top-N
- Correlation
- Trend analysis
- Analysis plan executor
- Visualization chart creation
- Invalid operations / columns
- Data integrity (source DataFrame not mutated)

Author : Member 4 (Analysis + Visualization)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pytest

from operations.statistics import (
    calculate_mean,
    calculate_median,
    calculate_mode,
    calculate_min,
    calculate_max,
    calculate_count,
    calculate_variance,
    calculate_std,
    calculate_quartiles,
    execute_statistic,
)
from operations.analysis import (
    group_by,
    sort_data,
    filter_data,
    top_n,
    calculate_correlation,
    analyze_trend,
    apply_analysis_plan,
    SUPPORTED_ANALYSIS_OPERATIONS,
)
from operations.visualization import (
    create_bar_chart,
    create_line_chart,
    create_histogram,
    create_scatter_plot,
    create_pie_chart,
    create_box_plot,
    create_heatmap,
    recommend_chart,
    build_chart_from_result,
    SUPPORTED_CHART_TYPES,
)
from agent.explainer import explain_result_local
from agent.planner import (
    parse_analysis_request_rule_based,
    generate_analysis_plan,
    parse_analysis_response,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_df():
    """A sample DataFrame for testing."""
    return pd.DataFrame({
        "product": ["A", "A", "B", "B", "C"],
        "sales": [100, 200, 300, 150, 250],
        "cost": [50, 80, 120, 60, 90],
        "region": ["East", "West", "East", "West", "East"],
    })


@pytest.fixture
def numeric_df():
    """DataFrame with only numeric columns."""
    return pd.DataFrame({
        "salary": [20000, 30000, 40000, 50000, 60000],
        "experience": [1, 3, 5, 7, 10],
        "age": [22, 25, 30, 35, 40],
    })


@pytest.fixture
def date_df():
    """DataFrame with a date column for trend analysis."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2023-01-15", "2023-02-20", "2023-03-10",
            "2023-04-05", "2023-05-18", "2023-06-22",
        ]),
        "sales": [100, 150, 130, 200, 180, 250],
        "units": [10, 15, 12, 20, 18, 25],
    })


@pytest.fixture
def categorical_df():
    """DataFrame with categorical columns."""
    return pd.DataFrame({
        "city": ["New York", "London", "New York", "Tokyo", "London", "London"],
        "department": ["Sales", "HR", "Sales", "IT", "Sales", "HR"],
    })


# ===========================================================================
# Statistics Tests
# ===========================================================================

class TestMean:
    def test_basic_mean(self, numeric_df):
        result = calculate_mean(numeric_df, "salary")
        assert result == 40000.0

    def test_mean_with_decimals(self):
        df = pd.DataFrame({"values": [20000, 30000, 40000]})
        result = calculate_mean(df, "values")
        assert result == 30000.0

    def test_mean_returns_float(self, numeric_df):
        result = calculate_mean(numeric_df, "salary")
        assert isinstance(result, float)


class TestMedian:
    def test_basic_median_odd(self, numeric_df):
        result = calculate_median(numeric_df, "salary")
        assert result == 40000.0

    def test_median_even(self):
        df = pd.DataFrame({"values": [10, 20, 30, 40]})
        result = calculate_median(df, "values")
        assert result == 25.0

    def test_median_returns_float(self, numeric_df):
        result = calculate_median(numeric_df, "salary")
        assert isinstance(result, float)


class TestMode:
    def test_single_mode(self):
        df = pd.DataFrame({"city": ["A", "B", "A", "C", "A"]})
        result = calculate_mode(df, "city")
        assert result == "A"

    def test_numeric_mode(self):
        df = pd.DataFrame({"values": [1, 2, 2, 3, 3, 3]})
        result = calculate_mode(df, "values")
        assert result == 3

    def test_multiple_modes(self):
        df = pd.DataFrame({"values": [1, 1, 2, 2, 3]})
        result = calculate_mode(df, "values")
        assert isinstance(result, list)
        assert 1 in result
        assert 2 in result


class TestMinMax:
    def test_min(self, numeric_df):
        result = calculate_min(numeric_df, "salary")
        assert result == 20000

    def test_max(self, numeric_df):
        result = calculate_max(numeric_df, "salary")
        assert result == 60000

    def test_min_string(self, categorical_df):
        result = calculate_min(categorical_df, "city")
        assert result == "London"

    def test_max_string(self, categorical_df):
        result = calculate_max(categorical_df, "city")
        assert result == "Tokyo"


class TestCount:
    def test_count_no_nulls(self, numeric_df):
        result = calculate_count(numeric_df, "salary")
        assert result == 5

    def test_count_with_nulls(self):
        df = pd.DataFrame({"values": [1, 2, None, 4, None]})
        result = calculate_count(df, "values")
        assert result == 3

    def test_count_returns_int(self, numeric_df):
        result = calculate_count(numeric_df, "salary")
        assert isinstance(result, int)


class TestVariance:
    def test_variance(self):
        df = pd.DataFrame({"values": [10, 20, 30, 40, 50]})
        result = calculate_variance(df, "values")
        assert result == pytest.approx(250.0)

    def test_variance_returns_float(self, numeric_df):
        result = calculate_variance(numeric_df, "salary")
        assert isinstance(result, float)


class TestStd:
    def test_std(self):
        df = pd.DataFrame({"values": [10, 20, 30, 40, 50]})
        result = calculate_std(df, "values")
        expected = pd.Series([10, 20, 30, 40, 50]).std()
        assert result == pytest.approx(expected)

    def test_std_returns_float(self, numeric_df):
        result = calculate_std(numeric_df, "salary")
        assert isinstance(result, float)


class TestQuartiles:
    def test_quartiles(self):
        df = pd.DataFrame({"values": list(range(1, 101))})
        result = calculate_quartiles(df, "values")
        assert "Q1" in result
        assert "Q2" in result
        assert "Q3" in result
        assert result["Q1"] == pytest.approx(25.75)
        assert result["Q2"] == pytest.approx(50.5)
        assert result["Q3"] == pytest.approx(75.25)

    def test_quartiles_returns_dict(self, numeric_df):
        result = calculate_quartiles(numeric_df, "salary")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"Q1", "Q2", "Q3"}


# ===========================================================================
# Invalid Input Tests (Statistics)
# ===========================================================================

class TestStatisticsValidation:
    def test_invalid_column(self, numeric_df):
        with pytest.raises(ValueError, match="does not exist"):
            calculate_mean(numeric_df, "nonexistent")

    def test_non_numeric_mean(self, categorical_df):
        with pytest.raises(ValueError, match="not numeric"):
            calculate_mean(categorical_df, "city")

    def test_non_numeric_variance(self, categorical_df):
        with pytest.raises(ValueError, match="not numeric"):
            calculate_variance(categorical_df, "city")

    def test_non_numeric_std(self, categorical_df):
        with pytest.raises(ValueError, match="not numeric"):
            calculate_std(categorical_df, "city")

    def test_non_numeric_quartiles(self, categorical_df):
        with pytest.raises(ValueError, match="not numeric"):
            calculate_quartiles(categorical_df, "city")

    def test_unsupported_statistic(self, numeric_df):
        with pytest.raises(ValueError, match="Unsupported"):
            execute_statistic(numeric_df, "invalid_op", "salary")


# ===========================================================================
# GroupBy Tests
# ===========================================================================

class TestGroupBy:
    def test_groupby_sum(self, sample_df):
        result = group_by(sample_df, "product", "sales", "sum")
        assert len(result) == 3
        # A: 100+200=300, B: 300+150=450, C: 250
        row_a = result[result["product"] == "A"]["sales"].values[0]
        row_b = result[result["product"] == "B"]["sales"].values[0]
        row_c = result[result["product"] == "C"]["sales"].values[0]
        assert row_a == 300
        assert row_b == 450
        assert row_c == 250

    def test_groupby_mean(self, sample_df):
        result = group_by(sample_df, "product", "sales", "mean")
        row_a = result[result["product"] == "A"]["sales"].values[0]
        assert row_a == 150.0  # (100+200)/2

    def test_groupby_count(self, sample_df):
        result = group_by(sample_df, "product", "sales", "count")
        row_a = result[result["product"] == "A"]["sales"].values[0]
        assert row_a == 2

    def test_groupby_median(self, sample_df):
        result = group_by(sample_df, "product", "sales", "median")
        row_a = result[result["product"] == "A"]["sales"].values[0]
        assert row_a == 150.0

    def test_groupby_min(self, sample_df):
        result = group_by(sample_df, "product", "sales", "min")
        row_a = result[result["product"] == "A"]["sales"].values[0]
        assert row_a == 100

    def test_groupby_max(self, sample_df):
        result = group_by(sample_df, "product", "sales", "max")
        row_a = result[result["product"] == "A"]["sales"].values[0]
        assert row_a == 200

    def test_groupby_invalid_column(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            group_by(sample_df, "nonexistent", "sales", "sum")

    def test_groupby_invalid_aggregation(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            group_by(sample_df, "product", "sales", "invalid_agg")


# ===========================================================================
# Sorting Tests
# ===========================================================================

class TestSorting:
    def test_sort_ascending(self, sample_df):
        result = sort_data(sample_df, "sales", ascending=True)
        assert list(result["sales"]) == [100, 150, 200, 250, 300]

    def test_sort_descending(self, sample_df):
        result = sort_data(sample_df, "sales", ascending=False)
        assert list(result["sales"]) == [300, 250, 200, 150, 100]

    def test_sort_invalid_column(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            sort_data(sample_df, "nonexistent")


# ===========================================================================
# Filtering Tests
# ===========================================================================

class TestFiltering:
    def test_filter_gt(self, sample_df):
        result = filter_data(sample_df, "sales", ">", 200)
        assert len(result) == 2  # 300, 250
        assert all(result["sales"] > 200)

    def test_filter_gte(self, sample_df):
        result = filter_data(sample_df, "sales", ">=", 200)
        assert len(result) == 3  # 200, 300, 250

    def test_filter_lt(self, sample_df):
        result = filter_data(sample_df, "sales", "<", 200)
        assert len(result) == 2  # 100, 150

    def test_filter_lte(self, sample_df):
        result = filter_data(sample_df, "sales", "<=", 200)
        assert len(result) == 3  # 100, 200, 150

    def test_filter_eq(self, sample_df):
        result = filter_data(sample_df, "sales", "==", 200)
        assert len(result) == 1

    def test_filter_neq(self, sample_df):
        result = filter_data(sample_df, "sales", "!=", 200)
        assert len(result) == 4

    def test_filter_string_eq(self, sample_df):
        result = filter_data(sample_df, "product", "==", "A")
        assert len(result) == 2

    def test_filter_invalid_operator(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            filter_data(sample_df, "sales", "in", [100, 200])

    def test_filter_invalid_column(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            filter_data(sample_df, "nonexistent", ">", 100)


# ===========================================================================
# Top-N Tests
# ===========================================================================

class TestTopN:
    def test_top_3(self, sample_df):
        result = top_n(sample_df, "sales", 3)
        assert len(result) == 3
        # Top 3 descending: 300, 250, 200
        assert list(result["sales"]) == [300, 250, 200]

    def test_top_n_ascending(self, sample_df):
        result = top_n(sample_df, "sales", 3, ascending=True)
        assert len(result) == 3
        assert list(result["sales"]) == [100, 150, 200]

    def test_top_n_exceeding_rows(self, sample_df):
        # Request more rows than available — should not crash
        result = top_n(sample_df, "sales", 100)
        assert len(result) == 5  # Clamped to available rows

    def test_top_n_invalid_n(self, sample_df):
        with pytest.raises(ValueError, match="positive integer"):
            top_n(sample_df, "sales", 0)

    def test_top_n_negative(self, sample_df):
        with pytest.raises(ValueError, match="positive integer"):
            top_n(sample_df, "sales", -5)

    def test_top_n_invalid_column(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            top_n(sample_df, "nonexistent", 5)


# ===========================================================================
# Correlation Tests
# ===========================================================================

class TestCorrelation:
    def test_correlation_matrix(self, numeric_df):
        result = calculate_correlation(numeric_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
        # Diagonal should be 1.0
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_correlation_specific_columns(self, numeric_df):
        result = calculate_correlation(numeric_df, columns=["salary", "experience"])
        assert result.shape == (2, 2)
        assert "salary" in result.columns
        assert "experience" in result.columns

    def test_correlation_too_few_numeric(self, categorical_df):
        with pytest.raises(ValueError, match="at least 2 numeric"):
            calculate_correlation(categorical_df)

    def test_correlation_nonexistent_column(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            calculate_correlation(numeric_df, columns=["salary", "nonexistent"])


# ===========================================================================
# Trend Analysis Tests
# ===========================================================================

class TestTrend:
    def test_basic_trend(self, date_df):
        result = analyze_trend(date_df, "date", "sales", "monthly")
        assert result["direction"] == "increasing"
        assert result["start_value"] is not None
        assert result["end_value"] is not None
        assert result["change"] is not None
        assert isinstance(result["trend_data"], pd.DataFrame)

    def test_trend_invalid_column(self, date_df):
        with pytest.raises(ValueError, match="does not exist"):
            analyze_trend(date_df, "nonexistent", "sales", "monthly")

    def test_trend_non_numeric_value(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "category": ["A", "B"],
        })
        with pytest.raises(ValueError, match="numeric"):
            analyze_trend(df, "date", "category", "monthly")

    def test_trend_invalid_frequency(self, date_df):
        with pytest.raises(ValueError, match="Unsupported frequency"):
            analyze_trend(date_df, "date", "sales", "weekly")


# ===========================================================================
# Analysis Plan Executor Tests
# ===========================================================================

class TestApplyAnalysisPlan:
    def test_mean_plan(self, numeric_df):
        plan = {"operation": "mean", "column": "salary"}
        result = apply_analysis_plan(numeric_df, plan)
        assert result["operation"] == "mean"
        assert result["result"] == 40000.0
        assert result["result_type"] == "scalar"

    def test_groupby_plan(self, sample_df):
        plan = {
            "operation": "groupby",
            "group_column": "product",
            "value_column": "sales",
            "aggregation": "sum",
            "sort": "descending",
            "limit": 1,
        }
        result = apply_analysis_plan(sample_df, plan)
        assert result["operation"] == "groupby"
        assert result["result_type"] == "dataframe"
        assert len(result["result"]) == 1
        assert result["result"]["product"].values[0] == "B"

    def test_filter_plan(self, sample_df):
        plan = {
            "operation": "filter",
            "column": "sales",
            "operator": ">",
            "value": 200,
        }
        result = apply_analysis_plan(sample_df, plan)
        assert result["operation"] == "filter"
        assert len(result["result"]) == 2

    def test_top_n_plan(self, sample_df):
        plan = {
            "operation": "top_n",
            "column": "sales",
            "n": 3,
        }
        result = apply_analysis_plan(sample_df, plan)
        assert result["operation"] == "top_n"
        assert len(result["result"]) == 3

    def test_correlation_plan(self, numeric_df):
        plan = {
            "operation": "correlation",
            "columns": ["salary", "experience"],
        }
        result = apply_analysis_plan(numeric_df, plan)
        assert result["operation"] == "correlation"
        assert result["result_type"] == "matrix"

    def test_sort_plan(self, sample_df):
        plan = {
            "operation": "sort",
            "column": "sales",
            "sort": "descending",
        }
        result = apply_analysis_plan(sample_df, plan)
        assert result["operation"] == "sort"
        assert list(result["result"]["sales"]) == [300, 250, 200, 150, 100]

    def test_unsupported_operation(self, sample_df):
        plan = {"operation": "train_model", "column": "sales"}
        with pytest.raises(ValueError, match="Unsupported operation"):
            apply_analysis_plan(sample_df, plan)

    def test_missing_operation(self, sample_df):
        plan = {"column": "sales"}
        with pytest.raises(ValueError, match="'operation'"):
            apply_analysis_plan(sample_df, plan)

    def test_invalid_column_in_plan(self, sample_df):
        plan = {"operation": "mean", "column": "nonexistent"}
        with pytest.raises(ValueError, match="does not exist"):
            apply_analysis_plan(sample_df, plan)


# ===========================================================================
# Data Integrity Tests
# ===========================================================================

class TestDataIntegrity:
    def test_groupby_does_not_modify_source(self, sample_df):
        original = sample_df.copy()
        group_by(sample_df, "product", "sales", "sum")
        pd.testing.assert_frame_equal(sample_df, original)

    def test_sort_does_not_modify_source(self, sample_df):
        original = sample_df.copy()
        sort_data(sample_df, "sales", ascending=False)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_filter_does_not_modify_source(self, sample_df):
        original = sample_df.copy()
        filter_data(sample_df, "sales", ">", 200)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_top_n_does_not_modify_source(self, sample_df):
        original = sample_df.copy()
        top_n(sample_df, "sales", 3)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_correlation_does_not_modify_source(self, numeric_df):
        original = numeric_df.copy()
        calculate_correlation(numeric_df)
        pd.testing.assert_frame_equal(numeric_df, original)

    def test_apply_plan_does_not_modify_source(self, sample_df):
        original = sample_df.copy()
        plan = {"operation": "mean", "column": "sales"}
        apply_analysis_plan(sample_df, plan)
        pd.testing.assert_frame_equal(sample_df, original)


# ===========================================================================
# Visualization Tests
# ===========================================================================

class TestVisualization:
    def test_bar_chart(self, sample_df):
        grouped = group_by(sample_df, "product", "sales", "sum")
        fig = create_bar_chart(grouped, "product", "sales")
        assert fig is not None
        assert hasattr(fig, "data")

    def test_line_chart(self, date_df):
        fig = create_line_chart(date_df, "date", "sales")
        assert fig is not None

    def test_histogram(self, numeric_df):
        fig = create_histogram(numeric_df, "salary")
        assert fig is not None

    def test_scatter_plot(self, numeric_df):
        fig = create_scatter_plot(numeric_df, "salary", "experience")
        assert fig is not None

    def test_pie_chart(self, sample_df):
        grouped = group_by(sample_df, "product", "sales", "sum")
        fig = create_pie_chart(grouped, "product", "sales")
        assert fig is not None

    def test_box_plot(self, numeric_df):
        fig = create_box_plot(numeric_df, "salary")
        assert fig is not None

    def test_heatmap(self, numeric_df):
        corr = calculate_correlation(numeric_df)
        fig = create_heatmap(corr)
        assert fig is not None


class TestChartRecommendation:
    def test_groupby_recommends_bar(self):
        result = {
            "operation": "groupby",
            "result": pd.DataFrame({"product": ["A"], "sales": [100]}),
            "result_type": "dataframe",
        }
        assert recommend_chart(result) == "bar"

    def test_correlation_recommends_heatmap(self):
        result = {
            "operation": "correlation",
            "result": pd.DataFrame(),
            "result_type": "matrix",
        }
        assert recommend_chart(result) == "heatmap"

    def test_trend_recommends_line(self):
        result = {
            "operation": "trend",
            "result": {},
            "result_type": "dict",
        }
        assert recommend_chart(result) == "line"

    def test_scalar_recommends_none(self):
        result = {
            "operation": "mean",
            "result": 42.0,
            "result_type": "scalar",
        }
        assert recommend_chart(result) is None

    def test_explicit_visualization_overrides(self):
        result = {
            "operation": "groupby",
            "result": pd.DataFrame({"product": ["A"], "sales": [100]}),
            "result_type": "dataframe",
            "visualization": "pie",
        }
        assert recommend_chart(result) == "pie"


class TestBuildChartFromResult:
    def test_build_bar_chart(self, sample_df):
        grouped = group_by(sample_df, "product", "sales", "sum")
        analysis_result = {
            "operation": "groupby",
            "result": grouped,
            "result_type": "dataframe",
            "visualization": "bar",
            "summary": {
                "group_column": "product",
                "value_column": "sales",
            },
        }
        fig = build_chart_from_result(analysis_result)
        assert fig is not None

    def test_build_heatmap_from_correlation(self, numeric_df):
        corr = calculate_correlation(numeric_df)
        analysis_result = {
            "operation": "correlation",
            "result": corr,
            "result_type": "matrix",
            "visualization": "heatmap",
            "summary": {},
        }
        fig = build_chart_from_result(analysis_result)
        assert fig is not None

    def test_returns_none_for_scalar(self):
        analysis_result = {
            "operation": "mean",
            "result": 42.0,
            "result_type": "scalar",
            "visualization": None,
            "summary": {},
        }
        fig = build_chart_from_result(analysis_result)
        assert fig is None


# ===========================================================================
# Explainer Tests
# ===========================================================================

class TestExplainer:
    def test_scalar_explanation(self):
        result = {
            "operation": "mean",
            "result": 40000.0,
            "result_type": "scalar",
            "summary": {"column": "salary", "operation": "mean"},
        }
        explanation = explain_result_local("What is the average salary?", result)
        assert "40,000" in explanation
        assert "salary" in explanation

    def test_groupby_explanation(self, sample_df):
        grouped = group_by(sample_df, "product", "sales", "sum")
        result = {
            "operation": "groupby",
            "result": grouped.sort_values("sales", ascending=False).head(1).reset_index(drop=True),
            "result_type": "dataframe",
            "summary": {
                "group_column": "product",
                "value_column": "sales",
                "aggregation": "sum",
            },
        }
        explanation = explain_result_local(
            "Which product has the highest sales?", result
        )
        assert "B" in explanation  # Product B has highest sum

    def test_empty_result_explanation(self):
        result = {
            "operation": "filter",
            "result": pd.DataFrame(),
            "result_type": "dataframe",
            "summary": {"filter": "sales > 999999"},
        }
        explanation = explain_result_local("Show high sales", result)
        assert "no matching" in explanation.lower()

    def test_correlation_explanation(self, numeric_df):
        corr = calculate_correlation(numeric_df)
        result = {
            "operation": "correlation",
            "result": corr,
            "result_type": "matrix",
            "summary": {"variables": list(corr.columns)},
        }
        explanation = explain_result_local(
            "Is salary correlated with experience?", result
        )
        assert "correlation" in explanation.lower()

    def test_explanation_does_not_claim_causation(self, numeric_df):
        corr = calculate_correlation(numeric_df)
        result = {
            "operation": "correlation",
            "result": corr,
            "result_type": "matrix",
            "summary": {"variables": list(corr.columns)},
        }
        explanation = explain_result_local(
            "Is salary correlated with experience?", result
        )
        assert "causes" not in explanation.lower()
        assert "caused" not in explanation.lower()


# ===========================================================================
# Planner Tests
# ===========================================================================

class TestRuleBasedPlanner:
    def test_average_question(self, numeric_df):
        plan = parse_analysis_request_rule_based(
            "What is the average salary?", numeric_df
        )
        assert plan is not None
        assert plan["operation"] == "mean"
        assert plan["column"] == "salary"

    def test_groupby_question(self, sample_df):
        plan = parse_analysis_request_rule_based(
            "What is the average sales by product?", sample_df
        )
        assert plan is not None
        assert plan["operation"] == "groupby"
        assert plan["aggregation"] == "mean"

    def test_top_n_question(self, sample_df):
        plan = parse_analysis_request_rule_based(
            "Show top 3 products by sales", sample_df
        )
        assert plan is not None
        assert plan["operation"] == "top_n"
        assert plan["n"] == 3

    def test_correlation_question(self, numeric_df):
        plan = parse_analysis_request_rule_based(
            "Is salary correlated with experience?", numeric_df
        )
        assert plan is not None
        assert plan["operation"] == "correlation"

    def test_unrecognized_question(self, numeric_df):
        plan = parse_analysis_request_rule_based(
            "Tell me something interesting", numeric_df
        )
        assert plan is None

    def test_highest_question(self, sample_df):
        plan = parse_analysis_request_rule_based(
            "Which product has the highest sales?", sample_df
        )
        assert plan is not None
        assert plan["operation"] in ("groupby", "max")

    def test_median_question(self, numeric_df):
        plan = parse_analysis_request_rule_based(
            "What is the median salary?", numeric_df
        )
        assert plan is not None
        assert plan["operation"] == "median"
        assert plan["column"] == "salary"


class TestParseAnalysisResponse:
    def test_valid_json(self):
        response = '{"operation": "mean", "column": "salary"}'
        plan = parse_analysis_response(response)
        assert plan["operation"] == "mean"

    def test_json_in_markdown_fences(self):
        response = '```json\n{"operation": "mean", "column": "salary"}\n```'
        plan = parse_analysis_response(response)
        assert plan["operation"] == "mean"

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_analysis_response("not a json at all")

    def test_missing_operation_key(self):
        with pytest.raises(ValueError, match="'operation'"):
            parse_analysis_response('{"column": "salary"}')


class TestGenerateAnalysisPlan:
    def test_rule_based_plan(self, numeric_df):
        plan = generate_analysis_plan("What is the average salary?", numeric_df)
        assert plan["operation"] == "mean"

    def test_unknown_question_returns_prompt(self, numeric_df):
        plan = generate_analysis_plan(
            "Tell me something very interesting about this data", numeric_df
        )
        assert plan.get("status") == "awaiting_ai_response"
        assert "prompt" in plan

    def test_invalid_column_validation(self, numeric_df):
        with pytest.raises(ValueError, match="does not exist"):
            generate_analysis_plan(
                "What is the average salary?",
                numeric_df,
                ai_response='{"operation": "mean", "column": "nonexistent"}',
            )

    def test_invalid_operation_validation(self, numeric_df):
        with pytest.raises(ValueError, match="Unsupported operation"):
            generate_analysis_plan(
                "Do something",
                numeric_df,
                ai_response='{"operation": "train_model", "column": "salary"}',
            )


# ===========================================================================
# Security Tests
# ===========================================================================

class TestSecurity:
    def test_no_eval_in_filter(self, sample_df):
        """Ensure arbitrary expressions are rejected."""
        with pytest.raises(ValueError):
            filter_data(sample_df, "sales", "in", [100, 200])

    def test_unsupported_operation_rejected(self, sample_df):
        plan = {"operation": "exec", "code": "print('hacked')"}
        with pytest.raises(ValueError, match="Unsupported"):
            apply_analysis_plan(sample_df, plan)

    def test_supported_operations_whitelist(self):
        """Verify the whitelist contains expected operations."""
        expected = {
            "mean", "median", "mode", "min", "max", "count",
            "variance", "std", "quartiles",
            "groupby", "sort", "filter", "top_n",
            "correlation", "trend",
        }
        assert SUPPORTED_ANALYSIS_OPERATIONS == expected

    def test_supported_chart_types_whitelist(self):
        """Verify the chart whitelist."""
        expected = {"bar", "line", "histogram", "scatter", "pie", "box", "heatmap"}
        assert SUPPORTED_CHART_TYPES == expected
