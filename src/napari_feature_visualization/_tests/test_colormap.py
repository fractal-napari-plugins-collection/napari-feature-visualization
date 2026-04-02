"""Tests for pure colormap logic in _colormap.py — no Qt or viewer needed."""

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from napari_feature_visualization._colormap import (
    check_default_label_column,
    compute_colormap,
    get_colormap_choices,
    get_contrast_limits,
    get_default_colormap,
)


@pytest.fixture()
def continuous_df():
    return pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5, 6],
            "feature1": [100.0, 200.0, 300.0, 500.0, 900.0, 1001.0],
        }
    )


@pytest.fixture()
def categorical_df():
    return pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5, 6],
            "cell_type": [
                "type_A",
                "type_A",
                "type_A",
                "type_B",
                "type_B",
                "type_C",
            ],
        }
    )


# ---------------------------------------------------------------------------
# check_default_label_column
# ---------------------------------------------------------------------------


def test_check_default_label_column_label():
    df = pd.DataFrame({"label": [1], "feature": [0.5]})
    assert check_default_label_column(df) == "label"


def test_check_default_label_column_Label():
    df = pd.DataFrame({"Label": [1], "feature": [0.5]})
    assert check_default_label_column(df) == "Label"


def test_check_default_label_column_index():
    df = pd.DataFrame({"index": [1], "feature": [0.5]})
    assert check_default_label_column(df) == "index"


def test_check_default_label_column_none():
    df = pd.DataFrame({"feature": [0.5]})
    assert check_default_label_column(df) == ""


# ---------------------------------------------------------------------------
# get_colormap_choices
# ---------------------------------------------------------------------------


def test_colormap_choices_continuous(continuous_df):
    choices = get_colormap_choices(continuous_df, "feature1")
    assert "viridis" in choices
    assert "tab10 (10)" not in choices


def test_colormap_choices_categorical_few(categorical_df):
    # 3 categories → all qualitative colormaps qualify
    choices = get_colormap_choices(categorical_df, "cell_type")
    assert all("(" in c for c in choices)
    assert "viridis" not in choices


def test_colormap_choices_categorical_many():
    # 11 categories → tab10 (max 10) should be excluded
    df = pd.DataFrame({"label": range(11), "cat": [f"c{i}" for i in range(11)]})
    choices = get_colormap_choices(df, "cat")
    assert not any(c.startswith("tab10 (") for c in choices)


def test_colormap_choices_categorical_fallback():
    # More categories than any qualitative colormap → label_colormap
    df = pd.DataFrame({"label": range(25), "cat": [f"c{i}" for i in range(25)]})
    choices = get_colormap_choices(df, "cat")
    assert choices == ["label_colormap"]


# ---------------------------------------------------------------------------
# get_default_colormap
# ---------------------------------------------------------------------------


def test_default_colormap_prefers_viridis():
    choices = ["magma", "viridis", "plasma"]
    assert get_default_colormap(choices) == "viridis"


def test_default_colormap_prefers_tab10():
    choices = ["Accent (8)", "tab10 (10)", "tab20 (20)"]
    assert get_default_colormap(choices) == "tab10 (10)"


def test_default_colormap_fallback():
    choices = ["Accent (8)", "Dark2 (8)"]
    assert get_default_colormap(choices) == "Accent (8)"


# ---------------------------------------------------------------------------
# get_contrast_limits
# ---------------------------------------------------------------------------


def test_contrast_limits(continuous_df):
    lower, upper = get_contrast_limits(continuous_df, "feature1")
    assert lower < upper
    assert lower >= 100.0
    assert upper <= 1001.0


# ---------------------------------------------------------------------------
# compute_colormap — continuous
# ---------------------------------------------------------------------------


def test_compute_colormap_continuous_keys(continuous_df):
    color_dict, props = compute_colormap(
        continuous_df, "feature1", "label", "viridis", 100.0, 1001.0
    )
    assert None in color_dict
    for label_id in [1, 2, 3, 4, 5, 6]:
        assert label_id in color_dict


def test_compute_colormap_continuous_rgba_shape(continuous_df):
    color_dict, _ = compute_colormap(
        continuous_df, "feature1", "label", "viridis", 100.0, 1001.0
    )
    for key, rgba in color_dict.items():
        assert len(rgba) == 4, f"Expected 4 channels for label {key}"


def test_compute_colormap_continuous_properties(continuous_df):
    _, props = compute_colormap(
        continuous_df, "feature1", "label", "viridis", 100.0, 1001.0
    )
    assert "feature1" in props


def test_compute_colormap_continuous_clipping(continuous_df):
    # All values below lower → all map to cmap(0), i.e. same color
    color_dict, _ = compute_colormap(
        continuous_df, "feature1", "label", "viridis", 5000.0, 10000.0
    )
    colors = [color_dict[i] for i in [1, 2, 3, 4, 5, 6]]
    for c in colors:
        np.testing.assert_array_almost_equal(c, colors[0], decimal=4)


# ---------------------------------------------------------------------------
# compute_colormap — categorical
# ---------------------------------------------------------------------------


def test_compute_colormap_categorical_keys(categorical_df):
    color_dict, props = compute_colormap(
        categorical_df, "cell_type", "label", "tab10 (10)", 0.0, 1.0
    )
    assert None in color_dict
    for label_id in [1, 2, 3, 4, 5, 6]:
        assert label_id in color_dict


def test_compute_colormap_categorical_same_category_same_color(categorical_df):
    color_dict, _ = compute_colormap(
        categorical_df, "cell_type", "label", "tab10 (10)", 0.0, 1.0
    )
    # labels 1, 2, 3 are all type_A → same color
    np.testing.assert_array_equal(color_dict[1], color_dict[2])
    np.testing.assert_array_equal(color_dict[1], color_dict[3])


def test_compute_colormap_categorical_different_categories_differ(categorical_df):
    color_dict, _ = compute_colormap(
        categorical_df, "cell_type", "label", "tab10 (10)", 0.0, 1.0
    )
    # type_A (label 1) vs type_B (label 4) vs type_C (label 6)
    assert not np.array_equal(color_dict[1], color_dict[4])
    assert not np.array_equal(color_dict[1], color_dict[6])
    assert not np.array_equal(color_dict[4], color_dict[6])


def test_compute_colormap_categorical_tab10_colors(categorical_df):
    color_dict, _ = compute_colormap(
        categorical_df, "cell_type", "label", "tab10 (10)", 0.0, 1.0
    )
    expected = mpl.colormaps["tab10"](np.arange(3))
    np.testing.assert_array_almost_equal(color_dict[1], expected[0], decimal=4)
    np.testing.assert_array_almost_equal(color_dict[4], expected[1], decimal=4)
    np.testing.assert_array_almost_equal(color_dict[6], expected[2], decimal=4)


def test_compute_colormap_categorical_label_colormap(categorical_df):
    # Should not raise even with the fallback colormap
    color_dict, _ = compute_colormap(
        categorical_df, "cell_type", "label", "label_colormap", 0.0, 1.0
    )
    assert None in color_dict
    assert len(color_dict) > 1
