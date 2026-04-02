"""Pure colormap logic — no Qt, no widget dependencies.

All functions here take plain DataFrames and return plain data structures,
making them easy to test without a napari viewer or Qt application.
"""

import matplotlib as mpl
import numpy as np
import pandas as pd
from napari.utils.colormaps import ensure_colormap, label_colormap
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS

# Qualitative matplotlib colormaps and their maximum number of distinct colors.
QUALITATIVE_CMAPS = {
    "Accent": 8,
    "Dark2": 8,
    "Paired": 12,
    "Pastel1": 9,
    "Pastel2": 8,
    "Set1": 9,
    "Set2": 8,
    "Set3": 12,
    "tab10": 10,
    "tab20": 20,
    "tab20b": 20,
    "tab20c": 20,
}


def check_default_label_column(df: pd.DataFrame) -> str:
    """Return the most likely label-ID column name, or '' if none found."""
    for name in ("label", "Label", "index"):
        if name in df.columns:
            return name
    return ""


def get_colormap_choices(df: pd.DataFrame, feature: str) -> list[str]:
    """Return the list of valid colormap names for the given feature column.

    Continuous features get all napari colormaps.
    Categorical features get only qualitative colormaps wide enough to cover
    the number of unique values, falling back to 'label_colormap'.
    """
    if pd.api.types.is_numeric_dtype(df[feature]):
        return list(AVAILABLE_COLORMAPS.keys())

    num_categories = df[feature].nunique()
    choices = [
        f"{cmap} ({max_colors})"
        for cmap, max_colors in QUALITATIVE_CMAPS.items()
        if max_colors >= num_categories
    ]
    return choices if choices else ["label_colormap"]


def get_default_colormap(choices: list[str]) -> str:
    """Return the preferred default from a list of colormap choices.

    Prefers 'viridis' for continuous colormaps, 'tab10' for categorical ones.
    """
    if "viridis" in choices:
        return "viridis"
    tab10 = next((c for c in choices if c.startswith("tab10 (")), None)
    if tab10:
        return tab10
    return choices[0]


def get_contrast_limits(df: pd.DataFrame, feature: str) -> tuple[float, float]:
    """Return (lower, upper) contrast limits as the 1st and 99th percentiles."""
    return (
        float(df[feature].quantile(0.01)),
        float(df[feature].quantile(0.99)),
    )


def compute_colormap(
    df: pd.DataFrame,
    feature: str,
    label_col: str,
    colormap_name: str,
    lower: float,
    upper: float,
) -> tuple[dict, dict]:
    """Map a feature column onto per-label RGBA colors.

    Parameters
    ----------
    df:
        DataFrame containing at least ``feature`` and ``label_col`` columns.
    feature:
        Column to visualize.
    label_col:
        Column whose integer values correspond to label IDs in the layer.
    colormap_name:
        For continuous features: any napari colormap name.
        For categorical features: a ``'<name> (<n>)'`` string from
        ``get_colormap_choices``, or ``'label_colormap'``.
    lower, upper:
        Contrast limits used to rescale continuous features to [0, 1].

    Returns
    -------
    color_dict:
        ``{label_id: rgba_array}`` mapping suitable for
        ``DirectLabelColormap(color_dict=...)``. Always includes
        ``None -> black`` for unlabeled pixels.
    label_properties:
        ``{feature_name: array}`` mapping suitable for
        ``label_layer.properties``.
    """
    site_df = df.copy()
    site_df["_label"] = site_df[label_col].astype(int)

    if pd.api.types.is_numeric_dtype(site_df[feature]):
        color_dict, label_properties = _continuous(
            site_df, feature, colormap_name, lower, upper
        )
    else:
        color_dict, label_properties = _categorical(site_df, feature, colormap_name)

    color_dict[None] = np.array([0.0, 0.0, 0.0, 1.0])
    return color_dict, label_properties


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _continuous(
    df: pd.DataFrame,
    feature: str,
    colormap_name: str,
    lower: float,
    upper: float,
) -> tuple[dict, dict]:
    scaled = ((df[feature] - lower) / (upper - lower)).clip(0, 1)
    cmap = ensure_colormap(colormap_name)
    colors = cmap.map(scaled.values)

    max_label = df["_label"].max()
    props = np.zeros(max_label + 1)
    props[df["_label"]] = df[feature]
    label_properties = {feature: np.round(props, decimals=2)}

    color_dict = dict(zip(df["_label"], colors, strict=False))
    return color_dict, label_properties


def _categorical(
    df: pd.DataFrame,
    feature: str,
    colormap_name: str,
) -> tuple[dict, dict]:
    # Separate valid rows from NaN rows — NaN is not a category.
    valid_mask = df[feature].notna()
    valid_df = df[valid_mask]

    unique_vals = valid_df[feature].unique()
    num_categories = len(unique_vals)

    # Map each category to a 1-based integer (0 is reserved for background)
    feature_map = {val: i + 1 for i, val in enumerate(unique_vals)}
    mapped = valid_df[feature].map(feature_map)

    cmap_name = colormap_name.split(" (")[0]

    # Convert to a plain int numpy array: pd.Series.map() may return float64
    # (when result is homogeneous) or preserve Categorical dtype, both of which
    # break numpy fancy indexing.
    indices = np.asarray(mapped, dtype=int)

    if cmap_name == "label_colormap" or cmap_name not in QUALITATIVE_CMAPS:
        cat_cmap = label_colormap(num_categories + 1)
        colors = cat_cmap.map(indices)
    else:
        mpl_cmap = mpl.colormaps[cmap_name]
        sampled = mpl_cmap(np.arange(num_categories))
        colors = sampled[indices - 1]

    max_label = df["_label"].max()
    props = np.zeros(max_label + 1, dtype=object)
    props[df["_label"]] = df[feature]
    label_properties = {feature: props}

    color_dict = dict(zip(valid_df["_label"], colors, strict=False))

    # Labels with NaN feature values → transparent (no data, not background)
    for label_id in df.loc[~valid_mask, "_label"]:
        color_dict[int(label_id)] = np.array([0.0, 0.0, 0.0, 0.0])

    return color_dict, label_properties
