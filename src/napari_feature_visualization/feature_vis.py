"""napari feature visualization widget"""

import pathlib

import matplotlib as mpl
import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.utils.colormaps import ensure_colormap, label_colormap
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS
from napari.utils.notifications import show_warning
from packaging import version

from napari_feature_visualization.utils import get_df

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


def check_default_label_column(df):
    if "label" in df:
        return "label"
    elif "Label" in df:
        return "Label"
    elif "index" in df:
        return "index"
    return ""


def _init(widget):
    def get_feature_choices(*args):
        if widget.load_features_from.value == "CSV File":
            try:
                df = get_df(widget.DataFrame.value)
                return list(df.columns)
            except OSError:
                return [""]
        else:
            try:
                df = pd.DataFrame(widget.label_layer.value.properties)
                return list(df.columns)
            except AttributeError:
                return [""]

    # set feature and label_column "default choices"
    # to be a function that gets the column names of the
    # currently loaded dataframe
    widget.feature._default_choices = get_feature_choices
    widget.label_column._default_choices = get_feature_choices

    @widget.DataFrame.changed.connect
    def update_df_columns(event):
        # event value will be the new path
        # get_df will give you the cached df
        # ...reset_choices() calls the "get_feature_choices" function above
        # to keep them updated with the current dataframe
        widget.feature.reset_choices()
        widget.label_column.reset_choices()
        features = widget.feature.choices
        widget.label_column.value = check_default_label_column(features)

        # if load_features_from is toggled, make the widget.DataFrame disappear
        if widget.load_features_from.value == "Layer Properties":
            widget.DataFrame.hide()
        else:
            widget.DataFrame.show()

    widget.load_features_from.changed.connect(update_df_columns)

    @widget.feature.changed.connect
    def update_rescaling(event):
        if widget.load_features_from.value == "CSV File":
            if widget.DataFrame.value != pathlib.Path("."):
                df = get_df(widget.DataFrame.value)
            else:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame(widget.label_layer.value.properties)

        try:
            current_colormap = widget.Colormap.value
            if pd.api.types.is_numeric_dtype(df[event]):
                quantiles = (0.01, 0.99)
                widget.lower_contrast_limit.value = df[event].quantile(
                    quantiles[0]
                )
                widget.upper_contrast_limit.value = df[event].quantile(
                    quantiles[1]
                )
                new_choices = list(AVAILABLE_COLORMAPS.keys())
                widget.Colormap.choices = new_choices
                if (
                    current_colormap not in new_choices
                    and "viridis" in new_choices
                ):
                    widget.Colormap.value = "viridis"
            else:
                num_categories = df[event].nunique()
                qual_choices = [
                    f"{cmap} ({max_colors})"
                    for cmap, max_colors in QUALITATIVE_CMAPS.items()
                    if max_colors >= num_categories
                ]
                if not qual_choices:
                    widget.Colormap.choices = ["label_colormap"]
                    widget.Colormap.value = "label_colormap"
                else:
                    widget.Colormap.choices = qual_choices
                    if current_colormap not in qual_choices:
                        tab10_choice = next(
                            (
                                c
                                for c in qual_choices
                                if c.startswith("tab10 (")
                            ),
                            None,
                        )
                        widget.Colormap.value = (
                            tab10_choice if tab10_choice else qual_choices[0]
                        )

        except KeyError:
            # Don't update the limits if a feature name is entered that isn't in the dataframe
            pass


# TODO: Set better limits for contrast_limits
@magic_factory(
    call_button="Apply Feature Colormap",
    layout="vertical",
    load_features_from={
        "widget_type": "RadioButtons",
        "choices": ["Layer Properties", "CSV File"],
        "value": "Layer Properties",
    },
    DataFrame={"mode": "r"},
    lower_contrast_limit={"min": -100000000, "max": 100000000},
    upper_contrast_limit={"min": -100000000, "max": 100000000},
    feature={"choices": [""]},
    Colormap={"choices": list(AVAILABLE_COLORMAPS.keys())},
    label_column={"choices": [""]},
    widget_init=_init,
)
def feature_vis(
    label_layer: napari.layers.Labels,
    load_features_from: str,
    DataFrame: pathlib.Path,
    feature="",
    Colormap="viridis",
    label_column="",
    lower_contrast_limit: float = 100,
    upper_contrast_limit: float = 900,
):
    if load_features_from == "CSV File":
        site_df = get_df(DataFrame)
    else:
        site_df = pd.DataFrame(label_layer.properties)

    if label_column == "":
        label_column = check_default_label_column(site_df)

    site_df.loc[:, "label"] = site_df[str(label_column)].astype(int)
    # Check that there is one unique label for every entry in the dataframe
    if len(site_df["label"].unique()) != len(site_df):
        show_warning(
            f"The selected label column '{label_column}' contains non-unique values. "
            "Please select a column where each row has a unique identifier."
        )
        return

    is_continuous = pd.api.types.is_numeric_dtype(site_df[feature])
    properties_array = np.zeros(site_df["label"].max() + 1)

    if is_continuous:
        # Rescale feature between 0 & 1 to make a colormap
        site_df["feature_scaled"] = (
            site_df[feature] - lower_contrast_limit
        ) / (upper_contrast_limit - lower_contrast_limit)
        # Cap the measurement between 0 & 1
        site_df.loc[site_df["feature_scaled"] < 0, "feature_scaled"] = 0
        site_df.loc[site_df["feature_scaled"] > 1, "feature_scaled"] = 1

        cmap = ensure_colormap(Colormap)
        colors = cmap.map(site_df["feature_scaled"].values)

        properties_array[site_df["label"]] = site_df[feature]
        label_properties = {feature: np.round(properties_array, decimals=2)}

        colormap = dict(zip(site_df["label"], colors, strict=False))
    else:
        # Categorical feature
        unique_features = site_df[feature].unique()
        num_categories = len(unique_features)

        # Create a mapping from categorical feature to a label colormap
        # Start at 1 because 0 is reserved for background/None in label colormaps
        feature_map = {feat: i + 1 for i, feat in enumerate(unique_features)}
        mapped_features = site_df[feature].map(feature_map)

        cmap_name = Colormap.split(" (")[0]

        if cmap_name == "label_colormap" or cmap_name not in QUALITATIVE_CMAPS:
            # napari's label_colormap generates a categorical colormap
            categorical_cmap = label_colormap(num_categories + 1)
            colors = categorical_cmap.map(mapped_features.values)
        else:
            mpl_cmap = mpl.colormaps[cmap_name]

            sampled_colors = mpl_cmap(np.arange(num_categories))

            # mapped_features is 1-indexed (from 1 to num_categories)
            indices = mapped_features.values - 1
            colors = sampled_colors[indices]

        # Object dtype arrays are fine for properties
        properties_array = np.zeros(site_df["label"].max() + 1, dtype=object)
        properties_array[site_df["label"]] = site_df[feature]
        label_properties = {feature: properties_array}

        colormap = dict(zip(site_df["label"], colors, strict=False))

    # Show missing objects as black
    colormap[None] = [0.0, 0.0, 0.0, 1.0]

    # Handle different colormap APIs depending on the napari version
    napari_version = version.parse(napari.__version__)
    if napari_version >= version.parse("0.4.19"):
        from napari.utils.colormaps import DirectLabelColormap

        label_layer.colormap = DirectLabelColormap(color_dict=colormap)
    else:
        label_layer.color = colormap

    if load_features_from == "CSV File":
        try:
            label_layer.properties = label_properties
        except UnboundLocalError:
            # If a napari version before 0.4.8 is used, this can't be displayed yet
            # This this thread on the bug: https://github.com/napari/napari/issues/2477
            print("Can't set label properties in napari versions < 0.4.8")
