"""napari feature visualization widget (magic_factory implementation).

Will be replaced by the Qt widget in _widget.py; kept during the transition.
"""

import pathlib

import napari
import pandas as pd
from magicgui import magic_factory
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS
from napari.utils.notifications import show_warning

from napari_feature_visualization._colormap import (
    check_default_label_column,
    compute_colormap,
    get_colormap_choices,
    get_contrast_limits,
    get_default_colormap,
)
from napari_feature_visualization.utils import get_df


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

    widget.feature._default_choices = get_feature_choices
    widget.label_column._default_choices = get_feature_choices

    @widget.DataFrame.changed.connect
    def update_df_columns(event):
        widget.feature.reset_choices()
        widget.label_column.reset_choices()
        widget.label_column.value = check_default_label_column(
            pd.DataFrame(columns=widget.feature.choices)
        )
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
            choices = get_colormap_choices(df, event)

            if pd.api.types.is_numeric_dtype(df[event]):
                lower, upper = get_contrast_limits(df, event)
                widget.lower_contrast_limit.value = lower
                widget.upper_contrast_limit.value = upper

            widget.Colormap.choices = choices
            if current_colormap not in choices:
                widget.Colormap.value = get_default_colormap(choices)
        except KeyError:
            pass


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

    # Validate that the label column has unique values
    labels = site_df[str(label_column)].astype(int)
    if labels.nunique() != len(site_df):
        show_warning(
            f"The selected label column '{label_column}' contains non-unique "
            "values. Please select a column where each row has a unique identifier."
        )
        return

    color_dict, label_properties = compute_colormap(
        site_df,
        feature,
        label_column,
        Colormap,
        lower_contrast_limit,
        upper_contrast_limit,
    )

    label_layer.colormap = DirectLabelColormap(color_dict=color_dict)

    if load_features_from == "CSV File":
        label_layer.properties = label_properties
