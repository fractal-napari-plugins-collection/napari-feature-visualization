import napari
import numpy as np
import pandas as pd
import pytest
import matplotlib as mpl
from packaging import version

from napari_feature_visualization.feature_vis import feature_vis


def create_label_img():
    shape = (50, 50)
    lbl_img_np = np.zeros(shape).astype("uint16")
    lbl_img_np[5:10, 5:10] = 1
    lbl_img_np[15:20, 5:10] = 2
    lbl_img_np[25:30, 5:10] = 3
    lbl_img_np[5:10, 15:20] = 4
    lbl_img_np[15:20, 15:20] = 5
    lbl_img_np[25:30, 15:20] = 6
    return lbl_img_np


def create_feature_df():
    d = {
        "test": [-100, 200, 300, 500, 900, 300],
        "label": [1, 2, 3, 4, 5, 6],
        "index": [1, 2, 3, 4, 5, 6],
        "feature1": [100, 200, 300, 500, 900, 1001],
        "feature2": [2200, 2100, 2000, 1500, 1300, 1001],
        "predicted_cell_type": ["cell_type1", "cell_type1", "cell_type1", "cell_type2", "cell_type2", "cell_type2"],
    }
    df = pd.DataFrame(data=d)
    # Ensure index is the labels for correct matching
    return df


@pytest.mark.parametrize(
    "load_features_from", ["CSV File", "Layer Properties"]
)
def test_feature_vis_widget(make_napari_viewer, load_features_from):
    lbl_img = create_label_img()
    df = create_feature_df()
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img)

    feature_vis_widget = feature_vis()

    if load_features_from == "CSV File":
        df.to_csv("example_data.csv", index=False)
        # if we "call" this object, it'll execute our function
        feature_vis_widget(
            label_layer=label_layer,
            load_features_from=load_features_from,
            DataFrame="example_data.csv",
            label_column="label",
            feature="feature1",
        )
    elif load_features_from == "Layer Properties":
        label_layer.features = df
        # if we "call" this object, it'll execute our function
        feature_vis_widget(
            label_layer=label_layer,
            load_features_from=load_features_from,
            feature="feature1",
            label_column="label",
        )

    napari_version = version.parse(napari.__version__)
    if napari_version >= version.parse("0.4.19"):
        assert len(label_layer.colormap.color_dict) == 8
        np.testing.assert_array_almost_equal(
            label_layer.colormap.color_dict[3],
            np.array([0.230223, 0.321297, 0.545488, 1.0]),
            decimal=4
        )
    else:
        assert len(label_layer.colormap.colors) == 7
        np.testing.assert_array_almost_equal(
            label_layer.colormap.colors[3],
            np.array([0.230223, 0.321297, 0.545488, 1.0]),
            decimal=4
        )


def test_feature_vis_categorical(make_napari_viewer):
    lbl_img = create_label_img()
    df = create_feature_df()
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img)
    label_layer.features = df

    feature_vis_widget = feature_vis()
    feature_vis_widget(
        label_layer=label_layer,
        load_features_from="Layer Properties",
        feature="predicted_cell_type",
        label_column="label",
        Colormap="tab10 (10)"
    )

    napari_version = version.parse(napari.__version__)

    if napari_version >= version.parse("0.4.19"):
        assert len(label_layer.colormap.color_dict) == 8

        expected_colors = mpl.colormaps["tab10"](np.arange(2))

        c1 = label_layer.colormap.color_dict[1] # label=1 -> cell_type1
        c2 = label_layer.colormap.color_dict[4] # label=4 -> cell_type2

        # Verify against expected tab10 colors
        np.testing.assert_array_almost_equal(c1, expected_colors[0], decimal=4)
        np.testing.assert_array_almost_equal(c2, expected_colors[1], decimal=4)

        assert not np.array_equal(c1, c2)

    else:
        assert len(label_layer.colormap.colors) == 7

        expected_colors = mpl.colormaps["tab10"](np.arange(2))

        c1 = label_layer.colormap.colors[1] # label=1 -> cell_type1
        c2 = label_layer.colormap.colors[4] # label=4 -> cell_type2

        # Verify against expected tab10 colors
        np.testing.assert_array_almost_equal(c1, expected_colors[0], decimal=4)
        np.testing.assert_array_almost_equal(c2, expected_colors[1], decimal=4)

        assert not np.array_equal(c1, c2)

# def test_feature_vis_from_csv(make_napari_viewer):
#     lbl_img = create_label_img()
#     df = create_feature_df()

#     viewer = make_napari_viewer()
#     label_layer = viewer.add_labels(lbl_img)
#     label_layer.features = df

#     feature_vis_widget = feature_vis()

#     # if we "call" this object, it'll execute our function
#     feature_vis_widget(
#         label_layer=label_layer,
#         load_features_from="CSV File",
#         DataFrame="example_data.csv",
#         feature="feature1",
#     )

#     napari_version = version.parse(napari.__version__)
#     if napari_version >= version.parse("0.4.19"):
#         assert len(label_layer.colormap.color_dict) == 8
#         np.testing.assert_array_almost_equal(
#             label_layer.colormap.color_dict[3],
#             np.array([0.230223, 0.321297, 0.545488, 1.0]),
#             decimal=4
#         )
#     else:
#         assert len(label_layer.colormap.colors) == 7
#         np.testing.assert_array_almost_equal(
#             label_layer.colormap.colors[3],
#             np.array([0.230223, 0.321297, 0.545488, 1.0]),
#             decimal=4
#         )
