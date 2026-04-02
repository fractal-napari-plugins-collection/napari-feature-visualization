import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from napari_feature_visualization.feature_vis import feature_vis


def create_label_img():
    shape = (50, 50)
    lbl_img = np.zeros(shape, dtype="uint16")
    lbl_img[5:10, 5:10] = 1
    lbl_img[15:20, 5:10] = 2
    lbl_img[25:30, 5:10] = 3
    lbl_img[5:10, 15:20] = 4
    lbl_img[15:20, 15:20] = 5
    lbl_img[25:30, 15:20] = 6
    return lbl_img


def create_feature_df():
    return pd.DataFrame(
        {
            "test": [-100, 200, 300, 500, 900, 300],
            "label": [1, 2, 3, 4, 5, 6],
            "index": [1, 2, 3, 4, 5, 6],
            "feature1": [100, 200, 300, 500, 900, 1001],
            "feature2": [2200, 2100, 2000, 1500, 1300, 1001],
            "predicted_cell_type": [
                "cell_type1",
                "cell_type1",
                "cell_type1",
                "cell_type2",
                "cell_type2",
                "cell_type2",
            ],
        }
    )


@pytest.mark.parametrize("load_features_from", ["CSV File", "Layer Properties"])
def test_feature_vis_widget(make_napari_viewer, load_features_from, tmp_path):
    lbl_img = create_label_img()
    df = create_feature_df()
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img)
    widget = feature_vis()

    if load_features_from == "CSV File":
        csv_path = tmp_path / "example_data.csv"
        df.to_csv(csv_path, index=False)
        widget(
            label_layer=label_layer,
            load_features_from=load_features_from,
            DataFrame=csv_path,
            label_column="label",
            feature="feature1",
        )
    else:
        label_layer.features = df
        widget(
            label_layer=label_layer,
            load_features_from=load_features_from,
            feature="feature1",
            label_column="label",
        )

    # 6 labels + None (background) + 1 for the zero-index gap = 8 entries
    assert len(label_layer.colormap.color_dict) == 8
    np.testing.assert_array_almost_equal(
        label_layer.colormap.color_dict[3],
        np.array([0.230223, 0.321297, 0.545488, 1.0]),
        decimal=4,
    )


def test_feature_vis_categorical(make_napari_viewer):
    lbl_img = create_label_img()
    df = create_feature_df()
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img)
    label_layer.features = df

    widget = feature_vis()
    widget(
        label_layer=label_layer,
        load_features_from="Layer Properties",
        feature="predicted_cell_type",
        label_column="label",
        Colormap="tab10 (10)",
    )

    assert len(label_layer.colormap.color_dict) == 8

    expected = mpl.colormaps["tab10"](np.arange(2))
    c1 = label_layer.colormap.color_dict[1]  # label 1 → cell_type1
    c2 = label_layer.colormap.color_dict[4]  # label 4 → cell_type2

    np.testing.assert_array_almost_equal(c1, expected[0], decimal=4)
    np.testing.assert_array_almost_equal(c2, expected[1], decimal=4)
    assert not np.array_equal(c1, c2)
