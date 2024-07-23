import numpy as np
import pandas as pd

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


def test_image_threshold_widget(make_napari_viewer):
    lbl_img = create_label_img()

    # Dummy df for this test
    d = {
        "test": [-100, 200, 300, 500, 900, 300],
        "label": [1, 2, 3, 4, 5, 6],
        "feature1": [100, 200, 300, 500, 900, 1001],
        "feature2": [2200, 2100, 2000, 1500, 1300, 1001],
    }
    df = pd.DataFrame(data=d)

    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img)
    label_layer.features = df

    feature_vis_widget = feature_vis()

    # if we "call" this object, it'll execute our function
    feature_vis_widget(
        label_layer=label_layer,
        load_features_from="Layer Properties",
        feature="feature1",
    )
    assert len(label_layer.colormap.colors) == 7
    np.testing.assert_array_almost_equal(
        label_layer.colormap.colors[3],
        np.array([0.229739, 0.322361, 0.545706, 1.0]),
    )
