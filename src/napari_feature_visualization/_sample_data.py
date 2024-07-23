"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    shape = (50, 50)
    # Make dtype uint32 to have `_add_layer_from_data` function add it as a
    # labels layer
    lbl_img_np = np.zeros(shape).astype("uint32")
    lbl_img_np[5:10, 5:10] = 1
    lbl_img_np[15:20, 5:10] = 2
    lbl_img_np[25:30, 5:10] = 3
    lbl_img_np[5:10, 15:20] = 4
    lbl_img_np[15:20, 15:20] = 5
    lbl_img_np[25:30, 15:20] = 6

    # Dummy df for this test
    d = {
        "test": [-100, 200, 300, 500, 900, 300],
        "label": [1, 2, 3, 4, 5, 6],
        "feature1": [100, 200, 300, 500, 900, 1001],
        "feature2": [2200, 2100, 2000, 1500, 1300, 1001],
    }
    df = pd.DataFrame(data=d)
    return [(lbl_img_np, {"features": df})]
