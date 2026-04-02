"""Sample data for napari-feature-visualization.

Provides a label image with both continuous and categorical features
attached as layer properties, so users can immediately try the widget.
"""

import numpy as np
import pandas as pd


def make_sample_data():
    shape = (50, 50)
    lbl_img = np.zeros(shape, dtype="uint16")
    lbl_img[5:10, 5:10] = 1
    lbl_img[15:20, 5:10] = 2
    lbl_img[25:30, 5:10] = 3
    lbl_img[5:10, 15:20] = 4
    lbl_img[15:20, 15:20] = 5
    lbl_img[25:30, 15:20] = 6

    # Row 0 = background (label 0); napari's features API aligns by position.
    df = pd.DataFrame(
        {
            "label": [0, 1, 2, 3, 4, 5, 6],
            "feature1": [float("nan"), 100, 200, 300, 500, 900, 1001],
            "feature2": [float("nan"), 2200, 2100, 2000, 1500, 1300, 1001],
            "cell_type": [
                None,
                "type_A",
                "type_A",
                "type_A",
                "type_B",
                "type_B",
                "type_C",
            ],
        }
    )

    return [
        (
            lbl_img,
            {"name": "Sample Labels", "features": df},
            "labels",
        )
    ]
