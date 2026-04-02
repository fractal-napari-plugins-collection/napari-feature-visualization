import numpy as np
import pandas as pd

from napari_feature_visualization._sample_data import make_sample_data


def test_make_sample_data():
    result = make_sample_data()
    assert len(result) == 1

    data, kwargs, layer_type = result[0]
    assert layer_type == "labels"
    assert data.dtype == np.uint16
    assert data.shape == (50, 50)
    assert data.max() == 6

    features = kwargs["features"]
    assert isinstance(features, pd.DataFrame)
    assert set(features.columns) >= {"label", "feature1", "feature2", "cell_type"}
    assert len(features) == 7

    # continuous columns are numeric
    assert pd.api.types.is_numeric_dtype(features["feature1"])
    assert pd.api.types.is_numeric_dtype(features["feature2"])

    # categorical column has multiple categories
    assert features["cell_type"].nunique() > 1
