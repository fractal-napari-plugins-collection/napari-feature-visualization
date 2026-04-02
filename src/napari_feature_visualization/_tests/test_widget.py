"""Tests for FeatureVisWidget and its colormap-bar helpers.

Test organisation
-----------------
* ``_compute_ticks`` and ``_fmt`` — pure functions, no Qt required.
  These hold all the tick-placement logic, so they get thorough parametrized
  coverage.
* ``_make_colorbar_pixmap`` — needs a Qt application but not a napari viewer.
* ``_ColormapBar`` — state assertions + a ``widget.grab()`` smoke test.
  Pixel-exact rendering checks are deliberately avoided: they are fragile
  across platforms and screen scales and add little value compared to the
  logic tests above.
* ``FeatureVisWidget`` — integration tests that require both Qt and a viewer.
"""

import numpy as np
import pandas as pd
import pytest
from napari.utils.colormaps import DirectLabelColormap

from napari_feature_visualization._widget import (
    FeatureVisWidget,
    _ColormapBar,
    _compute_ticks,
    _fmt,
    _make_colorbar_pixmap,
)


def _make_label_img():
    lbl = np.zeros((50, 50), dtype="uint16")
    lbl[5:10, 5:10] = 1
    lbl[15:20, 5:10] = 2
    lbl[25:30, 5:10] = 3
    lbl[5:10, 15:20] = 4
    lbl[15:20, 15:20] = 5
    lbl[25:30, 15:20] = 6
    return lbl


def _make_feature_df():
    return pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5, 6],
            "feature1": [100.0, 200.0, 300.0, 500.0, 900.0, 1001.0],
            "cell_type": ["A", "A", "B", "B", "C", "C"],
        }
    )


# ---------------------------------------------------------------------------
# _fmt — pure function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (1.0, "1"),
        (1000.0, "1000"),
        (0.1, "0.1"),
        (0.001, "0.001"),
        (0.00012345, "0.0001234"),
        (1234567.0, "1.235e+06"),
        (-42.5, "-42.5"),
    ],
)
def test_fmt(value, expected):
    assert _fmt(value) == expected


# ---------------------------------------------------------------------------
# _compute_ticks — pure function
# ---------------------------------------------------------------------------


def test_compute_ticks_includes_endpoints():
    ticks = _compute_ticks(0.0, 1.0)
    assert ticks[0] == pytest.approx(0.0)
    assert ticks[-1] == pytest.approx(1.0)


def test_compute_ticks_degenerate_equal():
    # lower == upper → single-element list
    assert _compute_ticks(5.0, 5.0) == [5.0]


def test_compute_ticks_degenerate_inverted():
    # lower > upper → single-element list (lower)
    assert _compute_ticks(10.0, 5.0) == [10.0]


def test_compute_ticks_max_ticks_respected():
    for max_ticks in (2, 3, 5, 7):
        ticks = _compute_ticks(0.0, 100.0, max_ticks=max_ticks)
        assert len(ticks) <= max_ticks, f"max_ticks={max_ticks}, got {ticks}"


def test_compute_ticks_endpoints_only_when_max_ticks_2():
    ticks = _compute_ticks(0.0, 1.0, max_ticks=2)
    assert len(ticks) == 2
    assert ticks[0] == pytest.approx(0.0)
    assert ticks[-1] == pytest.approx(1.0)


def test_compute_ticks_interior_are_sorted():
    ticks = _compute_ticks(-50.0, 150.0)
    assert ticks == sorted(ticks)


def test_compute_ticks_interior_strictly_between_endpoints():
    ticks = _compute_ticks(0.0, 100.0)
    for t in ticks[1:-1]:
        assert 0.0 < t < 100.0


def test_compute_ticks_crosses_zero():
    ticks = _compute_ticks(-100.0, 100.0)
    assert 0.0 in ticks


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (0.0, 1.0),
        (0.0, 100.0),
        (-50.0, 50.0),
        (1e-3, 9e-3),  # small range
        (1e6, 2e6),  # large range
        (0.0, 1e-10),  # very small range
    ],
)
def test_compute_ticks_always_has_endpoints(lower, upper):
    ticks = _compute_ticks(lower, upper)
    assert ticks[0] == pytest.approx(lower)
    assert ticks[-1] == pytest.approx(upper)


# ---------------------------------------------------------------------------
# _make_colorbar_pixmap — needs Qt, no viewer
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_make_colorbar_pixmap_size():
    pixmap = _make_colorbar_pixmap("viridis", width=200, height=16)
    assert not pixmap.isNull()
    assert pixmap.width() == 200
    assert pixmap.height() == 16


@pytest.mark.usefixtures("qapp")
def test_make_colorbar_pixmap_default_size():
    pixmap = _make_colorbar_pixmap("plasma")
    assert not pixmap.isNull()
    assert pixmap.width() == 512
    assert pixmap.height() == 16


# ---------------------------------------------------------------------------
# _ColormapBar — state and smoke tests
# ---------------------------------------------------------------------------


def test_colormap_bar_initial_state(qtbot):
    bar = _ColormapBar()
    qtbot.addWidget(bar)
    assert bar._colormap_name == ""
    assert bar._lower == 0.0
    assert bar._upper == 1.0
    assert bar._gradient is None


def test_colormap_bar_set_colormap_updates_state(qtbot):
    bar = _ColormapBar()
    qtbot.addWidget(bar)
    bar.set_colormap("viridis", 10.0, 200.0)
    assert bar._colormap_name == "viridis"
    assert bar._lower == 10.0
    assert bar._upper == 200.0


def test_colormap_bar_new_colormap_invalidates_gradient(qtbot):
    """Changing to a different colormap clears the cached gradient pixmap."""
    bar = _ColormapBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.resize(200, bar.height())
    bar.set_colormap("viridis", 0.0, 1.0)
    bar.grab()  # triggers paintEvent, populates _gradient cache
    assert bar._gradient is not None

    bar.set_colormap("plasma", 0.0, 1.0)  # different colormap
    assert bar._gradient is None


def test_colormap_bar_same_colormap_keeps_gradient(qtbot):
    """Updating limits without changing the colormap keeps the cached gradient."""
    bar = _ColormapBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.resize(200, bar.height())
    bar.set_colormap("viridis", 0.0, 1.0)
    bar.grab()
    gradient_before = bar._gradient

    bar.set_colormap("viridis", 0.0, 50.0)  # same colormap, new limits
    assert bar._gradient is gradient_before


def test_colormap_bar_resize_invalidates_gradient(qtbot):
    """Resizing the widget invalidates the gradient (it is width-dependent)."""
    bar = _ColormapBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.resize(200, bar.height())
    bar.set_colormap("viridis", 0.0, 1.0)
    bar.grab()
    assert bar._gradient is not None

    bar.resize(300, bar.height())
    assert bar._gradient is None


def test_colormap_bar_renders_with_colormap(qtbot):
    """paintEvent completes without error once a colormap is set."""
    bar = _ColormapBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.resize(300, bar.height())
    bar.set_colormap("viridis", 0.0, 100.0)
    pixmap = bar.grab()
    assert not pixmap.isNull()


def test_colormap_bar_renders_without_colormap(qtbot):
    """paintEvent returns early (no error) when no colormap has been set."""
    bar = _ColormapBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.resize(300, bar.height())
    pixmap = bar.grab()  # _colormap_name is "" → paintEvent returns immediately
    assert not pixmap.isNull()


def test_colormap_bar_fixed_height(qtbot):
    """Widget height is fixed and consistent across instances."""
    bar1 = _ColormapBar()
    bar2 = _ColormapBar()
    qtbot.addWidget(bar1)
    qtbot.addWidget(bar2)
    assert bar1.height() == bar2.height()
    assert bar1.minimumHeight() == bar1.maximumHeight()  # truly fixed


# ---------------------------------------------------------------------------
# Layer-selection edge cases
# ---------------------------------------------------------------------------


def test_widget_no_layers(make_napari_viewer, qtbot):
    """Widget initialised with an empty viewer — no combos populated."""
    viewer = make_napari_viewer()
    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)
    assert "No Labels" in w._layer_name_label.text()
    assert w._feature_combo.count() == 0
    assert w._label_col_combo.count() == 0


def test_widget_non_label_layer(make_napari_viewer, qtbot):
    """Active layer is an image, not a Labels layer — widget shows 'no layer'."""
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((10, 10)))
    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)
    assert "No Labels" in w._layer_name_label.text()
    assert w._feature_combo.count() == 0


def test_widget_label_layer_without_features(make_napari_viewer, qtbot):
    """Labels layer present but has no feature columns — combos remain empty."""
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((10, 10), dtype="uint16"))
    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)
    assert w._feature_combo.count() == 0


def test_widget_switching_to_non_label_layer(make_napari_viewer, qtbot):
    """Switching from a features-bearing Labels layer to an image clears combos."""
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(_make_label_img(), features=_make_feature_df())
    image_layer = viewer.add_image(np.zeros((50, 50)))

    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)

    viewer.layers.selection.active = label_layer
    assert w._feature_combo.count() > 0

    viewer.layers.selection.active = image_layer
    assert w._feature_combo.count() == 0
    assert "No Labels" in w._layer_name_label.text()


# ---------------------------------------------------------------------------
# Multiple-layer scenarios
# ---------------------------------------------------------------------------


def test_widget_multiple_label_layers_switching(make_napari_viewer, qtbot):
    """Switching active label layer updates combos to that layer's columns."""
    viewer = make_napari_viewer()
    df1 = _make_feature_df()
    df2 = pd.DataFrame({"label": [1, 2], "other_feat": [10.0, 20.0]})

    layer1 = viewer.add_labels(_make_label_img(), name="layer1", features=df1)
    layer2 = viewer.add_labels(
        np.zeros((50, 50), dtype="uint16"), name="layer2", features=df2
    )

    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)

    viewer.layers.selection.active = layer1
    items1 = [w._feature_combo.itemText(i) for i in range(w._feature_combo.count())]
    assert "feature1" in items1
    assert "other_feat" not in items1

    viewer.layers.selection.active = layer2
    items2 = [w._feature_combo.itemText(i) for i in range(w._feature_combo.count())]
    assert "other_feat" in items2
    assert "feature1" not in items2


def test_widget_multiple_layers_selected(make_napari_viewer, qtbot):
    """Multiple layers selected simultaneously — no crash, active layer is used."""
    viewer = make_napari_viewer()
    df = _make_feature_df()
    layer1 = viewer.add_labels(_make_label_img(), name="layer1", features=df)
    layer2 = viewer.add_labels(_make_label_img(), name="layer2", features=df)

    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)

    # Select both; napari keeps one as 'active'
    viewer.layers.selection.add(layer1)
    viewer.layers.selection.add(layer2)

    # Widget should not crash and should reflect the active layer
    assert w._feature_combo.count() > 0 or "No Labels" in w._layer_name_label.text()


# ---------------------------------------------------------------------------
# Apply — layer mode
# ---------------------------------------------------------------------------


def test_widget_apply_layer_mode(make_napari_viewer, qtbot):
    """Apply in layer mode sets a DirectLabelColormap on the active layer."""
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(_make_label_img(), features=_make_feature_df())

    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)
    w._feature_combo.setCurrentText("feature1")
    w._label_col_combo.setCurrentText("label")
    w._apply()

    assert isinstance(label_layer.colormap, DirectLabelColormap)
    assert None in label_layer.colormap.color_dict
    for label_id in [1, 2, 3, 4, 5, 6]:
        assert label_id in label_layer.colormap.color_dict


def test_widget_apply_categorical(make_napari_viewer, qtbot):
    """Apply with a categorical feature uses a qualitative colormap."""
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(_make_label_img(), features=_make_feature_df())

    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)
    w._feature_combo.setCurrentText("cell_type")
    w._label_col_combo.setCurrentText("label")
    w._apply()

    assert isinstance(label_layer.colormap, DirectLabelColormap)
    # Labels 1 and 2 are both "A" → same color
    np.testing.assert_array_equal(
        label_layer.colormap.color_dict[1],
        label_layer.colormap.color_dict[2],
    )
    # "A" and "B" differ
    assert not np.array_equal(
        label_layer.colormap.color_dict[1],
        label_layer.colormap.color_dict[3],
    )


def test_widget_apply_no_layer_does_not_raise(make_napari_viewer, qtbot):
    """Calling _apply() with no active layer shows a warning but does not raise."""
    viewer = make_napari_viewer()
    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)
    w._apply()  # must not raise


# ---------------------------------------------------------------------------
# Apply — CSV mode
# ---------------------------------------------------------------------------


def test_widget_apply_csv_mode(make_napari_viewer, qtbot, tmp_path):
    """Apply in CSV mode sets colormap without modifying layer.features."""
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(_make_label_img())

    csv_path = tmp_path / "features.csv"
    _make_feature_df().to_csv(csv_path, index=False)

    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)

    w._csv_radio.setChecked(True)
    w._on_source_changed()
    w._csv_path = csv_path
    w._csv_path_label.setText(str(csv_path))
    w._refresh_columns()

    w._feature_combo.setCurrentText("feature1")
    w._label_col_combo.setCurrentText("label")

    features_before = label_layer.features.copy()
    w._apply()

    assert isinstance(label_layer.colormap, DirectLabelColormap)
    # CSV mode must NOT write feature data back to the layer
    pd.testing.assert_frame_equal(label_layer.features, features_before)


@pytest.mark.parametrize("feature", ["feature1", "cell_type"])
def test_widget_csv_mode_both_dtypes(make_napari_viewer, qtbot, tmp_path, feature):
    """CSV mode works for both continuous and categorical features."""
    viewer = make_napari_viewer()
    viewer.add_labels(_make_label_img())

    csv_path = tmp_path / "features.csv"
    _make_feature_df().to_csv(csv_path, index=False)

    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)
    w._csv_radio.setChecked(True)
    w._on_source_changed()
    w._csv_path = csv_path
    w._csv_path_label.setText(str(csv_path))
    w._refresh_columns()

    w._feature_combo.setCurrentText(feature)
    w._label_col_combo.setCurrentText("label")
    w._apply()  # must not raise


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------


def test_widget_live_mode_applies_on_colormap_change(make_napari_viewer, qtbot):
    """With live mode on, changing the colormap auto-applies."""
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(_make_label_img(), features=_make_feature_df())

    w = FeatureVisWidget(viewer)
    qtbot.addWidget(w)
    w._feature_combo.setCurrentText("feature1")
    w._label_col_combo.setCurrentText("label")
    w._live_checkbox.setChecked(True)

    # Pick any colormap different from the current one
    current = w._colormap_combo.currentText()
    for i in range(w._colormap_combo.count()):
        candidate = w._colormap_combo.itemText(i)
        if candidate != current:
            w._colormap_combo.setCurrentText(candidate)
            break

    assert isinstance(label_layer.colormap, DirectLabelColormap)
