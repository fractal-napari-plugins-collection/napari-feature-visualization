"""Qt-based feature visualization widget.

Architecture notes
------------------
* Data source is either the active Labels layer's ``.features`` DataFrame or a
  user-selected CSV file.  The source toggle radio buttons control which path
  is used; ``_get_df()`` dispatches accordingly.
* The widget tracks ``viewer.layers.selection.events.active`` to always reflect
  the currently selected Labels layer without requiring the user to choose from
  a dropdown.
* **Do not add** ``from __future__ import annotations`` to this file.  napari
  injects the ``viewer`` argument by inspecting ``__init__``'s type annotations
  at runtime via ``inspect.signature()``.  The future-annotations import makes
  all annotations lazy strings, which breaks that mechanism.
"""

import math
import pathlib

import napari
import napari.layers
import numpy as np
import pandas as pd
from napari.utils.colormaps import DirectLabelColormap, ensure_colormap
from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPainter, QPalette, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleRangeSlider

from napari_feature_visualization._colormap import (
    check_default_label_column,
    compute_colormap,
    get_colormap_choices,
    get_contrast_limits,
    get_default_colormap,
)
from napari_feature_visualization.utils import get_df


def _fmt(v: float) -> str:
    """Format a contrast-limit value compactly for the slider labels."""
    return f"{v:.4g}"


def _make_colorbar_pixmap(
    colormap_name: str, width: int = 512, height: int = 16
) -> QPixmap:
    """Render a horizontal gradient for the given napari colormap name."""
    cmap = ensure_colormap(colormap_name)
    colors = cmap.map(np.linspace(0, 1, width))  # (width, 4) float64
    rgba = (colors * 255).clip(0, 255).astype(np.uint8)
    img_array = np.ascontiguousarray(np.tile(rgba[np.newaxis], (height, 1, 1)))
    h, w = img_array.shape[:2]
    qimg = QImage(img_array.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())  # type: ignore[return-value]


def _compute_ticks(lower: float, upper: float, max_ticks: int = 5) -> list[float]:
    """Return nicely rounded tick values spanning [lower, upper].

    Always includes the endpoints; fills in up to (max_ticks - 2) interior
    ticks at round intervals.
    """
    if lower >= upper:
        return [lower]
    span = upper - lower
    target_n = max(0, max_ticks - 2)  # desired interior count
    raw_step = span / (target_n + 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = magnitude  # fallback; always overwritten by the loop below
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = mult * magnitude
        if step >= raw_step:
            break
    eps = step * 1e-9
    first = math.ceil((lower + eps) / step) * step
    interior: list[float] = []
    val = first
    while val < upper - eps:
        # round away floating-point noise at the precision of the step
        interior.append(round(val, max(0, -int(math.floor(math.log10(step))) + 4)))
        val += step
    return [lower] + interior + [upper]


class _ColormapBar(QWidget):  # type: ignore[misc]
    """Horizontal colormap strip with axis tick annotations.

    Draws at the widget's actual width; caches the gradient pixmap and only
    regenerates it when the colormap name or width changes.
    """

    _BAR_H = 16
    _TICK_H = 5
    _TEXT_GAP = 2

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._colormap_name = ""
        self._lower = 0.0
        self._upper = 1.0
        self._gradient: QPixmap | None = None  # type: ignore[assignment]
        fm = self.fontMetrics()
        self.setFixedHeight(self._BAR_H + self._TICK_H + self._TEXT_GAP + fm.height())

    def set_colormap(self, name: str, lower: float, upper: float) -> None:
        if name != self._colormap_name:
            self._gradient = None  # invalidate gradient cache on colormap change
        self._colormap_name = name
        self._lower = lower
        self._upper = upper
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._gradient = None  # width changed → regenerate gradient
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._colormap_name:
            return
        w = self.width()
        if w <= 0:
            return

        painter = QPainter(self)

        # --- Gradient bar ---
        if self._gradient is None or self._gradient.width() != w:
            self._gradient = _make_colorbar_pixmap(self._colormap_name, w, self._BAR_H)
        painter.drawPixmap(0, 0, self._gradient)

        # --- Ticks and labels ---
        # Build candidate ticks, compute their pixel positions and label bounds,
        # then skip any interior tick whose label would overlap an already-drawn one.
        _GAP = 4  # minimum pixel clearance between adjacent labels

        ticks = _compute_ticks(self._lower, self._upper)
        fm = painter.fontMetrics()
        span = self._upper - self._lower
        n = len(ticks)
        text_y = self._BAR_H + self._TICK_H + self._TEXT_GAP + fm.ascent()

        def _tick_x(val: float) -> int:
            frac = (val - self._lower) / span if span > 0 else 0.0
            return int(frac * (w - 1))

        def _label_bounds(i: int, x: int, label: str) -> tuple[int, int]:
            tw = fm.horizontalAdvance(label)
            if i == 0:
                lx = x
            elif i == n - 1:
                lx = x - tw
            else:
                lx = max(0, min(x - tw // 2, w - tw))
            return lx, lx + tw

        # Endpoints are always drawn; seed the occupied-bounds list with them.
        drawn: list[tuple[int, int]] = [
            _label_bounds(0, _tick_x(ticks[0]), _fmt(ticks[0])),
            _label_bounds(n - 1, _tick_x(ticks[-1]), _fmt(ticks[-1])),
        ]
        selected = {0, n - 1}
        for i in range(1, n - 1):
            x = _tick_x(ticks[i])
            lo, hi = _label_bounds(i, x, _fmt(ticks[i]))
            if all(lo > db[1] + _GAP or hi < db[0] - _GAP for db in drawn):
                drawn.append((lo, hi))
                selected.add(i)

        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        for i in sorted(selected):
            val = ticks[i]
            x = _tick_x(val)
            label = _fmt(val)
            lx, _ = _label_bounds(i, x, label)
            painter.drawLine(x, self._BAR_H, x, self._BAR_H + self._TICK_H)
            painter.drawText(lx, text_y, label)

        painter.end()


class FeatureVisWidget(QWidget):  # type: ignore[misc]
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self._csv_path: pathlib.Path | None = None
        self._current_layer: napari.layers.Labels | None = None

        self._build_ui()
        self._connect_signals()
        # Populate from whichever layer is already active
        self._on_active_layer_changed()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        self.setLayout(root)

        # --- Source selection ---
        source_row = QHBoxLayout()
        self._layer_radio = QRadioButton("Layer Properties")
        self._csv_radio = QRadioButton("CSV File")
        self._layer_radio.setChecked(True)
        self._source_group = QButtonGroup(self)
        self._source_group.addButton(self._layer_radio)
        self._source_group.addButton(self._csv_radio)
        source_row.addWidget(self._layer_radio)
        source_row.addWidget(self._csv_radio)
        root.addLayout(source_row)

        # --- CSV file picker (hidden until CSV mode is selected) ---
        self._csv_row = QWidget()
        csv_layout = QHBoxLayout()
        csv_layout.setContentsMargins(0, 0, 0, 0)
        self._csv_row.setLayout(csv_layout)
        self._csv_path_label = QLabel("No file selected")
        self._csv_path_label.setWordWrap(True)
        self._csv_browse_btn = QPushButton("Browse…")
        csv_layout.addWidget(self._csv_path_label, stretch=1)
        csv_layout.addWidget(self._csv_browse_btn)
        root.addWidget(self._csv_row)
        self._csv_row.hide()

        # --- Active layer indicator ---
        self._layer_name_label = QLabel("No Labels layer selected")
        root.addWidget(self._layer_name_label)

        # --- Feature / label-column / colormap dropdowns ---
        form = QFormLayout()
        self._feature_combo = QComboBox()
        form.addRow("Feature:", self._feature_combo)

        self._label_col_combo = QComboBox()
        form.addRow("Label column:", self._label_col_combo)

        self._colormap_combo = QComboBox()
        form.addRow("Colormap:", self._colormap_combo)
        root.addLayout(form)

        # --- Contrast limits (hidden for categorical features) ---
        # Layout: "Contrast limits:" label on its own line, then
        #         [lower_val] [===slider===] [upper_val]
        self._limits_widget = QWidget()
        limits_layout = QVBoxLayout()
        limits_layout.setContentsMargins(0, 4, 0, 0)
        limits_layout.setSpacing(2)
        limits_header = QHBoxLayout()
        limits_header.setContentsMargins(0, 0, 0, 0)
        limits_header.addWidget(QLabel("Contrast limits:"))
        limits_header.addStretch()
        self._colorbar_checkbox = QCheckBox("Show colorbar")
        limits_header.addWidget(self._colorbar_checkbox)
        limits_layout.addLayout(limits_header)

        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(0, 0, 0, 0)

        self._contrast_lower_edit = QLineEdit("0")
        self._contrast_lower_edit.setFixedWidth(60)
        self._contrast_lower_edit.setAlignment(Qt.AlignmentFlag.AlignRight)

        self._contrast_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._contrast_slider.setRange(0.0, 1000.0)
        self._contrast_slider.setValue((100.0, 900.0))

        self._contrast_upper_edit = QLineEdit("1000")
        self._contrast_upper_edit.setFixedWidth(60)
        self._contrast_upper_edit.setAlignment(Qt.AlignmentFlag.AlignLeft)

        slider_row.addWidget(self._contrast_lower_edit)
        slider_row.addWidget(self._contrast_slider)
        slider_row.addWidget(self._contrast_upper_edit)
        limits_layout.addLayout(slider_row)

        self._colorbar_bar = _ColormapBar(self)
        self._colorbar_bar.hide()
        limits_layout.addWidget(self._colorbar_bar)

        self._limits_widget.setLayout(limits_layout)
        root.addWidget(self._limits_widget)

        # --- Live mode + Apply button ---
        bottom_row = QHBoxLayout()
        self._live_checkbox = QCheckBox("Live update")
        self._apply_btn = QPushButton("Apply Feature Colormap")
        bottom_row.addWidget(self._live_checkbox)
        bottom_row.addWidget(self._apply_btn)
        root.addLayout(bottom_row)
        root.addStretch()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._viewer.layers.selection.events.active.connect(
            self._on_active_layer_changed
        )
        self._source_group.buttonClicked.connect(self._on_source_changed)
        self._csv_browse_btn.clicked.connect(self._browse_csv)
        self._feature_combo.currentTextChanged.connect(self._on_feature_changed)
        self._colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self._colorbar_checkbox.toggled.connect(self._on_colorbar_toggled)
        self._contrast_slider.valueChanged.connect(self._on_slider_changed)
        self._contrast_slider.sliderReleased.connect(self._maybe_apply)
        self._contrast_slider.sliderReleased.connect(
            lambda: self._update_colorbar(self._colormap_combo.currentText())
        )
        self._contrast_lower_edit.editingFinished.connect(self._on_lower_edited)
        self._contrast_upper_edit.editingFinished.connect(self._on_upper_edited)
        self._apply_btn.clicked.connect(self._apply)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._viewer.layers.selection.events.active.disconnect(
            self._on_active_layer_changed
        )
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_active_layer_changed(self, event=None) -> None:
        active = self._viewer.layers.selection.active
        if isinstance(active, napari.layers.Labels):
            self._current_layer = active
            self._layer_name_label.setText(f"Layer: {active.name}")
            if self._layer_radio.isChecked():
                self._refresh_columns()
        else:
            self._current_layer = None
            self._layer_name_label.setText("No Labels layer selected")
            if self._layer_radio.isChecked():
                self._clear_columns()

    def _on_source_changed(self) -> None:
        self._csv_row.setVisible(self._csv_radio.isChecked())
        self._refresh_columns()

    def _browse_csv(self) -> None:
        dlg = QFileDialog(self, "Open CSV file")
        dlg.setNameFilter("CSV files (*.csv);;All files (*)")
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            self._csv_path = pathlib.Path(path)
            self._csv_path_label.setText(path)
            self._refresh_columns()

    def _on_feature_changed(self, feature: str) -> None:
        if not feature:
            return
        df = self._get_df()
        if df is None or feature not in df.columns:
            return

        is_continuous = pd.api.types.is_numeric_dtype(df[feature])
        choices = get_colormap_choices(df, feature)
        current_cmap = self._colormap_combo.currentText()

        self._colormap_combo.blockSignals(True)
        self._colormap_combo.clear()
        self._colormap_combo.addItems(choices)
        if current_cmap in choices:
            self._colormap_combo.setCurrentText(current_cmap)
        else:
            self._colormap_combo.setCurrentText(get_default_colormap(choices))
        self._colormap_combo.blockSignals(False)

        self._limits_widget.setVisible(is_continuous)
        if is_continuous:
            feat_min = float(df[feature].min())
            feat_max = float(df[feature].max())
            lower, upper = get_contrast_limits(df, feature)
            self._contrast_slider.setRange(feat_min, feat_max)
            self._contrast_slider.setValue((lower, upper))
            # valueChanged updates the labels, but set them explicitly in case
            # the value didn't change (e.g. same feature re-selected)
            self._contrast_lower_edit.setText(_fmt(lower))
            self._contrast_upper_edit.setText(_fmt(upper))
            self._update_colorbar(self._colormap_combo.currentText())
        self._maybe_apply()

    def _on_slider_changed(self, values: tuple) -> None:
        lower, upper = values
        self._contrast_lower_edit.setText(_fmt(lower))
        self._contrast_upper_edit.setText(_fmt(upper))

    def _on_lower_edited(self) -> None:
        lower, upper = self._contrast_slider.value()
        try:
            value = float(self._contrast_lower_edit.text())
        except ValueError:
            self._contrast_lower_edit.setText(_fmt(lower))
            return
        value = max(self._contrast_slider.minimum(), min(value, upper))
        self._contrast_slider.setValue((value, upper))
        self._maybe_apply()

    def _on_upper_edited(self) -> None:
        lower, upper = self._contrast_slider.value()
        try:
            value = float(self._contrast_upper_edit.text())
        except ValueError:
            self._contrast_upper_edit.setText(_fmt(upper))
            return
        value = min(self._contrast_slider.maximum(), max(value, lower))
        self._contrast_slider.setValue((lower, value))
        self._maybe_apply()

    def _on_colormap_changed(self, colormap_name: str) -> None:
        self._update_colorbar(colormap_name)
        self._maybe_apply()

    def _on_colorbar_toggled(self, checked: bool) -> None:
        self._colorbar_bar.setVisible(checked)
        if checked:
            self._update_colorbar(self._colormap_combo.currentText())

    def _update_colorbar(self, colormap_name: str) -> None:
        if not colormap_name or not self._colorbar_checkbox.isChecked():
            return
        lower, upper = self._contrast_slider.value()
        self._colorbar_bar.set_colormap(colormap_name, lower, upper)

    def _maybe_apply(self) -> None:
        if self._live_checkbox.isChecked():
            self._apply()

    def _apply(self) -> None:
        if self._current_layer is None:
            show_warning("No Labels layer selected.")
            return

        df = self._get_df()
        if df is None or df.empty:
            show_warning("No feature data available.")
            return

        feature = self._feature_combo.currentText()
        label_col = self._label_col_combo.currentText()
        colormap_name = self._colormap_combo.currentText()
        lower, upper = self._contrast_slider.value()

        if not feature or feature not in df.columns:
            show_warning(f"Feature '{feature}' not found in data.")
            return

        if df[label_col].astype(int).nunique() != len(df):
            show_warning(
                f"Label column '{label_col}' contains non-unique values. "
                "Please select a column where each row has a unique identifier."
            )
            return

        color_dict, label_properties = compute_colormap(
            df, feature, label_col, colormap_name, lower, upper
        )
        self._current_layer.colormap = DirectLabelColormap(color_dict=color_dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_df(self) -> pd.DataFrame | None:
        if self._csv_radio.isChecked():
            if self._csv_path is None:
                return None
            try:
                return get_df(self._csv_path)
            except OSError:
                return None
        if self._current_layer is None:
            return None
        df = pd.DataFrame(self._current_layer.features)
        return df if not df.empty else None

    def _refresh_columns(self) -> None:
        df = self._get_df()
        if df is None:
            self._clear_columns()
            return

        columns = list(df.columns)
        prev_feature = self._feature_combo.currentText()
        prev_label_col = self._label_col_combo.currentText()

        self._feature_combo.blockSignals(True)
        self._label_col_combo.blockSignals(True)

        self._feature_combo.clear()
        self._feature_combo.addItems(columns)
        self._label_col_combo.clear()
        self._label_col_combo.addItems(columns)

        self._feature_combo.setCurrentText(
            prev_feature if prev_feature in columns else columns[0]
        )
        label_default = (
            prev_label_col
            if prev_label_col in columns
            else check_default_label_column(df)
        )
        if label_default:
            self._label_col_combo.setCurrentText(label_default)

        self._feature_combo.blockSignals(False)
        self._label_col_combo.blockSignals(False)

        # Trigger colormap/limits update for current feature
        self._on_feature_changed(self._feature_combo.currentText())

    def _clear_columns(self) -> None:
        self._feature_combo.clear()
        self._label_col_combo.clear()
        self._colormap_combo.clear()
        self._limits_widget.hide()
