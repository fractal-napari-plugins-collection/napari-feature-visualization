# napari-feature-visualization

[![License BSD-3](https://img.shields.io/pypi/l/napari-feature-visualization.svg?color=green)](https://github.com/fractal-napari-plugins-collection/napari-feature-visualization/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-feature-visualization.svg?color=green)](https://pypi.org/project/napari-feature-visualization)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-feature-visualization.svg?color=green)](https://python.org)
[![tests](https://github.com/fractal-napari-plugins-collection/napari-feature-visualization/workflows/tests/badge.svg)](https://github.com/fractal-napari-plugins-collection/napari-feature-visualization/actions)
[![codecov](https://codecov.io/gh/fractal-napari-plugins-collection/napari-feature-visualization/branch/main/graph/badge.svg)](https://codecov.io/gh/fractal-napari-plugins-collection/napari-feature-visualization)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-feature-visualization)](https://napari-hub.org/plugins/napari-feature-visualization)

Visualize per-label feature measurements on label images in napari. Color each
label by any numeric or categorical feature column, with support for both
continuous colormaps (with adjustable contrast limits) and qualitative
colormaps for categorical data.

![feature_viz_demo](https://github.com/user-attachments/assets/4c7e2d37-2981-43bc-9bb6-1f59ac973dd1)

Feature data can be loaded from a CSV file or read directly from
`layer.features` — enabling visualization of measurements stored in OME-Zarrs
via plugins such as the [napari OME-Zarr navigator].

---

## Installation

```
pip install napari-feature-visualization
```

---

## Usage

Open the widget via **Plugins → Feature Visualization**.

### 1. Select a data source

The widget has two modes, selected with the radio buttons at the top:

- **Layer Properties** — reads feature columns directly from the active Labels
  layer's `.features` DataFrame. No file is needed; the features travel with
  the layer.
- **CSV File** — loads a CSV from disk. Click **Browse…** to pick the file.
  The CSV must have one row per label and at least one column whose values
  match the label IDs in the image.

<img width="1983" height="1243" alt="feature_viz_overview" src="https://github.com/user-attachments/assets/56a41144-9c2c-460e-ad06-11fc5df6b6d8" />

### 2. Select a Labels layer

Click a Labels layer in the napari layer list. The widget automatically tracks
the active layer — the layer name shown below the radio buttons updates as you
switch between layers. If the active layer is not a Labels layer or has no
feature columns the dropdowns will be empty.

### 3. Configure columns

| Dropdown | Purpose |
|---|---|
| **Feature** | The column to visualize (any numeric or string column) |
| **Label column** | The column whose integer values match the label IDs in the image |

The **Label column** defaults to the first column named `label`, `Label`, or
`index` if one exists.

### 4. Choose a colormap

For **continuous** features all napari colormaps are available. The default is
`viridis`.

For **categorical** (non-numeric) features only qualitative colormaps wide
enough to cover the number of unique categories are shown. The default is
`tab10` (up to 10 categories). When the number of categories exceeds all
available qualitative colormaps the napari `label_colormap` is used as
fallback.

<img width="1983" height="1243" alt="feature_viz_continuous" src="https://github.com/user-attachments/assets/babef956-8439-43b8-962a-ca8427f4f555" />

<img width="1983" height="1243" alt="feature_viz_categorical" src="https://github.com/user-attachments/assets/6a6deee3-6149-4217-a57b-d8dcebaabd7e" />


### 5. Adjust contrast limits (continuous features only)

The range slider sets the lower and upper contrast limits. Values below the
lower limit map to the first colormap color; values above the upper limit map
to the last. The limits default to the 1st and 99th percentiles of the feature
values.

The **Show colorbar** checkbox renders a labeled gradient bar below the slider.

<img width="578" height="98" alt="feature_viz_colorbar" src="https://github.com/user-attachments/assets/15fdca4e-b26b-4265-952d-4b50b7510b23" />


### 6. Apply

- Click **Apply Feature Colormap** to color the layer once.
- Check **Live update** to automatically re-apply whenever you change the
  feature, colormap, or contrast limits.

Labels whose feature value is `NaN` / `None` are rendered transparent.
Background (label 0) is always black.

---

## Contributing

Contributions are welcome. The project uses [pixi] for environment management.

```
git clone https://github.com/fractal-napari-plugins-collection/napari-feature-visualization
cd napari-feature-visualization
pixi install
pixi run test      # run the test suite
pixi run napari    # launch napari with the plugin installed
```

Linting and formatting use [ruff]:

```
pixi run lint      # check
pixi run format    # auto-format
pixi run check     # lint + test in one step
```

Please ensure the test coverage does not decrease before opening a pull
request.

---

## Releases

Releases are driven by git tags. The version is managed automatically by
[setuptools-scm] — there is no version number to edit manually.

1. Make sure `main` is clean and all tests pass.
2. Create and push a version tag:
   ```
   git tag v<major>.<minor>.<patch>
   git push origin v<major>.<minor>.<patch>
   ```
3. The GitHub Actions workflow detects the tag and automatically:
   - Builds the source distribution and wheel (`python -m build`)
   - Uploads to PyPI via Twine (requires the `TWINE_API_KEY` secret to be set
     in the repository settings)

There is no separate changelog file — use the GitHub Releases page to document
what changed.

---

## License

Distributed under the terms of the [BSD-3] license.

## Issues

[File an issue] on GitHub with a detailed description.

[napari OME-Zarr navigator]: https://github.com/fractal-napari-plugins-collection/napari-ome-zarr-navigator
[pixi]: https://pixi.sh
[ruff]: https://docs.astral.sh/ruff/
[setuptools-scm]: https://setuptools-scm.readthedocs.io/
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[File an issue]: https://github.com/fractal-napari-plugins-collection/napari-feature-visualization/issues
