import napari
import numpy as np
import pandas as pd

# 2D example image
shape = (50, 50)
lbl_img_np = np.zeros(shape).astype("uint16")
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
    "index": [1, 2, 3, 4, 5, 6],
    "feature1": [100, 200, 300, 500, 900, 1001],
    "feature2": [2200, 2100, 2000, 1500, 1300, 1001],
}
df = pd.DataFrame(data=d)
# df.to_csv("example_data.csv", index=False)

viewer = napari.Viewer()
label_layer = viewer.add_labels(lbl_img_np)
label_layer.features = df

viewer.show(block=True)
