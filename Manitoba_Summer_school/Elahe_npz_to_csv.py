import numpy as np
from pathlib import Path
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt

seg_dir = Path("HumanBlood_28Frames_tifANDnpz") 
seg_files = sorted(seg_dir.glob("*_seg.npz"))

all_cells = []

for seg_file in seg_files:
    data = np.load(seg_file, allow_pickle=True)
    masks = data["masks"]
    base_name = seg_file.stem
    frame = int(base_name.split("_")[0])  #filename are like 0000_seg.npz


    #get centroid and label
    props = measure.regionprops(masks)

    for prop in props:
        cell_id = prop.label
        cy, cx = prop.centroid
        area = prop.area
        bbox = prop.bbox 
        

        #boundary is a contour
        binary_mask = (masks == cell_id).astype(np.uint8)
        contours = measure.find_contours(binary_mask, level=0.5)

        if contours:
            boundary = max(contours, key=lambda x: len(x))
            #boundary += np.array([prop.bbox[0], prop.bbox[1]])

            all_cells.append({
                "file": base_name,
                "frame": frame,
                "cell_id": cell_id,
                "x_center": cx,
                "y_center": cy,
                "area": area,
                "bbox": list(bbox),  # tuple to list for saving
                "boundary": boundary.tolist(),
                "track_id": -1
            })

if all_cells:  #check it is not empty
    df = pd.DataFrame(all_cells)
    

    df.to_csv("HumanBlood_28Frames_tifANDnpz/humanBlood_dataset.csv", index=False)
else:
    print("No cells detected. CSV not saved.")

