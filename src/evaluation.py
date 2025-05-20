import pickle
import matplotlib.pyplot as plt
import torch
import sys
import pickle
import matplotlib.cm as cm
from PIL import Image, ImageDraw
from skimage import measure
import numpy as np


assert (len(sys.argv)-1) == 2
data_path = sys.argv[1]
out_dir_result = sys.argv[2]


with open(data_path, "rb") as f:
    data = pickle.load(f)


vmap_array = data["vmap_array"]
bbox_mask = data["bbox_mask"]
image = data["image"]


def draw_img(heatmap, bbox_mask, image):
    # Convert heatmap to RGBA
    colormap = cm.get_cmap('jet')
    heatmap_rgba = colormap(heatmap)
    heatmap_rgba[..., 3] = 0.5

    heatmap_uint8 = (heatmap_rgba * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_uint8, mode="RGBA")

    overlay = image.resize(heatmap.shape, resample=Image.LANCZOS).convert("RGBA")
    composite = Image.alpha_composite(overlay, heatmap_pil)

    composite_draw = composite.copy()
    draw = ImageDraw.Draw(composite_draw)

    # Find contours using skimage
    contours = measure.find_contours(bbox_mask, level=0.5)

    for contour in contours:
        # contour is a Nx2 array of (row, col), so we convert to (x, y)
        points = [(int(col), int(row)) for row, col in contour]
        if len(points) > 1:
            draw.line(points + [points[0]], fill=(255, 0, 0, 255), width=1)

    return composite_draw


def threshold_heatmap(vmap, threshold=0.5):
    vmap_norm = (vmap - vmap.min()) / (vmap.max() - vmap.min())
    return (vmap_norm > threshold)


def calculate_iou(bbox_mask, heatmap_mask):
    intersection = (bbox_mask * heatmap_mask).sum()
    union = ((bbox_mask + heatmap_mask) > 0).float().sum()
    return (intersection / union).item()



def calculate_precision(bbox_mask, heatmap_mask):
    tp = (heatmap_mask * bbox_mask).sum()
    predicted_positives =heatmap_mask.sum()
    return (tp / predicted_positives).item() if predicted_positives > 0 else 0.0



def calculate_recall(bbox_mask, heatmap_mask):
    tp = (heatmap_mask * bbox_mask).sum()
    actual_positives = bbox_mask.sum()
    return (tp / actual_positives).item() if actual_positives > 0 else 0.0


def calculate_f1(precision, recall):
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0



def calculate_metrics(bbox_mask, heatmap_mask):
    iou=calculate_iou(bbox_mask,heatmap_mask,)
    precision = calculate_precision(bbox_mask,heatmap_mask)
    recall = calculate_recall(bbox_mask,heatmap_mask)
    f1 = calculate_f1(precision, recall)
    return iou, precision, recall,f1


bbox_mask_torch = torch.from_numpy(bbox_mask)

resualt_list = [] 

for vmap in vmap_array:
    vmap = torch.from_numpy(vmap)
    vmap_mask = threshold_heatmap(vmap)
    result = calculate_metrics(bbox_mask_torch,vmap_mask)
    img = draw_img(vmap,bbox_mask,image)
    resualt_dict = {
        "iou": result[0],
        "precision": result[1],
        "recall": result[2],
        "f1": result[3],
        "img": img
    }
    resualt_list.append(resualt_dict)


name_parts = data_path.split("/")[-1].split(".")[0]

output_path = out_dir_result + '/' + name_parts + "-result.pkl"
with open(output_path, "wb") as f:
    pickle.dump(resualt_list, f)










































