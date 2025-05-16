import pickle
import matplotlib.pyplot as plt
import torch
import sys


assert (len(sys.argv)-1) == 2
data_path = sys.argv[1]
out_dir_result = sys.argv[2]


with open(data_path, "rb") as f:
    data = pickle.load(f)


vmap_array = data["vmap_array"]
bbox_mask = data["bbox_mask"]
image = data["image"]


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


bbox_mask = torch.from_numpy(bbox_mask)

resualt_list = [] 

for vmap in vmap_array:
    vmap = torch.from_numpy(vmap)
    vmap_mask = threshold_heatmap(vmap)
    result = calculate_metrics(bbox_mask,vmap_mask)
    resualt_dict = {
        "iou": result[0],
        "precision": result[1],
        "recall": result[2],
        "f1": result[3]
    }
    resualt_list.append(resualt_dict)



name_parts = data_path.split("/")[1].split(".")[0]
name_parts_without_extension = name_parts


output_path = out_dir_result + '/' + name_parts_without_extension + "-result.pkl"
with open(output_path, "wb") as f:
    pickle.dump(resualt_list, f)










































