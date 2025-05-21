import pickle
import sys
import numpy as np

assert (len(sys.argv)-1) == 2
data_path = sys.argv[1]
out_dir = sys.argv[2]

with open(data_path, "rb") as f:
    results = pickle.load(f)

ious = [d['iou'] for d in results]
precisions = [d['precision'] for d in results]
recalls = [d['recall'] for d in results]
f1s = [d['f1'] for d in results]

metrics_summary = {
    'iou': (float(np.mean(ious)), float(np.std(ious))),
    'precision': (float(np.mean(precisions)), float(np.std(precisions))),
    'recall': (float(np.mean(recalls)), float(np.std(recalls))),
    'f1': (float(np.mean(f1s)), float(np.std(f1s))),
}

print("metrics_summary",metrics_summary)

name_parts = data_path.split("/")[-1].split(".")[0]
output_path = out_dir + '/' + name_parts + "-metrics_summary.pkl"
with open(output_path, "wb") as f:
    pickle.dump(metrics_summary, f)













    











