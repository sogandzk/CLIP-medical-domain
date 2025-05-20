
import pickle
import matplotlib.cm as cm
from PIL import Image, ImageDraw
import cv2
import numpy as np




with open('../out/00007557_026-rot-0-crp-0.pkl', 'rb') as f:
    res = pickle.load(f)
    
heatmap = res['vmap_array'][2]

# Convert heatmap to RGBA using a colormap (e.g., 'jet', 'plasma', 'coolwarm')
colormap = cm.get_cmap('jet')
heatmap_rgba = colormap(heatmap)  # shape (H, W, 4), float32 in [0,1]
heatmap_rgba[..., 3] = 0.5  # set alpha channel to 0.5 for transparency

# Convert to uint8 and then to PIL
heatmap_uint8 = (heatmap_rgba * 255).astype(np.uint8)

# Convert base image to RGBA
heatmap_pil = Image.fromarray(heatmap_uint8, mode="RGBA")

# Overlay on the original image
overlay = res['image'].resize(heatmap.shape, resample=Image.LANCZOS).convert("RGBA")
composite = Image.alpha_composite(overlay, heatmap_pil)

# Convert mask to format for contour detection (0–255)
mask_for_cv = (res['bbox_mask'] * 255).astype(np.uint8)

# Find contours using OpenCV
contours, _ = cv2.findContours(mask_for_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert composite image to draw-ready format
composite_draw = composite.copy()
draw = ImageDraw.Draw(composite_draw)

# Draw contours
for cnt in contours:
    points = [(int(p[0][0]), int(p[0][1])) for p in cnt]
    draw.line(points + [points[0]], fill=(255, 0, 0, 255), width=1)

print(type(composite_draw))
composite_draw.save("../out/result_overlay.png")