print("Importing libs")

import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import clip
from transformers import CLIPProcessor, CLIPTokenizerFast

from scripts.clip_wrapper import ClipWrapper
from scripts.methods import vision_heatmap_iba

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For deterministic behavior (may affect speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1234)


class NIHSingleLabelBBoxDataset(Dataset):
    def __init__(self, csv_path, images_dir):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_fileName = row['Image Index']
        image_path = self.images_dir + '/' + image_fileName
        image_id = os.path.splitext(image_fileName)[0]
        label = row['Finding Label']
        bbox = [row['Bbox [x'], row['y'], row['w'], row['h]']]
        return image_id, image_path, label, bbox
    

def get_map(model, processor, tokenizer, device, image, text, vbeta, vvar, vlayer):
    # Preprocess image
    
    image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(device) # 3*224*224
    # Tokenize text
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
    # Train information bottleneck on image
    print("Training M2IB on the image...")
    vmap = vision_heatmap_iba(text_ids, image_feat, model, vlayer, vbeta, vvar)
    return vmap



def bbox_to_mask(bbox, frame_size):
    mask = np.zeros(frame_size, dtype=np.uint8)
    x1, x2 = int(bbox[0]), int(bbox[0] + bbox[2])
    y1, y2 = int(bbox[1]), int(bbox[1] + bbox[3])
    mask[y1:y2, x1:x2] = 1
    return mask

def rotate_image(image, degree):
    rotated_image = image.rotate(degree, expand=True)
    return rotated_image

def crop_image(image, degree):
    return 0


print("len(sys.argv)",len(sys.argv))
assert (len(sys.argv)-1) == 5
data_dir = sys.argv[1]
csv_index = sys.argv[2]
output_dir = sys.argv[3]
rotation_degree = sys.argv[4]
crop = sys.arg[5]


dataset = NIHSingleLabelBBoxDataset(
    csv_path = data_dir + '/BBox_List_2017.csv',
    images_dir = data_dir + '/images'
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Processing")

model, preprocess = clip.load("ViT-B/32", device=device, download_root=data_dir)
model.load_state_dict(torch.load(data_dir+"/clip-imp-pretrained_128_6_after_4.pt", map_location=device))
model = ClipWrapper(model)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=data_dir)
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", cache_dir=data_dir)


image_id, image_path, label, bbox = dataset[int(csv_index)]
text = 'findings include: 1. ' + label
image = Image.open(image_path).convert('RGB')

bbox_mask = bbox_to_mask(bbox, image.size)
bbox_mask_image = Image.fromarray(bbox_mask, mode='L')

if int(rotation_degree) != 0:
    image = rotate_image(image, rotation_degree)
    bbox_mask_image = rotate_image(bbox_mask_image, rotation_degree)
    if int(crop) == 1:
        image = crop_image(image,rotation_degree)
        bbox_mask_image = crop_image(bbox_mask_image,rotation_degree)




# output = []
# for rep in range(2):
#     print("repeat "+str(rep))
#     vmap = get_map(model=model, 
#                    processor=processor, 
#                    tokenizer=tokenizer, 
#                    device=device, 
#                    image=image, 
#                    text=text, 
#                    vbeta=1, 
#                    vvar=1, 
#                    vlayer=9)
#     output.append(vmap)
# output = np.stack(output)

# print("save output")
# output_path = output_dir + '/' + image_id + '_org_attrMap.pkl'
# with open(output_path, "wb") as f:
#     pickle.dump(output, f)
        


bbox_mask_image = bbox_mask_image.resize((224, 224), resample=Image.BICUBIC)


print("save output")
output_path = output_dir + '/' + image_id + rotation_degree + ' crop:' + crop + 'bboxmask.pkl'
with open(output_path, "wb") as f:
    pickle.dump(image, f)


output_path = output_dir + '/' + image_id + rotation_degree + ' crop:' + crop + 'attrMap.pkl'
with open(output_path, "wb") as f:
    pickle.dump(bbox_mask_image, f)

