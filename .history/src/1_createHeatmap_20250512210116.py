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
        return image_id, image_path, label
    

def get_map(model, processor, tokenizer, device, image_path, text, vbeta, vvar, vlayer):
    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(device) # 3*224*224
    # Tokenize text
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
    # Train information bottleneck on image
    print("Training M2IB on the image...")
    vmap = vision_heatmap_iba(text_ids, image_feat, model, vlayer, vbeta, vvar)
    return vmap


assert (len(sys.argv)-1) == 3
data_dir = sys.argv[1]
csv_index = sys.argv[2]
output_dir = sys.argv[3]


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


image_id, image_path, label = dataset[int(csv_index)]
text = 'findings include: 1. ' + label

output = []
for rep in range(2):
    print("repeat "+str(rep))
    vmap = get_map(model=model, 
                   processor=processor, 
                   tokenizer=tokenizer, 
                   device=device, 
                   image_path=image_path, 
                   text=text, 
                   vbeta=1, 
                   vvar=1, 
                   vlayer=9)
    output.append(vmap)
output = np.stack(output)

print("save output")
output_path = output_dir + '/' + image_id + '_org_attrMap.pkl'
with open(output_path, "wb") as f:
    pickle.dump(output, f)
