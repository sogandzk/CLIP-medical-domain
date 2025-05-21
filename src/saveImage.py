import os
import sys
import pickle
from PIL import Image
import matplotlib.pyplot as plt


assert (len(sys.argv)-1) == 2
data_path = sys.argv[1]
out_dir_result = sys.argv[2]


with open(data_path, "rb") as f:
    data = pickle.load(f)

for i in range(len(data)):
    image = data[i]['img']
    name_parts = data_path.split("/")[-1].split(".")[0]
    output_path = out_dir_result + '/' + name_parts +str(i)+"-img.png"
    image.save(output_path)





