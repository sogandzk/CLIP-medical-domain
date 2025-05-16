import os
import pandas as pd


csv_path = "../data/BBox_List_2017 copy.csv"
image_folder = "data/images"



df = pd.read_csv(csv_path)

available_images = set(os.listdir(image_folder))











