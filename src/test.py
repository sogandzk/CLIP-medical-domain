import os

import pickle
from PIL import Image
import matplotlib.pyplot as plt

with open('../outEvaluation/00000032_037-rot-0-crp-0-result.pkl', 'rb') as f:
    data = pickle.load(f)


# print(data[0]['img'])

image = data[0]['img']

image.show()


image.save("../outFinalImage/1.png")
