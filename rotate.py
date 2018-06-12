import random
from PIL import Image
import os

path_to_image_files = []
imagePath = 'train/'
f = []
for (dirpath, dirnames, filenames) in os.walk(imagePath):
    f.extend(filenames)
    break
for path in f:
    if path.split('.')[-1] == 'png' or path.split('.')[-1] == 'jpg':
        path_to_image_files.append(imagePath + path)

for filename in path_to_image_files:
    img = Image.open(path_to_image_files[0])
    img = img.rotate(20 * (random.random()-0.5))
    img.save(imagePath + str(int(1000*random.random())) + '_' + filename.split('_')[-1].split('/')[-1])