from imutils import paths
import random
import os


data = []
labels = []

imagePaths = sorted(list(paths.list_images('input')))

random.shuffle(imagePaths)

i = 0

for images in imagePaths:
    strtmp = images.split(os.path.sep)[-1]
    os.rename(images, 'valid/nvisiveis/'+strtmp)
    i += 1

    if i == 1350:
        break