import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set directory
dir_train = 'input/'


dataGen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

i = 0

for batch in dataGen.flow_from_directory(directory=dir_train,
                                         batch_size=1,
                                         target_size=(480, 752),
                                         save_to_dir='output/',
                                         save_prefix='aug',
                                         save_format='jpeg'):
    i += 1
    if i > 2500:
        break