from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Variables
epochs = 100
initLr = 1e-3
batchSize = 32
dirTrain = 'input/'

print("[INFO] loading images...")

dataGen = ImageDataGenerator(rescale=1. / 255)

dataTrain = dataGen.flow_from_directory(

)