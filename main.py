from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model.model import VtNet
import numpy as np
import matplotlib.pyplot as plt

tf.test.is_gpu_available()


# Variaveis
epochs = 100
initLr = 1e-4
batchSize = 32

dirTrain = 'input/train'
dirValid = 'input/valid'

print("[INFO] loading images...")

trainDatagen = ImageDataGenerator(rescale=1. / 255)

validDatagen = ImageDataGenerator(rescale=1. / 255)

trainGenerator = trainDatagen.flow_from_directory(
    directory=dirTrain,
    target_size=(28, 28),
    class_mode='categorical',
    batch_size=32
)

validGenerator = validDatagen.flow_from_directory(
    directory=dirValid,
    target_size=(28, 28),
    class_mode='categorical',
    batch_size=32
)

print("[INFO] compiling model...")

model = VtNet.build(width=28, height=28, depth=3, classes=2)

callback = tf.keras.callbacks.ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_accuracy',
                                              verbose=1, mode='auto')

opt = Adam(lr=initLr, decay=initLr / epochs)

model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

model.summary()

train_size = trainGenerator.n
valid_size = trainGenerator.n

h = model.fit_generator(trainGenerator, epochs=epochs, validation_data=validGenerator, callbacks=[callback])

# Analise treinamento

plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), h.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), h.history["accuracy"], label="train_acc")

plt.title("Training Loss and Accuracy on Day/Night")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('chart.png')
