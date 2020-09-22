from tensorflow.keras.models import load_model
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# carregar imagem
image = cv2.imread('/home/willian/vizentec/ocr/537859.jpg')
orig = image.copy()

# pre-processamento da imagem a ser passada
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# carregar modelo
print("[INFO] loading network...")
model = load_model('model.hdf5')

# classify the input image
(night, day) = model.predict(image)[0]

label = "visible" if day > night else "n_visible"
proba = day if day > night else night
label = "{}: {:.2f}%".format(label, proba * 100)

# desenhar label na imagem
output = imutils.resize(orig, width=800)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("output", output)
cv2.imwrite("output.jpg", output)
cv2.waitKey(0)
