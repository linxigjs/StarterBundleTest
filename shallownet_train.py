from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pyimagesearch.nn.conv.shallownet import ShallowNet
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images("datasets/animals"))

sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0


trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

print("[INFO] serializing network...")
model.save(args['model'])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog', 'panda']))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0,100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0,100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0,100), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()