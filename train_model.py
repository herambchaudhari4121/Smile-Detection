#import libraries
#to run this program run following command in command line
#python train_model.py --dataset datasets/SMILEsmileD 
#--model output/lenet.hdf5
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from nn.conv import lenet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

#argument parser for command line argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of faces")
#--dataset is the path to the SMILES directory residing on disk
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
#--model is the path to where the serialized LeNet weights will be 
#saved after training
args = vars(ap.parse_args())

#initialize the list of data and labels
data = []
labels = []

#loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
	#load the image, convert it to grayscale, preprocess it, and store
	#it to data list
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width=28)
	image = img_to_array(image)
	data.append(image)

	#extract the class label from the image path and update the
	#labels list
	label = imagePath.split(os.path.sep)[-3]
	label = "Smiling" if label == "positives" else "Not Smiling"
	labels.append(label)

#scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

#convert the labels from integers to vectors
le = LabelEncoder()
labels = np_utils.to_categorical(le.fit_transform(labels), 2)

#account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

#partition the data into training and testing such that 80% of the data 
#is training data and 20% of data is testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
	test_size=0.20, stratify=labels, random_state=42)

print("[Info] compiling network...")
model = lenet.LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", 
	metrics=["accuracy"])

print("[Info] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), 
	class_weight=classWeight, batch_size=64, epochs=15, verbose=1)

print("[Info] evaluating network...")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis=1), 
	predictions.argmax(axis=1), target_names=le.classes_))

print("[Info] serializing network...")
model.save(args["model"])

#plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()