import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from CNN.smallervggnet import SmallerVGGNet
from helper import helper
import time

imageSet = []
controllerSet = []
leftSet = []
rightSet = []
leftBoostSet = []
rightBoostSet = []
boostSet = []
noneActionSet = []
IMAGE_DIMS = [15, 15, 1]
module_path = os.path.dirname(__file__)


def ParseTxt():  # numpy way
	module_path = os.path.dirname(__file__)
	dataFileName = module_path + "/testOct0914.txt"
	a = np.loadtxt(dataFileName)
	print("Loading Data from database.")
	a = a.reshape((len(a), IMAGE_DIMS[0], IMAGE_DIMS[1]))
	helper.showImage(a[300])


def ParseFullData(FileName):  # tradition way
	global imageSet, controllerSet
	dataFileName = module_path + FileName
	with open(dataFileName, "r") as f:
		x = 0
		for line in f.readlines():
			line = line.strip()
			if x % 2 == 0:
				imageSet.append(list(map(int, line.split(" "))))
			else:
				controllerSet.append(list(map(int, line.split(" "))))
			x += 1
	# for i in range(len(imageSet)):     	# find the Wrong Data
	# 	if len(imageSet[i]) != 225:
	# 		print(i,"   ",len(imageSet[i]))
	# print(len(imageSet))
	optimiseData()
	KerasPreprocessing()


def KerasPreprocessing():
	data = np.array(imageSet, dtype="float") / 255.0
	data = data.reshape(len(data), 15, 15)
	labels = classifyController()
	labels = np.array(labels)
	print("Data.shape ",data.shape, "________ Label shape: ", labels.shape)
	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)
	(train_images, test_images, train_labels, test_labels) = train_test_split(data, labels, test_size=0.2, random_state=42)
	print("train_images: ",len(train_images), "test_images :",len(test_images),"train_labels :", len(train_labels), "test_labels :",len(test_labels))
	train_images = np.expand_dims(train_images, axis=3)
	test_images = np.expand_dims(test_images, axis=3)
	model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2])
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(train_images, train_labels, validation_split=0.2, epochs=50, batch_size=20)
	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print("test_acc :", test_acc)
	showModel(history)


def showModel(history):
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(module_path + "/MatplotImages/ModelAccuracy" + helper.getNameByTime() + ".png")
	plt.show()
	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(module_path + "/MatplotImages/ModelLoss" + helper.getNameByTime() + ".png")
	plt.show()



def classifyController():
	labels = []
	for controller in controllerSet:
		if controller[1] == 1:
			if controller[6] == 1:
				labels.append("LB")
			elif controller[7] == 1:
				labels.append("RB")
			else:
				labels.append("B")
		elif controller[6] == 1:
			labels.append("L")
		elif controller[7] == 1:
			labels.append("R")
		else:
			labels.append("")
	return labels


# remove the zero
def optimiseData():
	global imageSet, controllerSet
	fast = 0
	slow = 0
	for fast in range(len(imageSet)):
		if (not isNullAction(controllerSet[fast])):
			imageSet[slow] = imageSet[fast]
			controllerSet[slow] = controllerSet[fast]
			slow += 1
			fast += 1
		else:
			fast += 1
	imageSet = imageSet[0: slow]
	controllerSet = controllerSet[0: slow]
	for (x, y) in zip(imageSet, controllerSet):
		classify(x, y)
	# print(optimisedLabel)
	print("noneActionSet", len(noneActionSet))
	print("leftSet", len(leftSet))
	print("leftBoostSet", len(leftBoostSet))
	print("rightSet", len(rightSet))
	print("rightBoostSet", len(rightBoostSet))
	print("boostSet", len(boostSet))


def isNullAction(list):
	for i in list:
		if i != 0:
			return False
	return True


def classify(image, controller):  # Deprecated only for validate Data
	global leftSet, rightSet, leftBoostSet, rightBoostSet, boostSet
	if controller[1] == 1:  # boost
		if controller[6] == 1:
			leftBoostSet.append(image)
		elif controller[7] == 1:
			rightBoostSet.append(image)
		else:
			boostSet.append(image)
	elif controller[6] == 1:
		leftSet.append(image)
	elif controller[7] == 1:
		rightSet.append(image)
	else:
		noneActionSet.append(image)


def dataSetPreprocessing():  # Deprecated
	global imageSet, controllerSet, leftSet, rightSet, leftBoostSet, rightBoostSet, boostSet
	for (x, y) in zip(imageSet, controllerSet):
		classify(x, y)
	boostSet = np.array(boostSet)
	boostSet = boostSet.reshape(len(boostSet), 15, 15)
	leftSet = np.array(leftSet)
	leftSet = leftSet.reshape(len(leftSet), 15, 15)
	rightSet = np.array(rightSet)
	rightSet = rightSet.reshape(len(rightSet), 15, 15)
	leftBoostSet = np.array(leftBoostSet)
	leftBoostSet = leftBoostSet.reshape(len(leftBoostSet), 15, 15)
	rightBoostSet = np.array(rightBoostSet)
	rightBoostSet = rightBoostSet.reshape(len(rightBoostSet), 15, 15)
	print("leftSet", len(leftSet))
	print("rightSet", len(rightSet))
	print("leftBoostSet", len(leftBoostSet))
	print("rightBoostSet", len(rightBoostSet))
	print("boostSet", len(boostSet))
	print(leftBoostSet[0])


if __name__ == "__main__":
	# ParseTxt()
	ParseFullData("/FullData2.txt")
