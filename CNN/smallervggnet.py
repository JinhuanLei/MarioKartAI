# import the necessary packages
import keras
from keras import layers
from keras import models


class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = models.Sequential()
		# CONV => RELU => POOL
		model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, depth)))
		# strides default (1,1) , kernel_size I think is (3,3)
		model.add(layers.MaxPooling2D((2, 2)))  #
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2, 2)))
		# model.summary()
		# softmax classifier
		model.add(layers.Flatten())
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(50, activation='relu'))
		model.add(layers.Dense(50, activation='relu'))
		model.add(layers.Dense(5, activation='softmax'))  # softmax layer
		# return the constructed network architecture
		return model
