import matplotlib.pyplot as plt
import time


class helper:
	@staticmethod
	def showImage(imageData):
		fig = plt.figure()
		plotwindow = fig.add_subplot(111)
		plt.axis('off')
		plt.imshow(imageData, interpolation='nearest', cmap='gray')
		plt.show()

	@staticmethod
	def getNameByTime():
		localTime = time.strftime("%m%d%H%M", time.localtime())
		return str(localTime)
