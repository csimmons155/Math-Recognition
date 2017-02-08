
import Image as im
import numpy as np
import cv2


#image = im.open("P0.bmp")
#image = im.open("09.bmp")
#imarray = np.asarray(image.crop(image.getbbox()).resize((10, 10))).astype(float)
image = cv2.imread("P0.bmp",1)
image = cv2.resize(image,(20,28))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray,(15,15)).astype(float)
imarray = np.asarray(gray)
print imarray
print ""
#imarray = imarray[(imarray != 255)]
#print imarray

nonzero = np.transpose(imarray.nonzero())



print "\n"
print nonzero
cv2.imwrite("test.bmp",gray)


"""
#image = cv2.imread("P0.bmp",1)
cap = cv2.VideoCapture(0)
ret, image = cap.read()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray,(20,20)).astype(float)
cv2.imwrite("test.bmp",gray)

"""

"""
#cv2.imwrite("test.bmp",gray)
cap.release()
cv2.destroyAllWindows()
"""