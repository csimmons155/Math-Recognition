from pybrain.datasets import SupervisedDataSet
import fileinput
import Image as im
import numpy as np
import cv2

class getInput:

    #methods not bound to merely an instance or class, but to either one
    @staticmethod
    def getImageArray(path, num):
        if num == 1:
            #TRAINING -- num = 1
            try:
                # each training image
                # image = im.open(path)
                # getbox -- bounding box of nonzero region in Image, resize to 10x10 image w/ float pix values
                # imarray = np.asarray(image.crop(image.getbbox()).resize((20,20))).astype(float)
                path = cv2.imread(path, 1)
                path = cv2.resize(path, (20,28))
                path = cv2.resize(path, (15,15)).astype(float)
                imarray = np.asarray(path)
                return imarray
            except IOError:
                print "Input file not found"
                return
        else:
            #LIVE INPUT -- num = 2
            path = cv2.resize(path, (20, 28))
            path = cv2.resize(path,(15, 15)).astype(float)
            imarray = np.asarray(path)
            return imarray




    @staticmethod
    #Normalize
    def normImage(inarray,x,y):
        #original range = (1,5)
        for a in range(1,4):
            for i in range(x-a,x+a+1):
                for j in range(y-a, y+a+1):
                    if i <= 14 and i >= 0 and j <= 14 and j >= 0:
                        if inarray[i,j].all() < inarray[x,y].all() and inarray[i,j].all() < (inarray[x,y]-0.2*a).all():
                            inarray[i,j] = inarray[x,y]-0.2*a

    #uses the above method to normalize input image array
    @staticmethod
    def getNormImage(imarray):
        #.nonzero() -- returns the indices of the nonzero elements in the 2D image array
        #nonzero = n x 2 vertical matrix
        nonzero = np.transpose(imarray.nonzero())
        #the number of nonzero pixels in the image
        num = nonzero.shape[0]
        for i in range(0,num):
            getInput.normImage(imarray,nonzero[i][0],nonzero[i][1])
        return imarray

    @staticmethod
    def getNormFromPath(impath, num):
        #param: the array form of the input image -- get the normalized image
        return getInput.getNormImage(getInput.getImageArray(impath, num))


    @staticmethod
    def getImageVector(impath, num):
        imarray = getInput.getNormFromPath(impath, num)
        vec = np.resize(imarray,(1,225))
        #horizontal 1D array
        #print "Vector   ", vec[0], "\n"
        return vec[0]

#make Dataset for the neural network
class dataSet:

    #get szie of input (number of pixels) and labels (size 1, where an input mataches to 1 label)
    def __init__(self, insize, labelsize):
        self.insize = insize
        self.labelsize = labelsize
        #called to add a single line of data from output.txt
        self.DS = SupervisedDataSet(self.insize,self.labelsize)

    def addData(self, indata, labeldata):
        try:
            if indata.size == self.insize and labeldata.size == self.labelsize:
                #add image files and labels to the dataset
                self.DS.appendLinked(indata,labeldata)
                return 1
        except AttributeError:
            print "Error in input Size"
            return 0

    def getDataSet(self):
        return self.DS

    #the folder where the file:label correspondences are .....

    def makeDataSet(self):
        num = 1;
        for li in fileinput.input("output.txt"):
            x = li.split(';')
            #adding data line by line to complete the whole dataset
            print "Current: ", num, " / 154"
            self.addData(getInput.getImageVector(x[0], 1), np.array([int(x[1])]))
            num+=1
        return 1



