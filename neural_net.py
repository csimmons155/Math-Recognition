from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer

from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.shortcuts import buildNetwork


class neural_net:
    #the length of the arguments must be at least 2 (input_size, # of hidden layers, output size)
    def __init__(self, *args):
        self.nnet = buildNetwork(*args, bias=True)
        #NetworkWriter.writeToFile(self.nnet, 'filename2.xml')
        #self.nnet = NetworkReader.readFrom('filename2.xml')
        self.insize = args[0]
        self.outsize = args[len(args) - 1]
        #to activate the neural netowork -- internally initializes the net to be used
        # we have input, hidden, and output modules
        self.nnet.sortModules()

    def loadData(self, trainData):
        if trainData.getDimension('input') == self.insize and trainData.getDimension('target') == self.outsize:
            #BackpropTrainer -- nnet = the module to be trained, trainData = the Dataset from output.txt
            self.trainer = BackpropTrainer(self.nnet, trainData,verbose=True)
            return 1
        else:
            print "Dataset Size error"
            return 0

    #train data w/ n iterations
    def trainData(self, n):
        #self.trainer.trainUntilConvergence()

        for i in range(1,n+1):
            #trains the data from BackPropTrainer above
            self.trainer.train()
            print "Train: " + str(i)


    #propagate some data though the neural network
    def activate(self, testData):

        if testData.size == self.insize:
            #the output of the testData propagated thru neural network
            prop = self.nnet.activate(testData)
            return int(round(prop[0],0))
        else:
            print "Size error for activate()"
            return 0


