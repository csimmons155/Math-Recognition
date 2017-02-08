import neural_net as nn
import training as tt
import time
import cv2
from pybrain.tools.customxml.networkreader import NetworkReader

#neural network w/ 100 pixel as input, 80 hidden neurons, and 1 output

#hidden was: 80
# prev. 400 in size


n = nn.neural_net(225,90,1)
print "Neural Network initialized.."

#"""
d = tt.dataSet(225,1)
print "Dataset initialized.."

t0 = time.clock()

if (d.makeDataSet()):
    print "Training Dataset Created"

print "Dataset Creation Time (min): ", (time.clock() - t0)/60

if n.loadData(d.getDataSet()):
    print "Dataset loaded into Neural Network"



#n = NetworkReader.readFrom('filename.xml')

print "Training Data...."
n.trainData(300)

#"""

#Labels:
# + : 10
# * : 11
# \ : 12
# - : 13



filename = "/home/csimmons155/Handwritten_NN2 /train_data2/08.bmp"
x = n.activate(tt.getInput.getImageVector(filename, 1))
print "Number 0 is read as: " + str(x)

filename = "/home/csimmons155/Handwritten_NN2 /train_data2/18.bmp"
x = n.activate(tt.getInput.getImageVector(filename, 1))
print "Number 1 is read as: " + str(x)

filename = "/home/csimmons155/Handwritten_NN2 /train_data2/28.bmp"
x = n.activate(tt.getInput.getImageVector(filename, 1))
print "Number 2 is read as: " + str(x)

filename = "/home/csimmons155/Handwritten_NN2 /train_data2/38.bmp"
x = n.activate(tt.getInput.getImageVector(filename, 1))
print "Number 3 is read as: " + str(x)

filename = "/home/csimmons155/Handwritten_NN2 /train_data2/43.bmp"
x = n.activate(tt.getInput.getImageVector(filename, 1))
print "Number 4 is read as: " + str(x)

filename = "/home/csimmons155/Handwritten_NN2 /train_data2/57.bmp"
x = n.activate(tt.getInput.getImageVector(filename, 1))
print "Number 5 is read as: " + str(x)


filename = "/home/csimmons155/Handwritten_NN2 /train_data2/62.bmp"
x = n.activate(tt.getInput.getImageVector(filename, 1))
print "Number 6 is read as: " + str(x)

""""
cap = cv2.VideoCapture(1)

contours_length = 0

while(True):
    ret,im = cap.read()

    #convert to grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.bilateralFilter(im_gray,9,70,16)
    im_gray = cv2.medianBlur(im_gray,5)

    ret, thres = cv2.threshold(im_gray,110,255,cv2.THRESH_BINARY)

    #cv2.imshow("thes",thres)

    contours, hierarchy = cv2.findContours(thres.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) != contours_length):

        contours_length = len(contours)
        num = 0

        for i in contours:

            M = cv2.moments(i)

            if(M['m10'] != 0 and M['m00'] != 0 and M['m01'] != 0):

                #area = the total intensity of the image
                area = cv2.contourArea(i)


                # get the centriod pixel (based on intensity value) of each contour
                #m00 = the sum of all intensities
                #m10 = center of mass for intensity in x direction
                #m01 = for y direction

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                #bounding rectange around the contour
                x,y,w,h = cv2.boundingRect(i)



                if(area > 400):

                    try:
                        num = num + 1
                        crop = thres[y-(h/4):y+h+h/3, x-(w/4):x+w+w/3].copy()
                        #crop = cv2.resize(crop, (20,20))
                        cv2.imshow("cropped window", crop)

                        test_crop = cv2.resize(crop,(20,28))
                        cv2.imshow("20x28 crop", test_crop)


                        #cv2.imwrite("test_input.bmp", crop)

                        test = tt.getInput.getImageVector(crop, 2)
                        print "Now, digits....."

                        #input the cropped image into the neural network
                        #digit = n.activate(tt.getInput.getImageVector(crop, 2))
                        digit = n.activate(test)


                        print "Recognized digit -  " + str(digit)

                        #search for digit in the dictionary (int : string)
                        # append the string to a list ......

                        cv2.rectangle(im, (x - (w / 4), y - (h / 4)), (x + w + w / 3, y + h + h / 3), (255, 0, 0), 2)

                        #CHANGE THIS ....
                        cv2.putText(im, str(digit),(x,y-y/4),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)



                    except:
                        pass

        cv2.imshow("image",im)
        cv2.imshow("thres", thres)


    key = cv2.waitKey(1) & 0xFF
    if(key == ord('q')):
        break



cap.release()
cv2.destroyAllWindows()
"""



'''

""""
while(True):
    x = raw_input("Enter a task:\t")
    if x == "q" or x == "Q":
        break
    elif x == "t":
        t = int(raw_input("How many iterations?\t"))
        n.trainData(t)
    elif x == "i":
        i = raw_input("Enter filename:\t")
        x = n.activate(tt.getInput.getImageVector(i))
        print "The Image is: " + str(x) + "\n"
    else:
        break

"""""

im = Image.open("test.bmp")
print im.format, im.size, im.mode

print im.getbbox()


'''
