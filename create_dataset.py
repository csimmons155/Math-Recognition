
import os
import fileinput

BASE_PATH = "/home/csimmons155/Handwritten_NN2 /train_data2/"

def makeOutput():
    file_1 = "output.txt"
    text_file = open(file_1,"w")


    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        count = 0;
        for filename in os.listdir(dirname):
            abs_path = "%s%s" % (dirname,filename)
            try:
                text_file.write("%s;%d\n" % (abs_path, int(filename[0])))
            except ValueError:
                text_file.write("%s;%s\n" % (abs_path, (filename[0])))




makeOutput()



