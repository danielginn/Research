import yaml
import os
import string
import json


#path = 'N:\\NUpbr\\meta\\'
#path = 'D:\\VLocNet++\\Research\\yaml\\'
path = "D:\\VLocNet++\\Research\\NUbotsDatasets\\NUbotsSoccerField1\\{}\\".format("train")

files = []
file1 = open("ListOfFiles.txt", "w+")

# r=root, d=directories, f = files
count = 0
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            print(file)
            files.append(os.path.join(r, file))

            json_filename = file[0:-4] + '.json'
            print(json_filename)
            files.append(os.path.join(r, json_filename))
            count += 1

for f in files:
    file1.write("%s\n" % (f))

file1.close()


