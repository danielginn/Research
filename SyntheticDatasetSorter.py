import yaml
import os



#path = 'N:\\NUpbr\\meta\\'
path = 'D:\\VLocNet++\\Research\\yaml\\'

files = []
file1 = open("ListOfFiles.txt", "w+")

# r=root, d=directories, f = files
count = 0
for r, d, f in os.walk(path):
    for file in f:
        if '.yaml' in file:
            files.append(os.path.join(r, file))
            count += 1
            print("A%s" % count)

for f in files:
    print(f)
    file1.write("%s\n" % (f))

file1.close()


