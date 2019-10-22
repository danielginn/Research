import yaml
import os
import scipy
from scipy.spatial.transform import Rotation as R
import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

path = 'N:\\NUpbr\\meta\\'
#path = 'D:\\VLocNet++\\Research\\yaml\\'

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
    file1.write("%s\n" % (f))

file1.close()

file1 = open("ListOfFiles.txt", "r")
file2 = open("NoBallImages.txt", "w+")
count = 0
rad_count = 0
rect_count = 0
for line in file1:
    line = line[:-1]
    with open(line,'r') as stream:
        data_loaded = yaml.safe_load(stream)
        ball_pos = data_loaded.get('ball').get('position')
        robot_pos = data_loaded.get('camera').get('left').get('position')
        robot_rot = data_loaded.get('camera').get('left').get('rotation')
        ball_pos[2] = 0
        robot_pos[2] = 0
        Rbf_f = R.from_euler('z',angle_between(robot_pos, ball_pos),degrees=False)  #Rotation from the field origin to the ball in field space
        Rcf_f = R.from_dcm(robot_rot)

        something = Rbf_f.as_euler()





        count += 1
        print("B%s" % count)
        if lens_type == 'PERSPECTIVE':
            file2.write("%s\n" % (line))
            rect_count += 1
        else:
            file3.write("%s\n" % (line))
            rad_count += 1

print("# of Perspective images:",rect_count)
print("# of Radial images:",rad_count)


