import cv2
import math
import numpy as np
import json

image = cv2.imread(".\\dataset2\\P1.JPG", cv2.IMREAD_COLOR)
height, width, channels = image.shape

print("height: " + str(height))
print("width: " + str(width))

cx = 2681 #P1
#cx = 2632 #P2

radius = 0.5*width/math.pi
px_020deg = width/1800
#cx = int(width/2) - 1000
cy = int(height/2)


steps_left = math.floor(cx/px_020deg)
x_start = round(cx - steps_left*px_020deg)
print(x_start)

position = (0, 0, 0.8)

centre_crop = np.zeros([672, 672, 3])
x = x_start
deg = -steps_left / 5
while (x < width):
    if (x < (width - 672)):
        centre_crop = image[cy-336:cy+336, x:x+672, :]
    else:
        diff = width-x
        centre_crop[:,0:diff,:] = image[cy-336:cy+336, width-diff:width,:]
        centre_crop[:,diff:672,:] = image[cy-336:cy+336, 0:672-diff, :]

    resized = cv2.resize(centre_crop,(224,224), interpolation=cv2.INTER_LINEAR)
    outfilename = ".\\dataset2\\" + str(x).zfill(4) + ".JPG"
    cv2.imwrite(filename=outfilename, img=resized)
    json_outfilename = ".\\dataset2\\" + str(x).zfill(4) + ".json"
    with open(json_outfilename, 'w') as outfile:
        json.dump({'position':position, 'orientation':deg}, outfile, indent=4)
    steps_left -= 1
    x = round(cx - steps_left*px_020deg)
    deg += 0.2

