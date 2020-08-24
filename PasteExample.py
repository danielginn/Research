from PIL import Image
from keras.preprocessing.image import img_to_array
import CustomImageGen
import numpy as np
import math
import random
#Read the two images
image = Image.open('NUbotsField/train/03619.jpg')
#image1.show()



#resize, first image
image_size = image.size






scale = 0.5
height = 224 / scale
img_resized = image.resize((int(round(height * 1.25)), int(round(height))), Image.ANTIALIAS)
#img_resized.show()
img_np = img_to_array(img_resized)
cropped_image = CustomImageGen.crop_generator(img_np, 224, isRandom=False)

obscure_pcnt = 0.15
long_dim = round(math.sqrt(50176 * obscure_pcnt) * 1.414)
short_dim = round(long_dim / 2)
rand_hor = random.randint(0, 224 - short_dim)
rand_ver = 112-short_dim
for m in range(rand_ver, rand_ver + long_dim):
    for n in range(rand_hor, rand_hor + short_dim):
        for c in range(3):
            cropped_image[m, n, c] = 0

result = Image.fromarray(np.uint8(cropped_image)).convert('RGB')
result.show()
#input("Press Enter to continue...")
#result.save("Person/example4.jpg",format='JPEG')
