from keras.applications import ResNet50
import ResNet50Modifications as ResNetMods
import CustomImageGen
from CustomImageGen import image_generator as img_gen
from keras.optimizers import Adam
import math
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


#####################################################################
# Load in Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("ResNet50 model loaded...")
#for layer in model.layers[:143]: #175 is the final Activation layer: Activation_49, #143 is another one too.
#    layer.trainable = False
#    print(layer.name

model = ResNetMods.change_activation_function(model)
model = ResNetMods.additional_final_layers(model)
#model = ResNetMods.feedback_loop(model)
model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-10), loss=CustomImageGen.geo_loss, metrics=[CustomImageGen.xyz_error])
model.summary()

#####################################################################
# Find Dataset
x_train_files = CustomImageGen.list_of_files("NUbots","Full\\24Locations-L23")
x_test_files = CustomImageGen.list_of_files("NUbots","Full\\1Location-L23")

#####################################################################
# Train
batch_size = 32
train_SPE = int(math.floor(len(x_train_files)/batch_size))
test_SPE = int(math.floor(len(x_test_files)/batch_size))

file1 = open(".\\Results\\Results.txt", "w")
mycallback = CustomImageGen.MyMetrics(CustomImageGen.image_generator(x_test_files, batch_size, feedback_loop=False, is_random=False, steps=test_SPE), test_SPE, batch_size)
results_train = model.fit(x=CustomImageGen.image_generator(x_train_files, batch_size, False, True, train_SPE), steps_per_epoch=train_SPE, epochs=1, verbose=2, validation_data=CustomImageGen.image_generator(x_test_files, batch_size, False, False, test_SPE), validation_steps=test_SPE, validation_freq=1, callbacks=[mycallback])
print(results_train.history)
print("mycallback:")
print(mycallback.get_median())
epoch_counter = 0
file1.write("%s,%s,%s\n" % (epoch_counter, results_train.history["xyz_error"][0], mycallback.get_median()))
file1.close()
val_freq = 3

for i in range(26):
    print("train: Round " + str(i))
    results_train = model.fit(x=CustomImageGen.image_generator(x_train_files, batch_size, False, True, train_SPE), steps_per_epoch=train_SPE, epochs=val_freq, verbose=2, validation_data=CustomImageGen.image_generator(x_test_files, batch_size, False, False, test_SPE), validation_steps=test_SPE, validation_freq=1, callbacks=[mycallback])
    print("test:")
    print(mycallback.get_median())
    #print(mycallback.get_outliers())
    if i == 23:
        model.save_weights('my_weights_Res5_NU23')
    if i == 24:
        model.save_weights('my_weights_Res5_NU24')
    if i == 25:
        model.save_weights('my_weights_Res5_NU25')
    epoch_counter += val_freq
    file1 = open(".\\Results\\Results.txt", "a")
    file1.write("%s,%s,%s\n" % (epoch_counter, results_train.history["xyz_error"][0], mycallback.get_median()))
    file1.close()



