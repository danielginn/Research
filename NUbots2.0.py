from keras.applications import ResNet50
import ResNet50Modifications as ResNetMods
import CustomImageGen
from keras.optimizers import Adam
import math
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


#####################################################################
# Load in Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("ResNet50 model loaded...")
#for layer in model.layers[:143]: #175 is the final Activation layer: Activation_49, #143 is another one too.
#    layer.trainable = False
#    print(layer.name)

model = ResNetMods.additional_final_layers(model)

model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-10), loss='mean_squared_error', metrics=[CustomImageGen.q_error])
#global_pose_network.summary()

#####################################################################
# Find Dataset
x_train_files = CustomImageGen.list_of_files("train")
x_test_files = CustomImageGen.list_of_files("test")

#####################################################################
# Train
batch_size = 32
train_SPE = int(math.ceil(len(x_train_files)/batch_size))
test_SPE = int(math.ceil(len(x_test_files)/batch_size))

file1 = open(".\\Results\\Results.txt", "w")
results_train = model.fit_generator(generator=CustomImageGen.image_generator(x_train_files, batch_size), steps_per_epoch=train_SPE, epochs=1, verbose=2)
results_test = model.evaluate_generator(generator=CustomImageGen.image_generator(x_test_files, batch_size), steps=test_SPE)
epoch_counter = 0
file1.write("%s,%s,%s\n" % (epoch_counter, results_train.history['q_error'], results_test[1]))
file1.close()
val_freq = 3
for i in range(50):

    print("train: Round " + str(i))
    results_train = model.fit_generator(generator=CustomImageGen.image_generator(x_train_files, batch_size), steps_per_epoch=train_SPE, epochs=val_freq, verbose=2)
    print("test:")
    results_test = model.evaluate_generator(generator=CustomImageGen.image_generator(x_test_files, batch_size), steps=test_SPE)
    epoch_counter += val_freq
    file1 = open(".\\Results\\Results.txt", "a")
    file1.write("%s,%s,%s\n" % (epoch_counter, results_train.history['q_error'][val_freq-1], results_test[1]))
    file1.close()
    print(results_test)



