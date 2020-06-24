from keras.applications import ResNet50
import ResNet50Modifications as ResNetMods
import CustomImageGen
from CustomImageGen import image_generator as img_gen
from keras.optimizers import Adam
import math

#####################################################################
# Load in Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = ResNetMods.change_activation_function(model)
model = ResNetMods.additional_final_layers(model)
#model = ResNetMods.feedback_loop(model)
model.load_weights('./my_weights_24')
model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-10), loss=CustomImageGen.geo_loss, metrics=[CustomImageGen.xyz_error])


#####################################################################
# Test
folders = ["seq-03", "seq-04", "seq-06", "seq-12", "seq-14"]
for i in range(5):
    x_test_files = CustomImageGen.list_of_files(folders[i],"test")
    batch_size = 32;
    test_SPE = int(math.floor(len(x_test_files) / 32))

    mycallback = CustomImageGen.MyMetrics(CustomImageGen.image_generator(x_test_files, batch_size, feedback_loop=False, is_random=True, steps=test_SPE), test_SPE, batch_size)
    #results_train = model.fit(x=CustomImageGen.image_generator(x_train_files, batch_size, False, True, train_SPE), steps_per_epoch=train_SPE, epochs=1, verbose=2,
    #                          validation_data=CustomImageGen.image_generator(x_test_files, batch_size, False, False, test_SPE), validation_steps=test_SPE, validation_freq=1, callbacks=[mycallback])

    model.evaluate(x=CustomImageGen.image_generator(x_test_files, 32, False, True, test_SPE), steps=test_SPE, callbacks=[mycallback])
    file1 = open(".\\Results\\Errors" + str(i) + ".txt", "w")
    errors = mycallback.get_all_errors()
    for e in errors:
        file1.write("%s\n" % (e))
    file1.close()