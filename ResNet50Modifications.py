from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
def additional_final_layers(model):
    x = model.output
    x = GlobalAveragePooling2D()(x)  # **** Assuming 2D, with no arguments required
    x = Dense(1024, activation='relu', name='fc1')(x)  # **** Assuming relu
    xyz = Dense(3, name='xyz')(x)  # **** Assuming softmax is the correct activation here
    q = Dense(4, name='q')(x)  # **** Assuming softmax (rho/theta/phi) and quaternians

    return Model(inputs=model.inputs, outputs=[xyz,q])