from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
def additional_final_layers(model):
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    xyzq = Dense(7, name='xyz')(x)
    #q = Dense(4, name='q')(x)  # **** Assuming softmax (rho/theta/phi) and quaternians

    return Model(inputs=model.inputs, outputs=xyzq)