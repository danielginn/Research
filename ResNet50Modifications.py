from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
def additional_final_layers(model):
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, name='fc1')(x)
    xyzq = Dense(7, name='xyzq_output')(x)
    #q = Dense(4, name='q_output')(x)

    return Model(inputs=model.inputs, outputs=xyzq)

def change_activation_function(model):
    model.layers[4].activation = 'elu'   # Activation_1
    model.layers[9].activation = 'elu'   # Activation_2
    model.layers[12].activation = 'elu'  # Activation_3
    model.layers[18].activation = 'elu'  # Activation_4
    model.layers[21].activation = 'elu'  # Activation_5
    model.layers[24].activation = 'elu'  # Activation_6
    model.layers[28].activation = 'elu'  # Activation_7
    model.layers[31].activation = 'elu'  # Activation_8
    model.layers[34].activation = 'elu'  # Activation_9
    model.layers[38].activation = 'elu'  # Activation_10
    model.layers[41].activation = 'elu'  # Activation_11
    model.layers[44].activation = 'elu'  # Activation_12
    model.layers[50].activation = 'elu'  # Activation_13
    model.layers[53].activation = 'elu'  # Activation_14
    model.layers[56].activation = 'elu'  # Activation_15
    model.layers[60].activation = 'elu'  # Activation_16
    model.layers[63].activation = 'elu'  # Activation_17
    model.layers[66].activation = 'elu'  # Activation_18
    model.layers[70].activation = 'elu'  # Activation_19
    model.layers[73].activation = 'elu'  # Activation_20
    model.layers[76].activation = 'elu'  # Activation_21
    model.layers[80].activation = 'elu'  # Activation_22
    model.layers[83].activation = 'elu'  # Activation_23
    model.layers[86].activation = 'elu'  # Activation_24
    model.layers[92].activation = 'elu'  # Activation_25
    model.layers[95].activation = 'elu'  # Activation_26
    model.layers[98].activation = 'elu'  # Activation_27
    model.layers[102].activation = 'elu' # Activation_28
    model.layers[105].activation = 'elu' # Activation_29
    model.layers[108].activation = 'elu' # Activation_30
    model.layers[112].activation = 'elu' # Activation_31
    model.layers[115].activation = 'elu' # Activation_32
    model.layers[118].activation = 'elu' # Activation_33
    model.layers[122].activation = 'elu' # Activation_34
    model.layers[125].activation = 'elu' # Activation_35
    model.layers[128].activation = 'elu' # Activation_36
    model.layers[132].activation = 'elu' # Activation_37
    model.layers[135].activation = 'elu' # Activation_38
    model.layers[138].activation = 'elu' # Activation_39
    model.layers[142].activation = 'elu' # Activation_40
    model.layers[145].activation = 'elu' # Activation_41
    model.layers[148].activation = 'elu' # Activation_42
    model.layers[154].activation = 'elu' # Activation_43
    model.layers[157].activation = 'elu' # Activation_44
    model.layers[160].activation = 'elu' # Activation_45
    model.layers[164].activation = 'elu' # Activation_46
    model.layers[167].activation = 'elu' # Activation_47
    model.layers[170].activation = 'elu' # Activation_48
    model.layers[174].activation = 'elu' # Activation_49

    return model