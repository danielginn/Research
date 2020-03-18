import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import CustomMethods

inputs = Input(shape=(5,))
hidden_layer = Dense(4)(inputs)
predictions = Dense(7)(hidden_layer)

model = Model(inputs=inputs, outputs=predictions)

weights1 = np.array([(0.12, 0.52, -0.55, -0.86), (0.72, -0.15, 0.81, -0.35), (-0.14, 0.65, 0.69, -0.24),
                     (0.34, -0.43, 0.51, 0.27), (0.91, -0.54, 0.13, 0.61)])
bias1 = np.array([0,0,0,0])

weights2 = np.array([(0.05, -0.42, 0.62, 0.14, 0.81, -0.54, -0.02), (-0.71, -0.25, 0.62, 0.44, -0.37, 0.71, 0.08),
                     (0.08, -0.22, 0.34, -0.81, -0.33, 0.29, 0.71), (0.22, 0.67, -0.81, 0.46, 0.67, -0.61, -0.37)])
bias2 = np.array([0,0,0,0,0,0,0])
model.layers[1].set_weights([weights1,bias1])
model.layers[2].set_weights([weights2,bias2])

for layer in model.layers:
    layer.trainable = False
model.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=[CustomMethods.Mean_XYZ_Error(batch=3)])
#model.summary()
for layer in model.layers:
    layer.trainable = False

# Test inputs arbitrarily chosen
x_test = np.array([(0.62, 0.47, -0.83, 0.14, -0.73),
                   (-0.32, 0.43, -0.45, 0.55, 0.62),
                   (-0.34, 0.45, 0.56, 0.81, -0.65),
                   (0.47, -0.68, 0.63, 0.94, -0.74),
                   (0.67, 0.37, 0.65, -0.37, 0.33),
                   (0.43, -0.65, 0.74, -0.33, 0.47)]).astype(np.float32)


# y outputs from model.predition(x_test)
y_pred = np.array([(-0.28116903, -0.459356, 0.519044, 0.04214301, -0.5115801, 0.47157708, -0.05442897),
                   (1.044567, 0.200127, -0.42541397, -0.44605693, 1.6032941, -1.6598661, 0.01791695),
                   (-0.03015498, -0.38127497, 0.581105, -1.0691489, -0.67217696, 0.6224439, 0.976834),
                   (-0.68979496, -0.16718298, 0.343256, -0.03582197, -1.3347809, 1.322675, 0.27426398),
                   (-0.6214119, -0.91565996, 1.393861, -0.17258298, -0.49567592, 0.79362494, 0.494478),
                   (-0.555941, -0.08948898, 0.27793297, 0.510076, -0.38429204, 0.570791, -0.15075898)])

# rounded y_preds to simulate "truth data" that is different from the predicted outputs of model during training
y_true = np.round(y_pred, 2).astype(np.float32)

# true xyz error
xyz_errors = np.zeros((6, 1))
xyz_error_sum = 0
for i in range(0, 3):
    xyz_errors[i] = np.sqrt(np.square(y_true[i, 0] - y_pred[i, 0]) + np.square(y_true[i, 1] - y_pred[i, 1]) + np.square(y_true[i, 2] - y_pred[i, 2]))
    xyz_error_sum += xyz_errors[i]
print("my calc of xyz_avg_error:",xyz_error_sum/3)

xyz_error_sum = 0
for i in range(3, 6):
    xyz_errors[i] = np.sqrt(np.square(y_true[i, 0] - y_pred[i, 0]) + np.square(y_true[i, 1] - y_pred[i, 1]) + np.square(y_true[i, 2] - y_pred[i, 2]))
    xyz_error_sum += xyz_errors[i]
print("my calc of xyz_avg_error:",xyz_error_sum/3)

xyz_error_sum = 0
for i in range(0, 6):
    xyz_errors[i] = np.sqrt(np.square(y_true[i, 0] - y_pred[i, 0]) + np.square(y_true[i, 1] - y_pred[i, 1]) + np.square(y_true[i, 2] - y_pred[i, 2]))
    xyz_error_sum += xyz_errors[i]
print("my calc of total xyz_avg_error:",xyz_error_sum/6)

#results1 = model.fit(x=x_test, y=y_true, batch_size=3, verbose=1, epochs=2, shuffle=False)
results2 = model.evaluate(x=x_test, y=y_true, batch_size=3, steps=2, verbose=1)
print(results2)