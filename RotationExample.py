from scipy.spatial.transform import Rotation as R
import numpy as np
import CustomImageGen
from keras.models import Model
from keras.layers import Input, Activation, Dense
from keras.optimizers import Adam

R1 = R.from_euler('zyx',[45, 0, 0], degrees=True)
R2 = R.from_euler('zyx',[46, 0, 0], degrees=True)
R3 = R.from_euler('zyx',[48, 0, 0], degrees=True)
R4 = R.from_euler('zyx',[50, 0, 0], degrees=True)
R5 = R.from_euler('zyx',[52, 0, 0], degrees=True)
R6 = R.from_euler('zyx',[54, 0, 0], degrees=True)
R7 = R.from_euler('zyx',[56, 0, 0], degrees=True)

q1 = R1.as_quat()
q2 = R2.as_quat()
q3 = R3.as_quat()
q4 = R4.as_quat()
q5 = R5.as_quat()
q6 = R6.as_quat()
q7 = R7.as_quat()

q1array = np.zeros((6, 4))
q2to7 = np.zeros((6, 4))

q1array[0, :] = q1
q1array[1, :] = q1
q1array[2, :] = q1
q1array[3, :] = q1
q1array[4, :] = q1
q1array[5, :] = q1

q2to7[0, :] = q2
q2to7[1, :] = q3
q2to7[2, :] = q4
q2to7[3, :] = q5
q2to7[4, :] = q6
q2to7[5, :] = q7

#print(q2to7)

a = Input(shape=(4,))
b = Dense(4, name='myoutput')(a)
model = Model(inputs=a, outputs=b)
#for layer in model.layers:
#    layer.trainable = False

model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-10), loss='mean_squared_error', metrics=['mae'])
metrics = CustomImageGen.Metrics()
model.fit(x=q2to7, y=q1array, batch_size=2, epochs=1, verbose=2, callbacks=[metrics], validation_data=(q2to7, q1array))
print(metrics.losses)