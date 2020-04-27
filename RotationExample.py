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

q1 = R1.as_quat()
q2 = R2.as_quat()
q3 = R3.as_quat()
q4 = R4.as_quat()
q5 = R5.as_quat()
q6 = R6.as_quat()

q1array = np.zeros((5, 7))
q2to6 = np.zeros((5, 7))

q1array[0, 3:7] = q1
q1array[1, 3:7] = q1
q1array[2, 3:7] = q1
q1array[3, 3:7] = q1
q1array[4, 3:7] = q1

q2to6[0, 3:7] = q2
q2to6[1, 3:7] = q3
q2to6[2, 3:7] = q4
q2to6[3, 3:7] = q5
q2to6[4, 3:7] = q6

#print(q2to6)

a = Input(shape=(7,))
model = Model(inputs=a, outputs=a)
#for layer in model.layers:
#    layer.trainable = False

model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-10), loss='mean_squared_error', metrics=[CustomImageGen.q_error])
result = model.evaluate(x=q2to6, y=q1array)
print(result)