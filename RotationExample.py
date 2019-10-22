import scipy
from scipy.spatial.transform import Rotation as R
import numpy as np

v1 = np.array([0.5, 0.5, 0.2])
v2 = np.array([0.5, -0.6, 0.1])

something = np.arctan2(v2[1],v2[0]) - np.arctan2(v1[1],v1[0])
print(something)
Rbf_f = R.from_euler('z',something,degrees=False)
print("Rbf_f:",Rbf_f.as_euler('zyx',degrees=True))
#Rcf_f = R.from_euler('zyx',[40,10,15],degrees=True)
#Rbc_f = R.from_euler('zyx',Rbf_f.as_euler('zyx',degrees=True) - Rcf_f.as_euler('zyx',degrees=True),degrees=True)

print(Rbc_f.as_euler('zyx',degrees=True)[0])