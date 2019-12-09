from unittest import TestCase
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import LocalisationNetwork
import numpy as np

class TestXyz_error(TestCase):
    def test_xyz_error(self):
        inputs = Input(shape=(5,))
        hidden_layer = Dense(4)(inputs)
        predictions = Dense(3)(hidden_layer)

        model = Model(inputs=inputs, outputs=predictions)

        weights1 = np.array([(0.12, 0.52, -0.55, -0.86), (0.72, -0.15, 0.81, -0.35), (-0.14, 0.65, 0.69, -0.24),
                             (0.34, -0.43, 0.51, 0.27), (0.91, -0.54, 0.13, 0.61)])
        bias1 = np.array([0,0,0,0])

        weights2 = np.array([(0.05, -0.42, 0.62), (-0.71, -0.25, 0.62), (0.08, -0.22, 0.34), (0.22, 0.67, -0.81)])
        bias2 = np.array([0,0,0])
        model.layers[1].set_weights([weights1,bias1])
        model.layers[2].set_weights([weights2,bias2])

        model.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=[LocalisationNetwork.xyz_error])

        x_test = np.array([(0.62, 0.47, -0.83, 0.14, -0.73), (-0.32, 0.43, -0.45, 0.55, 0.62), (-0.34, 0.45, 0.56, 0.81, -0.65), (0.47, -0.68, 0.63, 0.94, -0.74), (0.67, 0.37, 0.65, -0.37, 0.33), (0.43, -0.65, 0.74, -0.33, 0.47)])
        y_pred = np.array([(-0.28116903, -0.459356, 0.519044), (1.044567, 0.20012699, -0.42541394), (-0.030155, -0.381275, 0.581105), (-0.68979496, -0.16718295, 0.3432559), (-0.621412, -0.91566, 1.393861), (-0.555941, -0.08948898, 0.27793297)])
        y_true = np.array([(-0.3, -0.45, 0.52), (1.0, 0.2, -0.4), (-0.02, -0.4, 0.59), (-0.7, -0.2, 0.35), (-0.62, -0.9, 1.4), (-0.56, -0.09, 0.28)])

        xyz_errors = np.zeros((6,1))
        for i in range(0,6):
            xyz_errors[i] = np.sqrt(np.square(y_true[i,0]-y_pred[i,0]) + np.square(y_true[i,1]-y_pred[i,1]) + np.square(y_true[i,2]-y_pred[i,2]))

        xyz_errors_sorted = np.sort(a=xyz_errors,axis=0)
        median = (xyz_errors_sorted[2] + xyz_errors_sorted[3])/2.0
        results = model.evaluate(x=x_test, y=y_true)

        self.assertTrue(np.absolute(median[0] - results[1]) < 0.000001)

