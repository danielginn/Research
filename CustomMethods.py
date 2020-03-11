import tensorflow as tf

class Mean_XYZ_Error(tf.keras.metrics.Metric):

    def __init__(self, name='mean_xyz_error', batch=32, **kwargs):
        super(Mean_XYZ_Error, self).__init__(name=name, **kwargs)
        self.avg_error_sum = self.add_weight(name='ms', initializer='zeros')
        self.div = self.add_weight(name='div', initializer='zeros')
        self.batch = self.add_weight(name='batch', initializer='zeros')
        self.batch.assign(batch)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_diff = tf.math.subtract(y_pred,y_true)
        xyz_error = tf.math.reduce_sum(tf.math.reduce_euclidean_norm(y_diff[:, 0:3], 1))
        self.avg_error_sum.assign_add(xyz_error)
        self.div.assign_add(self.batch)

    def reset_states(self):
        self.avg_error_sum.assign(0.)
        self.div.assign(0)

    def result(self):
        return tf.math.divide(self.avg_error_sum, self.div)


class Median_XYZ_Error(tf.keras.metrics.Metric):

    def __init__(self, name='median_xyz_error', batch=32, array_size=30000, **kwargs):
        super(Median_XYZ_Error, self).__init__(name=name, **kwargs)
        self.batch = self.add_weight(name='batch', initializer='zeros')
        self.batch.assign(batch)
        self.array_size = self.add_weight(name='array_size', initializer='zeros')
        self.array_size.assign(array_size)
        self.array = self.add_weight(name='array', shape=[array_size], initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.array = tf.slice(self.array,0,3)+tf.slice(self.array,3,3)

    #def reset_states(self):
    #    self.array = [0,0,0,0,0,0]

    def result(self):
        return self.array