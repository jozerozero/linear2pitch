import tensorflow as tf
import numpy as np
# import tensorflow.contrib.layers as tf_contrib_layers
# from cell_state_lstm import LSTMwithCellState


class FrameProjection:
    """Projection layer to r * num_mels dimensions or num_mels dimensions
	"""

    def __init__(self, shape=80, activation=None, scope=None):
        """
		Args:
			shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
			activation: callable, activation function
			scope: FrameProjection scope.
		"""
        super(FrameProjection, self).__init__()

        self.shape = shape
        self.activation = activation

        self.scope = 'Linear_projection' if scope is None else scope
        self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = self.dense(inputs)

            return output


def shuffle_aligned_list(data):
    num = len(data)
    p = np.random.permutation(num)
    return [data[i] for i in p]


def batch_generator(data, batch_size):
    batch_count = 0
    # print("batch_size", batch_size)
    while True:
        if batch_count * batch_size + batch_size >= len(data):
            batch_count = 0
            data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield data[start: end]


class Model:

    def __init__(self):
        pass

    def inference(self, input_feature, label, is_reuse):
        with tf.variable_scope("model", reuse=is_reuse):

            print(input_feature)
            # exit()
            kernel_1 = tf.get_variable(name="kernel_1", shape=[100, 1, 1])
            input_feature = tf.expand_dims(input_feature, axis=-1)
            input_feature = tf.squeeze(tf.nn.conv1d(input_feature, kernel_1, 1, 'VALID'), axis=-1)
            # print(input_feature)
            # exit()

            pitch_projection_1 = FrameProjection(500, scope="cbhg_pitch_projection", activation=tf.nn.relu)
            pitch_output = pitch_projection_1(input_feature)
            # print(pitch_output)
            # assert linear_outputs.shape[0] == tower_pitch_input[i].shape[0]
            kernel_2 = tf.get_variable(name="kernel_2", shape=[100, 1, 1])
            pitch_output = tf.expand_dims(pitch_output, axis=-1)
            pitch_output = tf.squeeze(tf.nn.conv1d(pitch_output, kernel_2, 1, 'VALID'), axis=-1)
            # print(pitch_output)
            # exit()
            pitch_projection_2 = FrameProjection(10, scope="cbhg_pitch_projection_2", activation=tf.nn.relu)
            pitch_output = pitch_projection_2(pitch_output)

            pitch_projection_3 = FrameProjection(1, scope="cbhg_pitch_projection_3", activation=tf.nn.relu)
            pitch_output = pitch_projection_3(pitch_output)

            loss = tf.sqrt(tf.reduce_sum(tf.square(pitch_output - label)))

        return loss


if __name__ == '__main__':

    dataset = list()
    batch_size = 128
    training_step = 100000

    for line in open("training_data/train_2.txt").readlines():
        line = line.strip().split("|")
        dataset.append([line[2], line[4], line[6]])

    iterator = batch_generator(data=dataset, batch_size=batch_size)

    graph = tf.get_default_graph()
    with graph.as_default():
        feature = tf.placeholder(dtype=tf.float32, shape=[1, None, 1025], name="linear_frame")
        label = tf.placeholder(dtype=tf.float32, shape=[1, None, 1], name="pitch")
        learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

        model = Model()
        batch_loss_list = list()
        for index, sample in enumerate(tf.split(feature, 1)):
            sample = tf.squeeze(sample, 0)
            y = label[index]
            if index == 0:
                is_reuse = False
            else:
                is_reuse = True
            sample_mse = model.inference(sample, y, is_reuse=is_reuse)
            batch_loss_list.append(sample_mse)

        mse = tf.reduce_mean(batch_loss_list)
        train_opt = tf.train.AdamOptimizer(0.001).minimize(loss=mse)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        exit()
        with tf.Session(config=config) as session:
            tf.global_variables_initializer().run()
            for _ in range(training_step):
                batch_data = next(iterator)
                batch_input_list = list()
                batch_label_list = list()

                for sample in batch_data:
                    batch_input_list = np.expand_dims(np.load("training_data/linear/%s" % sample[0]), axis=0)
                    batch_label_list = np.expand_dims(np.expand_dims(np.load("training_data/pitch/%s" % sample[2]), axis=0), axis=-1)
                    # batch_input_list = np.asarray(batch_input_list)
                    # batch_label_list = np.asarray(batch_label_list)
                    # p[[rin(batch_input_list)
                    # exit()
                    _, batch_loss = session.run([train_opt, mse],
                                     feed_dict={feature: batch_input_list,
                                                label: batch_label_list})

                    print(batch_loss)
