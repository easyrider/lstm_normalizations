"""adapted from https://github.com/OlavHN/bnlstm to store separate population statistics per state"""
import tensorflow as tf, numpy as np

RNNCell = tf.nn.rnn_cell.RNNCell


class BNLSTMCell(RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''

    def __init__(self,
                 num_units,
                 max_bn_steps,
                 forget_bias=1.0,
                 is_training_tensor=False,
                 initial_scale=0.1,
                 activation=tf.tanh,
                 decay=0.95):
        """
		* max bn steps is the maximum number of steps for which to store separate population stats
		"""
        self._num_units = num_units
        self._training = is_training_tensor
        self._max_bn_steps = max_bn_steps
        self._activation = activation
        self._decay = decay
        self._forget_bias = forget_bias
        self._initial_scale = initial_scale

    @property
    def state_size(self):
        return self._num_units, self._num_units, 1

    @property
    def output_size(self):
        return self._num_units

    def _batch_norm(self,
                    x,
                    name_scope,
                    step,
                    epsilon=1e-7,
                    no_offset=False,
                    set_forget_gate_bias=False):
        '''Assume 2d [batch, values] tensor'''

        with tf.variable_scope(name_scope):
            size = x.get_shape().as_list()[1]

            scale = tf.get_variable(
                'scale', [size],
                initializer=tf.truncated_normal_initializer(
                    self._initial_scale))
            if no_offset:
                offset = 0
            elif set_forget_gate_bias:
                offset = tf.get_variable(
                    'offset', [size], initializer=offset_initializer())
            else:
                offset = tf.get_variable(
                    'offset', [size], initializer=tf.zeros_initializer)

            # print(self._max_bn_steps)
            pop_mean_all_steps = tf.get_variable(
                'pop_mean_all', [self._max_bn_steps, size],
                initializer=tf.zeros_initializer,
                trainable=False)
            pop_var_all_steps = tf.get_variable(
                'pop_var_all', [self._max_bn_steps, size],
                initializer=tf.ones_initializer(),
                trainable=False)

            step = tf.minimum(step, self._max_bn_steps - 1)
            # tf.Print(step, [tf.reduce_mean(step)], 'step')

            pop_mean = pop_mean_all_steps[step]
            pop_var = pop_var_all_steps[step]

            batch_mean, batch_var = tf.nn.moments(x, [0])

            # batch_mean = tf.Print(batch_mean, [tf.reduce_mean(batch_mean)], 'batch_mean')
            # batch_var = tf.Print(batch_var, [tf.reduce_mean(batch_var)], 'batch_var')

            def batch_statistics():
                pop_mean_new = pop_mean * self._decay + batch_mean * (
                    1 - self._decay)
                pop_var_new = pop_var * self._decay + batch_var * (
                    1 - self._decay)
                with tf.control_dependencies([
                        pop_mean.assign(pop_mean_new),
                        pop_var.assign(pop_var_new)
                ]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var,
                                                     offset, scale, epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(x, pop_mean, pop_var, offset,
                                                 scale, epsilon)

            if type(self._training) == bool:
                if self._training:
                    return batch_statistics()
                else:
                    return population_statistics()
            else:
                return tf.cond(self._training, batch_statistics,
                               population_statistics)

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):

            c, h, step = state
            _step = tf.squeeze(tf.gather(tf.cast(step, tf.int32), 0))

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable(
                'W_xh', [x_size, 4 * self._num_units],
                initializer=tf.orthogonal_initializer)
            W_hh = tf.get_variable(
                'W_hh', [self._num_units, 4 * self._num_units],
                initializer=identity_initializer(0.9))

            hh = tf.matmul(h, W_hh)
            xh = tf.matmul(x, W_xh)

            hidden = tf.cond(_step < self._max_bn_steps-1,
             lambda: self._batch_norm(hh, 'hh', _step, set_forget_gate_bias=True)+
              self._batch_norm(xh, 'xh', _step, no_offset=True),
                lambda: xh + hh)

            f, i, o, j = tf.split(hidden, 4, 1)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(
                i) * self._activation(j)

            bn_new_c = tf.cond(_step < self._max_bn_steps - 1,
                               lambda: self._batch_norm(new_c, 'c', _step),
                               lambda: new_c)

            new_h = self._activation(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h, step + 1)


def identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = np.identity(size) * scale
        t[:, size * 2:size * 3] = np.identity(size) * scale
        t[:, size * 3:] = np.identity(size) * scale
        return tf.constant(t, dtype)

    return _initializer


def offset_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        size = shape[0]
        assert size % 4 == 0
        size = size // 4
        res = [np.ones((size)), np.zeros((size * 3))]
        return tf.constant(np.concatenate(res, axis=0), dtype)

    return _initializer
