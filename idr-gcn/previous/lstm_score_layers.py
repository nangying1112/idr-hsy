from previous.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])



class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, A_matrix, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=True,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        # self.support = placeholders['support']
        self.A_matrix = A_matrix
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        print(type(input_dim),output_dim)

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights' ] = glorot([input_dim, output_dim],
                                                        name='weights' )
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        outputs = x[0]
        for i in range(x.shape[0].value):
            temp = x[i]

            print('temp.shape', x[i].shape)

            # dropout
            if self.sparse_inputs:
                temp = sparse_dropout(temp, 1-self.dropout, self.num_features_nonzero)
            else:
                temp = tf.nn.dropout(temp, 1-self.dropout)

            # convolve
            if not self.featureless:
                pre_sup = dot(temp, self.vars['weights'],
                              sparse=False)
            else:
                pre_sup = self.vars['weights']
            output = dot(self.A_matrix[i], pre_sup, sparse=False)
            output = tf.expand_dims(output,axis=0)
            # bias
            if self.bias:
                output += self.vars['bias']
            if i == 0:
                outputs = output
            else:
                outputs = tf.concat([outputs,output],axis=0)
        print(self.A_matrix.shape,outputs.shape)

        self.embedding = outputs #output
        return self.act(outputs)


class MLP(Layer):


    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.,
                 sparse_inputs=False,act=tf.nn.relu,bias=True):
        super(MLP,self).__init__()
        self.act = act
        self.bias = bias
        self.sparse_inputs = sparse_inputs
        self.dropout = dropout
        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_1'] = glorot([input_dim, hidden_dim],
                                            name='weights_1' )
            self.vars['weights_2'] = glorot([hidden_dim, output_dim],
                                            name='weights_2')


            if self.bias:
                self.vars['bias1'] = zeros([hidden_dim], name='bias1')
                self.vars['bias2'] = zeros([output_dim], name='bias2')






    def _call(self, inputs):
        hidden = tf.matmul(inputs, self.vars['weights_1']) + self.vars['bias1']
        drop1 = tf.nn.dropout(hidden, self.dropout)
        BN1= tf.layers.batch_normalization(drop1)
        hidden_out = self.act(BN1)
        output = tf.matmul(hidden_out, self.vars['weights_2']) + self.vars['bias2']
        drop3 = tf.nn.dropout(output,self.dropout)
        BN2 = tf.layers.batch_normalization(drop3)
        return self.act(BN2)



class Classifier(Layer):

    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.,
                 sparse_inputs=False,act=tf.nn.relu,bias=True):
        super(Classifier,self).__init__()
        self.act = act
        self.bias = bias
        self.sparse_inputs = sparse_inputs
        self.dropout = dropout
        with tf.variable_scope(self.name + '_vars'):
            print(self.name)
            self.vars['weights_1'] = glorot([input_dim, hidden_dim],
                                            name='weights_1' )
            self.vars['weights_2'] = glorot([hidden_dim, output_dim],
                                            name='weights_2')
            if self.bias:
                self.vars['bias1'] = zeros([hidden_dim], name='bias1')
                self.vars['bias2'] = zeros([output_dim], name='bias2')

    def _call(self, inputs):
        hidden = tf.matmul(inputs, self.vars['weights_1']) + self.vars['bias1']
        #BN1= tf.layers.batch_normalization(hidden)
        #drop1 = tf.nn.dropout(BN1, self.dropout)
        hidden_out = self.act(hidden)
        out = tf.matmul(hidden_out, self.vars['weights_2']) + self.vars['bias2']
        BN2 = tf.layers.batch_normalization(out)
        drop2 = tf.nn.dropout(BN2, self.dropout)
        output = self.act(drop2)
        return output
