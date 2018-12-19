from previous.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class LSTM_GCN():

    def __init__(self, placeholders, vocab_size, embedding_size, hidden_size, **kwargs):

        print('LSTM_GCN initing...')
        self.agu1 = placeholders['batch_agu1']
        self.agu2 = placeholders['batch_agu2']
        self.labels = placeholders['batch_labels']
        self.emb_dropout_keep_prob = placeholders['emb_dropout_keep_prob']
        self.rnn_dropout_keep_prob = placeholders['rnn_dropout_keep_prob']
        self.agu1_seq_length = placeholders['agu1_seq_length']
        self.agu2_seq_length = placeholders['agu2_seq_length']
        self.embed = placeholders['embed']
        self.dropout_keep_prob = placeholders['dropout']
        self.num_classes = placeholders['batch_labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.loss = 0
        self.accuracy = 0

        # word embedding layer
        with tf.variable_scope('word_embedding'):
            #self.vocab_embedding = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-0.25,0.25),name='vocab_embedding')
            self.vocab_embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                            trainable=False, name="vocab_embedding")
            self.embedding_init = self.vocab_embedding.assign(self.embed)
            self.agu1_embedding = tf.nn.embedding_lookup(self.vocab_embedding,self.agu1)
            self.agu2_embedding = tf.nn.embedding_lookup(self.vocab_embedding,self.agu2)
        # word embedding dropout
        with tf.variable_scope('embedding_dropout'):
            self.agu1_embedding = tf.nn.dropout(self.agu1_embedding, self.emb_dropout_keep_prob)
            self.agu2_embedding = tf.nn.dropout(self.agu2_embedding, self.emb_dropout_keep_prob)

        with tf.variable_scope('argument') as arg_scope:
            arg_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            # arg_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size)
            arg_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            # arg_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size)

            arg1_outputs, arg1_final_states = tf.nn.bidirectional_dynamic_rnn(arg_fw_cell, arg_bw_cell, self.agu1_embedding,
                                                                   sequence_length=self.agu1_seq_length, dtype=tf.float32,
                                                                              # initial_state_fw=init_fw_state,
                                                                              # initial_state_bw=init_bw_state,
                                                                              )

            arg_scope.reuse_variables()
            arg2_outputs, arg2_final_states = tf.nn.bidirectional_dynamic_rnn(arg_fw_cell, arg_bw_cell, self.agu2_embedding,
                                                                sequence_length=self.agu2_seq_length, dtype=tf.float32,
                                                                              # initial_state_fw=init_fw_state,
                                                                              # initial_state_bw=init_bw_state,
                                                                              )

        #
        bi_arg1_final_states_c = tf.concat([arg1_final_states[0][0], arg1_final_states[1][0]], axis=1)
        bi_arg2_final_states_c = tf.concat([arg2_final_states[0][0], arg2_final_states[1][0]], axis=1)
        # [batch_size,2*hidden_size] => [batch_size,4*hidden_size]
        self.rnn_outputs = tf.concat([bi_arg1_final_states_c, bi_arg2_final_states_c], axis=1)
        self.build()

    def build(self):

        dense1_out = tf.layers.dense(self.rnn_outputs, 64, name='dense1', reuse=False,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.5))
        # dense1_out_bn = tf.layers.batch_normalization(dense1_out)
        dense1_out_ac = tf.nn.sigmoid(dense1_out)
        dense1_out_drop = tf.layers.dropout(dense1_out_ac, rate=self.dropout_keep_prob, training=True)
        self.outputs = tf.layers.dense(dense1_out_drop, 2, name='dense2', reuse=False,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(0.5))
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
        print('model_vars:', variables)
        self.vars = {var.name: var for var in variables}
        self._loss()
        self._accuracy()
        '''
        gradients = self.optimizer.compute_gradients(self.loss)
        print('hh', [var for grad, var in gradients if grad is not None])
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.opt_op = self.optimizer.apply_gradients(capped_gradients)
        '''
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.placeholders['learning_rate'])
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            # 梯度截断
            gradients, _ = tf.clip_by_global_norm(gradients, 5.)
            self.opt_op = optimizer.apply_gradients(zip(gradients, v))

        '''
        # A_matrix ,X_matrix
        self.A_matrix = self.get_A_matrix(self.agu1_rnn_outputs, self.agu2_rnn_outputs)
        self.X_matrix = self.agu2_rnn_outputs
        self.layers.append(GraphConvolution(input_dim=self.agu2_rnn_outputs.shape[2].value,
                                            output_dim=200,
                                            A_matrix=self.A_matrix,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            ))

        self.layers.append(GraphConvolution(input_dim=200,
                                            output_dim=100,
                                            A_matrix=self.A_matrix,
                                            placeholders=self.placeholders,
                                             act=lambda x: x,
                                             dropout=True,
                                             logging=self.logging))

        self.layers.append(Classifier(input_dim=400,
                                      hidden_dim=200,
                                      output_dim=4,
                                      bias=True,
                                      dropout=0.5,
                                      act=tf.nn.sigmoid,
                                      sparse_inputs=False
                                      ))
        '''
        

    def _loss(self):

        # print([var for var in tf.trainable_variables() if 'weights' in var.name])
        print('trainable_vars:',[var for var in tf.trainable_variables()])


        self.loss += tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.labels,
                                                  )

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels
                                        )
    def predict(self):
        with tf.name_scope('predict'):
            predictions = tf.nn.softmax(self.outputs, name='predictions')
            # print('pred:',predictions)
        return predictions

    def get_A_matrix(self, agu1, agu2):

        # agu1,agu2: [batch_size,seq_length,embedding_dim]
        # agu1_trans = tf.transpose(agu1,[1,0,2])
        # agu2_trans = tf.transpose(agu2,[1,0,2])
        for i in range(agu2.shape[0].value):
            score_ij = tf.nn.softmax(
                tf.matmul(agu2[i], agu1[i], transpose_a=False, transpose_b=True))  # score_ij :[seq_length,seq_length]
            score_ij = tf.expand_dims(score_ij, axis=0, name='score_ij')
            # print(score_ij.shape)
            if i == 0:
                self.A_matrix = score_ij
            else:
                self.A_matrix = tf.concat([self.A_matrix, score_ij], axis=0, name='A_matrix')

        print('A_matrix:',self.A_matrix)
        return self.A_matrix

