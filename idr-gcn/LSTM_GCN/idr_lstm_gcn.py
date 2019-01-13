"""
IDR基础模型，双向lstm，arg1、arg2共享lstm参数
"""
import tensorflow as tf
import time
import os
from sklearn import metrics
import numpy as np
import itertools
from LSTM_GCN.pdtb_data import pddata
from multi_head import transformer

epsilon_bn =1e-6
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('classes', 2, 'num-class classification:[2,4]')
# ['Expansion','Contingency','Comparison','Temporal',None]
flags.DEFINE_string('pos_class', 'Comparison', 'positive class in 2-class classification:')
flags.DEFINE_integer('embedding_size', 300, 'embedding size.')
flags.DEFINE_integer('rnn_size', 256, 'hidden_units_size of lstm')
flags.DEFINE_integer('gcn_size', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 64, 'Model string.')
flags.DEFINE_integer('seq_length', 200, 'seq_length.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 15, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, ' keep probability')
flags.DEFINE_float('weight_decay', 0.9,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4


class idr_base_model():

    def __init__(self, batch_size, learning_rate, vocabulary_size, embedding_size, rnn_size, gcn_size, clip_value, epoch,
                 iterations, embedding=None, dropout=0, sess=None, gd_op='Adam', classes=2):
        # embedding matrix
        self.lr = tf.Variable(learning_rate, trainable=False)

        self.trainable = tf.placeholder(tf.bool, None)
        self.lr_decay_factor = tf.placeholder(tf.float32, None)
        self.lr_decay_op = tf.assign(self.lr, self.lr * self.lr_decay_factor)
        self.arg1_max_len = FLAGS.seq_length
        self.arg2_max_len = FLAGS.seq_length
        # self.arg1_max_len = tf.placeholder(tf.int32, shape=(),name="arg1_max_len")
        # self.arg2_max_len = tf.placeholder(tf.int32, shape=(),name="arg2_max_len")

        # batch_size, steps
        self.arg1_ids = tf.placeholder(tf.int32, [batch_size, FLAGS.seq_length], "arg1_ids")
        self.arg1_len = tf.placeholder(tf.int32, [batch_size], "arg1_len")
        self.arg2_ids = tf.placeholder(tf.int32, [batch_size, FLAGS.seq_length], "arg2_ids")
        self.arg2_len = tf.placeholder(tf.int32, [batch_size], "arg2_len")
        self.labels = tf.placeholder(tf.int32, [batch_size], 'labels')
        if self.trainable == True:
            self.input_size = batch_size
        else:
            self.input_size = batch_size  # size of test_dataset
        self.global_step = tf.Variable(0, False)

        # vocabulary: pad, unk, ...
        if type(embedding) == type(None):
            embedding = tf.get_variable('embedding', [vocabulary_size-1, embedding_size], tf.float32,
                                             tf.truncated_normal_initializer)
            # pad 不更新
            pad_embedding = tf.zeros([1, embedding_size], tf.float32)
            embedding = tf.concat([pad_embedding, embedding], axis=0)
        else:
            #
            embedding = tf.constant(embedding, tf.float32)
            # pad_embedding = tf.zeros([1, self.embedding_size], tf.float32)
            # unk_embedding = tf.get_variable('unk_embed', [1, self.embedding_size], tf.float32, tf.truncated_normal_initializer)
            # self.embedding = tf.concat([pad_embedding, unk_embedding, embedding], axis=0)

        # batch_size * steps * embedding_size
        self.arg1_embedded = tf.nn.embedding_lookup(embedding, self.arg1_ids)
        self.arg2_embedded = tf.nn.embedding_lookup(embedding, self.arg2_ids)

        with tf.variable_scope('argument') as arg_scope:
            arg_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            # arg_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size)
            arg_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            # arg_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size)

            init_fw_state = tf.nn.rnn_cell.LSTMStateTuple(
                0.001*tf.truncated_normal([tf.shape(self.labels)[0], rnn_size], dtype=tf.float32),
                0.001 * tf.truncated_normal([tf.shape(self.labels)[0], rnn_size],dtype=tf.float32)
            )

            init_bw_state = tf.nn.rnn_cell.LSTMStateTuple(
                0.001 * tf.truncated_normal([tf.shape(self.labels)[0], rnn_size], dtype=tf.float32),
                0.001 * tf.truncated_normal([tf.shape(self.labels)[0], rnn_size], dtype=tf.float32)
            )

            # 2(f,b),batch_size,steps,hidden_size
            # 2(f,b),2(c,h),batch_size,hidden_size
            arg1_outputs, arg1_final_states = tf.nn.bidirectional_dynamic_rnn(arg_fw_cell, arg_bw_cell, self.arg1_embedded,
                                                                   sequence_length=self.arg1_len, dtype=tf.float32,
                                                                              # initial_state_fw=init_fw_state,
                                                                              # initial_state_bw=init_bw_state,
                                                                              )
            arg_scope.reuse_variables()

            # ??? 用0 or arg1_final_state 初始化cell
            # 2(f,b),batch_size,steps,hidden_size
            # arg2_outputs, _ = tf.nn.bidirectional_dynamic_rnn(arg_fw_cell, arg_bw_cell, self.de_input_embedded,
            #                                                 initial_state_fw=arg1_final_states[0],
            #                                                 initial_state_bw=arg1_final_states[1])
            arg2_outputs, arg2_final_states = tf.nn.bidirectional_dynamic_rnn(arg_fw_cell, arg_bw_cell, self.arg2_embedded,
                                                                sequence_length=self.arg2_len, dtype=tf.float32,
                                                                              # initial_state_fw=init_fw_state,
                                                                              # initial_state_bw=init_bw_state,
                                                                              )
            # final_states
        # bi_arg1_final_states_c = tf.concat([arg1_final_states[0][0], arg1_final_states[1][0]], axis=1)
        # bi_arg2_final_states_c = tf.concat([arg2_final_states[0][0], arg2_final_states[1][0]], axis=1)
        # rnn_out = tf.concat([bi_arg1_final_states_c, bi_arg2_final_states_c], axis=1)
        # rnn_out_drop = tf.layers.dropout(rnn_out, rate=dropout, training=self.trainable)

        arg1_outputs = tf.concat([arg1_outputs[0],arg1_outputs[1]],axis=2)
        arg2_outputs = tf.concat([arg2_outputs[0],arg2_outputs[1]],axis=2)
        arg1_outputs_drop = tf.layers.dropout(arg1_outputs, rate=dropout, training=self.trainable)
        arg2_outputs_drop = tf.layers.dropout(arg2_outputs, rate=dropout, training=self.trainable)

        # GCN
        '''
        # multi-head attention
        if self.trainable == True:
            mode = tf.estimator.ModeKeys.TRAIN
        else:
            mode = tf.estimator.ModeKeys.PREDICT
        _, self.mh_att = transformer.multi_head_attention(num_heads=4,
                                                          queries=arg1_outputs_drop,
                                                          memory=arg2_outputs_drop,
                                                          num_units=FLAGS.rnn_size * 2,
                                                          mode=mode,
                                                          return_attention=True,
                                                          dropout=0.2
                                                          )
        self.mh_att = tf.nn.softmax(tf.reduce_mean(self.mh_att, axis=1))
        print('A_multi:', self.mh_att)
        self.A_matrix = self.mh_att
        '''

        self.X_matrix = tf.concat([arg1_outputs_drop, arg2_outputs_drop], axis=1, name="lstm_initial_X_matrix")


        # self.X_matrix = self.embedding_postprocessor(self.X_matrix,
        #                                              position_embedding_name="position_embeddings",
        #                                              initializer_range=0.02,
        #                                              max_position_embeddings=self.X_matrix.shape[1].value
        #                                              )
        # get adjacent matrix
        self.A_matrix = self.get_A_matrix(arg1_outputs_drop, arg2_outputs_drop)
        # degree matrix d, D = d^-1/2
        self.D_matrix = self.get_D_matrix(self.A_matrix)
        # Normalized matrix Norm_A_matrix = DAD
        self.Norm_A_matrix = tf.matmul(tf.matmul(self.D_matrix,self.A_matrix),self.D_matrix,name="Norm_A_matrix")
        # one-layer graph convolution
        W1 = tf.Variable(tf.truncated_normal([rnn_size*2, gcn_size], stddev=0.1),name='gcn_weights')
        b1 = tf.Variable(tf.zeros([gcn_size]))
        for i in range(self.input_size):
            temp = self.X_matrix[i]
            # temp = tf.nn.dropout(temp, 0.5)
            pre_sup = tf.matmul(temp, W1)
            output = tf.matmul(self.Norm_A_matrix[i], pre_sup)
            output = tf.expand_dims(output,axis=0)
            # bias
            output += b1
            if i == 0:
                outputs = output
            else:
                outputs = tf.concat([outputs,output],axis=0)
        self.outputs = tf.nn.relu(outputs)
        '''
        # get last nodes of arg1&arg2 for classification
        for i in range(batch_size):
            arg1_last_node = tf.expand_dims(outputs[i][self.arg1_len[i]-1],axis=0)
            arg2_last_node = tf.expand_dims(outputs[i][self.arg1_max_len+self.arg2_len[i]-1],axis=0)
            last_node = tf.concat([arg1_last_node,arg2_last_node],axis=1)
            if i == 0:
                last_nodes = last_node
            else:
                last_nodes = tf.concat([last_nodes,last_node],axis=0)
        '''
        # or pooling for classification
        outputs_max = tf.reduce_max(outputs,axis=1)
        outputs_mean = tf.reduce_mean(outputs,axis=1)
        # inputs of dense layer
        dense_inputs = tf.concat([outputs_mean,outputs_max],axis=1)
        # dense_inputs = last_nodes

        # dense layers for classification
        dense1_out = tf.layers.dense(dense_inputs, 64, name='dense1', reuse=False,
                                     # kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)
                                     )
        dense1_out_drop = tf.layers.dropout(dense1_out, rate=dropout, training=self.trainable)
        dense1_out_ac = tf.nn.relu(dense1_out_drop)
        batch_mean, batch_var = tf.nn.moments(dense1_out_ac, [0])
        scale2 = tf.get_variable('bn_scale', initializer=tf.ones([64]))
        beta2 = tf.get_variable('bn_beta', initializer=tf.zeros([64]))
        dense1_out_bn = tf.nn.batch_normalization(dense1_out_ac, batch_mean, batch_var, beta2, scale2, epsilon_bn)
        self.dense2_out = tf.layers.dense(dense1_out_ac, classes, name='dense2', reuse=False,
                                          # kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)
                                          )
        self.out = tf.nn.softmax(self.dense2_out)
        self.predict = tf.argmax(self.dense2_out, axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.dense2_out)
        # self.loss += tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'weights' in var.name])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if gd_op=='GD':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif gd_op=='Adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            # gradients clipping
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
            self.train_op = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)

        # run
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoch):
            print('---epoch %d---' % epoch)
            if epoch > 1:
                sess.run(self.lr_decay_op, feed_dict={self.lr_decay_factor: FLAGS.weight_decay})
            # 历史最小loss
            min_loss = float("inf")
            # 高于最近一次min_loss的loss次数
            pre_counter = 0
            for iteration in range(iterations):
                if classes == 4:
                    arg1, arg2, arg1_len, arg2_len, label, arg1_max_len, arg2_max_len= data.next_multi_rel(batch_size, 'train')
                else:
                    arg1, arg2, arg1_len, arg2_len, label, arg1_max_len, arg2_max_len= data.next_single_rel(batch_size, 'train')
                fd = {self.arg1_ids: arg1,
                      self.arg2_ids: arg2,
                      self.labels: label,
                      self.arg1_len: arg1_len,
                      self.arg2_len: arg2_len,
                      self.trainable: True
                      }
                step = sess.run(self.global_step)
                v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                loss, _ = sess.run([self.loss, self.train_op], feed_dict=fd)
                # 开始不稳定，跳过
                if iteration > 10:
                    # 大于等于最小loss
                    if loss >= min_loss:
                        pre_counter += 1
                    else:
                        pre_counter = 0
                        min_loss = loss
                # 连续20次大于历史最低值，衰减lr
                if pre_counter >= 20:
                    sess.run(self.lr_decay_op, feed_dict={self.lr_decay_factor: 0.99})
                    pre_counter = 0
                if step % 10 == 0:
                    # 重置acc的全局total count
                    sess.run(tf.local_variables_initializer())
                    # 10000 samples
                    if classes == 4:
                        arg1, arg2, arg1_len, arg2_len, label, arg1_max_len, arg2_max_len = data.next_multi_rel(None, 'test')
                    else:
                        arg1, arg2, arg1_len, arg2_len, label, arg1_max_len, arg2_max_len = data.next_single_rel(None, 'test')
                    label_multi = []
                    for one in label:
                        label_multi.append(one)
                    for i in range(len(arg1)//batch_size):

                        fd = {self.arg1_ids: arg1[i*batch_size:(i+1)*batch_size],
                              self.arg2_ids: arg2[i*batch_size:(i+1)*batch_size],
                              self.arg1_len: arg1_len[i*batch_size:(i+1)*batch_size],
                              self.arg2_len: arg2_len[i*batch_size:(i+1)*batch_size],
                              self.trainable: False
                              }
                        predict, dense2_out, pre_pro, lr = sess.run([self.predict, self.dense2_out, self.out, self.lr],
                                                                    feed_dict=fd)

                        if i == 0:
                            predicts = predict
                            dense2_outs = dense2_out
                            pre_proes = pre_pro
                        else:
                            predicts = np.concatenate([predicts,predict],axis=0)
                            dense2_outs = np.concatenate([dense2_outs, dense2_out], axis=0)
                            pre_proes = np.concatenate([pre_proes, pre_pro], axis=0)

                    fd = {self.arg1_ids: arg1[-batch_size:],
                          self.arg2_ids: arg2[-batch_size:],
                          self.arg1_len: arg1_len[-batch_size:],
                          self.arg2_len: arg2_len[-batch_size:],
                          self.trainable: False
                          }
                    predict, dense2_out, pre_pro, lr = sess.run([self.predict,
                                                                 self.dense2_out,
                                                                 self.out,
                                                                 self.lr],
                                                                 feed_dict=fd)
                    predicts = np.concatenate([predicts, predict[-1046%batch_size:]], axis=0)
                    dense2_outs = np.concatenate([dense2_outs, dense2_out[-1046%batch_size:]], axis=0)
                    pre_proes = np.concatenate([pre_proes, pre_pro[-1046%batch_size:]], axis=0)
                    for index in range(len(label)):
                        if predicts[index] in label[index]:
                            label[index] = predicts[index]
                        else:
                            # 为了简便，直接取第一个值。取不同的值会影响不同rel的f1值
                            label[index] = label[index][0]
                    test_loss = sess.run(tf.losses.sparse_softmax_cross_entropy(label, dense2_outs))
                    acc = metrics.accuracy_score(label, predicts)
                    if classes == 2:
                        f1 = metrics.f1_score(label, predicts, pos_label=1, average='binary')
                        auc = metrics.roc_auc_score(label, pre_proes[:, 1])
                        f1_max, f1_max_weight = self._max_f1(label, pre_proes)

                        print(
                                'epoch:%d iter_num:%d train_loss:%.4f test_loss:%.4f acc:%.2f auc:%.3f f1:%.4f f1_max:%.2f f1_weight:%.2f lr: %.4f'
                                % (epoch, iteration, loss, test_loss, acc, auc, f1, f1_max, f1_max_weight, lr))
                    if classes == 4:
                        f1 = metrics.f1_score(label, predicts, average='macro')
                        print( 'epoch:%d iter_num:%d train_loss:%.4f test_loss:%.4f acc:%.2f f1:%.4f lr: %.4f'
                                % (epoch, iteration, loss, test_loss, acc, f1, lr))
                        if f1>0.45:
                            f1_max = self._max_multi_f1(label_multi, pre_proes)
                            print('max_f1:',f1_max)

    # positional embedding
    def embedding_postprocessor(self,
                                input_tensor,
                                use_position_embeddings=True,
                                position_embedding_name="position_embeddings",
                                initializer_range=0.0002,
                                max_position_embeddings=512
                                ):
        """Performs various post-processing on a word embedding tensor.
        Args:
          input_tensor: float Tensor of shape [batch_size, seq_length,
            embedding_size].
          use_token_type: bool. Whether to add embeddings for `token_type_ids`.
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            Must be specified if `use_token_type` is True.
          token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
          token_type_embedding_name: string. The name of the embedding table variable
            for token type ids.
          use_position_embeddings: bool. Whether to add position embeddings for the
            position of each token in the sequence.
          position_embedding_name: string. The name of the embedding table variable
            for positional embeddings.
          initializer_range: float. Range of the weight initialization.
          max_position_embeddings: int. Maximum sequence length that might ever be
            used with this model. This can be longer than the sequence length of
            input_tensor, but cannot be shorter.
          dropout_prob: float. Dropout probability applied to the final output tensor.
        Returns:
          float tensor with same shape as `input_tensor`.
        Raises:
          ValueError: One of the tensor shapes or input values is invalid.
        """
        input_shape = input_tensor.shape.as_list()
        print('input_shape',input_shape)
        batch_size = input_shape[0]
        seq_length = max_position_embeddings
        width = input_shape[2]
        output = input_tensor
        if use_position_embeddings:
            assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
            with tf.control_dependencies([assert_op]):
                full_position_embeddings = tf.get_variable(
                    name=position_embedding_name,
                    shape=[seq_length, width],
                    initializer= tf.truncated_normal_initializer(stddev=initializer_range))
                # Since the position embedding table is a learned variable, we create it
                # using a (long) sequence length `max_position_embeddings`. The actual
                # sequence length might be shorter than this, for faster training of
                # tasks that do not have long sequences.
                #
                # So `full_position_embeddings` is effectively an embedding table
                # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
                # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
                # perform a slice.
                position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                               [seq_length, -1])
                num_dims = len(output.shape.as_list())
                # Only the last two dimensions are relevant (`seq_length` and `width`), so
                # we broadcast among the first dimensions, which is typically just
                # the batch size.
                position_broadcast_shape = []
                for _ in range(num_dims - 2):
                    position_broadcast_shape.append(1)
                position_broadcast_shape.extend([seq_length, width])
                print(position_broadcast_shape)
                position_embeddings = tf.reshape(position_embeddings,
                                                 position_broadcast_shape)
                output += position_embeddings
                print(output)
        return output


    def get_A_matrix(self, agu1, agu2):

        # agu1,agu2: [batch_size,seq_length,embedding_dim]
        self.A_matrix = tf.nn.softmax(
            tf.matmul(agu1, agu2, transpose_a=False, transpose_b=True))  # score_ij :[arg1_seq_length,arg2_seq_length]
        zeros = tf.zeros([FLAGS.batch_size, self.arg1_max_len, self.arg1_max_len], dtype=tf.float32)
        tmp1 = tf.concat([zeros, self.A_matrix], axis=2)
        A_trans = tf.transpose(self.A_matrix, [0, 2, 1])
        zeros = tf.zeros([FLAGS.batch_size, self.arg2_max_len, self.arg2_max_len], dtype=tf.float32)
        tmp2 = tf.concat([A_trans, zeros], axis=2)
        concat = tf.concat([tmp1, tmp2], axis=1)
        # adding self-connection
        diag = tf.matrix_diag(tf.ones([2*FLAGS.seq_length]))
        concat = tf.add(concat,diag, name='adjacent_matrix')
        return concat

    def get_bilinear_matrix(self,arg1,arg2):

        W1 = tf.Variable(tf.truncated_normal([2*FLAGS.rnn_size, 2*FLAGS.rnn_size], stddev=0.1), name='inter_wei')
        zeros = tf.zeros([FLAGS.seq_length,FLAGS.seq_length])
        for i in range(FLAGS.batch_size):
            matrix11 = zeros
            matrix22 = zeros
            matrix12 = tf.matmul(arg1[i], W1, transpose_a=False, transpose_b=True)
            matrix12 = tf.expand_dims(tf.nn.softmax(
                tf.matmul(matrix12, arg2[i], transpose_a=False, transpose_b=True)),axis=0)
            if i==0:
                self.matrix11 = matrix11
                self.matrix22 = matrix22
                self.matrix12 = matrix12
            else:
                self.matrix11 = tf.concat([self.matrix11,matrix11],axis=0)
                self.matrix22 = tf.concat([self.matrix22,matrix22],axis=0)
                self.matrix12 = tf.concat([self.matrix12,matrix12],axis=0)
        self.matrix21 =  tf.transpose(self.matrix12, [0, 2, 1])
        self.matrix1 = tf.concat([self.matrix11,self.matrix12],axis=2)
        self.matrix2 = tf.concat([self.matrix21,self.matrix22],axis=2)
        self.A_matrix = tf.concat([self.matrix1,self.matrix2],axis=1)
        return self.A_matrix


    def get_D_matrix(self,A):
        indices = []
        d_matrix = tf.reduce_sum(A,axis=2,name="degree_matrix")
        print(d_matrix)
        diag = tf.expand_dims(tf.matrix_diag(tf.pow(d_matrix[0],-0.5)), axis=0)
        for i in range(1,FLAGS.batch_size):
            one_diag = tf.expand_dims(tf.matrix_diag(tf.pow(d_matrix[i],-0.5)), axis=0)
            diag = tf.concat([diag, one_diag], axis=0,name="D_matrix")
        print(diag)
        return diag

    # 分段线性函数
    def _piecewise_linear(self, x, boundary=0.5):
        b = tf.ones_like(x) * boundary
        # return tf.where(tf.greater(x, b), -x+1, -10*x+1.5)
        return tf.where(tf.greater(x, b), 0.0, -x+0.5)

    # 线性loss
    def _loss(self, label, logits):
        # batch
        cross_entrop = tf.reduce_sum(-label*tf.log(logits), axis=1)
        # tmp = self._piecewise_linear(logits)
        zero = tf.zeros_like(cross_entrop)
        # batch
        loss = tf.where(tf.equal(self.labels, self.predict), 0.1*cross_entrop, cross_entrop)
        # scalar
        loss = tf.reduce_mean(loss)
        return loss

    # 调节概率界限以获取正类最大f1值
    def _max_f1(self, label, pre_pro, step=0.01):
        f1s = []
        f1s_weight = []
        for boundary in np.arange(0.1, 0.91, step):
            pre = []
            for pro in pre_pro:
                if pro[0] >= boundary:
                    pre.append(0)
                else:
                    pre.append(1)
            f1s.append(metrics.f1_score(label, pre)*100)
            f1s_weight.append(metrics.f1_score(label, pre, average='macro')*100)
        max_f1 = max(f1s)
        max_f1_weight = max(f1s_weight)
        # boundary = 1-(f1s.index(max_f1)+10)*0.01
        return max_f1, max_f1_weight


    def _calculate_acc_f1(self, label, predict, distribution, class_num=4):

        # 预测正确的样本数量
        pre_true_count = 0
        for index, one in enumerate(predict):
            if one in label[index]:
                pre_true_count += 1
        acc = pre_true_count / len(predict)

        f1s = []
        # 针对每一类，单独计算一次f1值
        for rel in range(class_num):
            label_new = []
            predict_new = [int(rel == one) for one in predict]
            for index in range(len(label)):
                if rel in label[index]:
                    if len(label[index]) == 2:
                        # 即正负都对
                        label_new.append(predict_new[index])
                    else:
                        label_new.append(1)
                else:
                    label_new.append(0)
            f1s.append(metrics.f1_score(label_new, predict_new))

        # f1_weight = np.sum([distribution[index] * f1s[index] for index in range(4)])

        return acc, f1s
    # 调整偏移向量offset-[r1,r2,r3,r4]
    # 最终的预测为argmax(predict_probability + offset)
    def _max_multi_f1(self, label, pre_pro):
        # 保存各个类别f1值
        f1s_list = []
        # 保存算术平均f1值
        f1_ave_list = []
        acc_list = []
        # 保存偏移量
        offset_list = []
        # one 即当前的offset，三个参数，固定第一个参数
        for one in itertools.product(np.arange(-0.4, 0.41, 0.05), repeat=3):
            one = list(one)
            # 加入固定的第一个参数0
            one.insert(0, 0)
            offset_list.append(one)
            one = np.array(one)
            # 加入偏移量之后的预测“概率”，单个样本4个类别概率综合已经超过1，但不影响求最可能的类别
            pre_pro_2 = pre_pro + np.tile(one, [len(pre_pro), 1])
            # 求最大概率，即求预测类别
            pre = np.argmax(pre_pro_2, 1)
            # f1 = metrics.f1_score(label, pre, average='macro')
            dis = []
            acc, f1s = self._calculate_acc_f1(label, pre,dis, class_num=4)
            f1_ave_list.append(np.mean(f1s))
            f1s_list.append(f1s)
            acc_list.append(acc)

        # 最大算术平均f1的下标
        max_index = np.argmax(f1_ave_list)

        return f1_ave_list[max_index]


if __name__ == '__main__':
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # ['Expansion', 'Contingency', 'Comparison', 'Temporal']
    rel = FLAGS.pos_class
    print(FLAGS.classes,'-class')
    if FLAGS.classes == 2:
        print('pos_class:', rel)
    data = pddata(rel,FLAGS.seq_length)

    if FLAGS.classes == 4:
        data.gen_whole_data()

    else:
        data.gen_rel_data(rel)

    # 13046:7004    3340    1942    760
    # 14000 19400 22200 24600

    sess = tf.Session(config=config)
    para_dict = {'batch_size': FLAGS.batch_size,
                 'learning_rate': FLAGS.learning_rate,
                 'vocabulary_size': 72847,
                 'embedding_size': FLAGS.embedding_size,
                 'rnn_size': FLAGS.rnn_size,
                 'gcn_size':FLAGS.gcn_size,
                 'clip_value': 5,
                 'epoch': FLAGS.epochs,
                 'iterations': 250,
                 'embedding': data.embedding,
                 'sess':sess,
                 'gd_op':"Adam",
                 'classes':FLAGS.classes}
    print('batch_size:',FLAGS.batch_size)
    print('rnn_size:',FLAGS.rnn_size)
    print('gcn_size:',FLAGS.gcn_size)
    print('lr:',FLAGS.learning_rate)
    print('seq_length:',FLAGS.seq_length)
    print()

    model = idr_base_model(**para_dict)

    print("time:%.1f(minute)" % ((time.time() - start_time) / 60))