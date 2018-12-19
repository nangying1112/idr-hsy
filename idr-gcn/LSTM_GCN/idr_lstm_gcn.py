"""
IDR基础模型，双向lstm，arg1、arg2共享lstm参数
"""
import tensorflow as tf
import time
import os
from sklearn import metrics
import numpy as np
import scipy.sparse as sp
from pdtb_data import pddata
import itertools

epsilon_bn =1e-2
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('classes', 2, 'num-class classification:[2,4]')
# ['Expansion','Contingency','Comparison','Temporal',None]
flags.DEFINE_string('pos_class', 'Expansion', 'positive class in 2-class classification:')
flags.DEFINE_integer('embedding_size', 300, 'embedding size.')
flags.DEFINE_integer('rnn_size', 256, 'hidden_units_size of lstm')
flags.DEFINE_integer('gcn_size', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 64, 'Model string.')
flags.DEFINE_integer('seq_length', 50, 'seq_length.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
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
        self.arg1_max_len = tf.placeholder(tf.int32, shape=(),name="arg1_max_len")
        self.arg2_max_len = tf.placeholder(tf.int32, shape=(),name="arg2_max_len")

        # print(self.arg1_max_len)
        # batch_size, steps
        self.arg1_ids = tf.placeholder(tf.int32, [batch_size, None], "arg1_ids")
        self.arg1_len = tf.placeholder(tf.int32, [batch_size], "arg1_len")
        self.arg2_ids = tf.placeholder(tf.int32, [batch_size, None], "arg2_ids")
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


        bi_arg1_final_states_c = tf.concat([arg1_final_states[0][0], arg1_final_states[1][0]], axis=1)
        bi_arg2_final_states_c = tf.concat([arg2_final_states[0][0], arg2_final_states[1][0]], axis=1)
        # [batch_size,2*hidden_size] => [batch_size,4*hidden_size]
        rnn_out = tf.concat([bi_arg1_final_states_c, bi_arg2_final_states_c], axis=1)

        arg1_outputs_drop = tf.layers.dropout(arg1_outputs, rate=dropout, training=self.trainable)

        arg2_outputs_drop = tf.layers.dropout(arg2_outputs, rate=dropout, training=self.trainable)

        rnn_out_drop = tf.layers.dropout(rnn_out, rate=dropout, training=self.trainable)

        # GCN
        self.A_matrix = self.get_A_matrix(arg1_outputs_drop, arg2_outputs_drop)
        self.A_concat = self.get_A_concat_matrix(self.A_matrix)
        self.D_matrix = self.get_D_matrix(self.A_concat)
        self.N_A_matirx = tf.matmul(self.D_matrix,self.A_concat)
        self.N_A_matirx = tf.matmul(self.N_A_matirx,self.D_matrix)
        print('N_A',self.N_A_matirx)
        '''
        for i in range(self.input_size):
            piece = tf.matmul(self.D_matrix[i],self.A_concat[i])
            piece = tf.expand_dims(tf.matmul(piece,self.D_matrix[i]),axis=0)
            if i == 0:
                self.N_A_matirx  = piece
            else:
                self.N_A_matirx = tf.concat([self.N_A_matirx,piece],axis=0)
        '''
        self.X_matrix = tf.concat([arg1_outputs_drop,arg2_outputs_drop],axis=2)

        W1 = tf.Variable(tf.truncated_normal([rnn_size*2, gcn_size], stddev=0.1),name='gcn_weights')
        b1 = tf.Variable(tf.zeros([gcn_size]))
        for i in range(self.input_size):
            temp = tf.concat([self.X_matrix[0][i],self.X_matrix[1][i]],axis=1)
            # temp = tf.nn.dropout(temp, 0.5)

            pre_sup = tf.matmul(temp, W1)
            output = tf.matmul(self.N_A_matirx[i], pre_sup)
            output = tf.expand_dims(output,axis=0)
            # bias
            output += b1
            if i == 0:
                outputs = output
            else:
                outputs = tf.concat([outputs,output],axis=0)
        self.outputs = tf.nn.relu(outputs)

        for i in range(batch_size):
            arg1_last_node = tf.expand_dims(outputs[i][self.arg1_len[i]-1],axis=0)
            arg2_last_node = tf.expand_dims(outputs[i][self.arg1_max_len+self.arg2_len[i]-1],axis=0)
            last_node = tf.concat([arg1_last_node,arg2_last_node],axis=1)
            if i == 0:
                last_nodes = last_node
            else:
                last_nodes = tf.concat([last_nodes,last_node],axis=0)

        outputs_max = tf.reduce_max(outputs,axis=1)
        outputs_mean = tf.reduce_mean(outputs,axis=1)

        dense_inputs = tf.concat([last_nodes,outputs_max],axis=1)
        dense_inputs = last_nodes

        dense1_out = tf.layers.dense(dense_inputs, 64, name='dense1', reuse=False,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.5))
        dense1_out_drop = tf.layers.dropout(dense1_out, rate=dropout, training=self.trainable)
        dense1_out_ac = tf.nn.relu(dense1_out_drop)

        batch_mean, batch_var = tf.nn.moments(dense1_out_ac, [0])
        scale2 = tf.get_variable('bn_scale', initializer=tf.ones([64]))
        beta2 = tf.get_variable('bn_beta', initializer=tf.zeros([64]))
        dense1_out_bn = tf.nn.batch_normalization(dense1_out_ac, batch_mean, batch_var, beta2, scale2, epsilon_bn)

        self.dense2_out = tf.layers.dense(dense1_out_ac, classes, name='dense2', reuse=False,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.5))

        self.out = tf.nn.softmax(self.dense2_out)
        self.predict = tf.argmax(self.dense2_out, axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.dense2_out)
        self.loss += tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'weights' in var.name])


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if gd_op=='GD':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif gd_op=='Adam':
                optimizer = tf.train.AdamOptimizer(self.lr)

            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            # 梯度截断
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
                      self.trainable: True,
                      self.arg1_max_len:arg1_max_len,
                      self.arg2_max_len:arg2_max_len}

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
                              self.trainable: False,
                              self.arg1_max_len:arg1_max_len,
                              self.arg2_max_len:arg2_max_len}

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
                          self.trainable: False,
                          self.arg1_max_len: arg1_max_len,
                          self.arg2_max_len: arg2_max_len
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
                        if f1>0.39:
                            f1_max = self._max_multi_f1(label_multi, pre_proes)
                            print('max_f1:',f1_max)

    def embedding_postprocessor(input_tensor,
                                use_token_type=False,
                                token_type_ids=None,
                                token_type_vocab_size=16,
                                token_type_embedding_name="token_type_embeddings",
                                use_position_embeddings=True,
                                position_embedding_name="position_embeddings",
                                initializer_range=0.02,
                                max_position_embeddings=512,
                                dropout_prob=0.1):
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
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        output = input_tensor

        if use_token_type:
            if token_type_ids is None:
                raise ValueError("`token_type_ids` must be specified if"
                                 "`use_token_type` is True.")
            token_type_table = tf.get_variable(
                name=token_type_embedding_name,
                shape=[token_type_vocab_size, width],
                initializer=create_initializer(initializer_range))
            # This vocab will be small so we always do one-hot here, since it is always
            # faster for a small vocabulary.
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
            token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings,
                                               [batch_size, seq_length, width])
            output += token_type_embeddings

        if use_position_embeddings:
            assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
            with tf.control_dependencies([assert_op]):
                full_position_embeddings = tf.get_variable(
                    name=position_embedding_name,
                    shape=[max_position_embeddings, width],
                    initializer=create_initializer(initializer_range))
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
                position_embeddings = tf.reshape(position_embeddings,
                                                 position_broadcast_shape)
                output += position_embeddings

        output = layer_norm_and_dropout(output, dropout_prob)
        return output

    def get_A_matrix(self, agu1, agu2):

        # agu1,agu2: [batch_size,seq_length,embedding_dim]
        # agu1_trans = tf.transpose(agu1,[1,0,2])
        # agu2_trans = tf.transpose(agu2,[1,0,2])
        for i in range(self.input_size):
            agu1_fw_bw = tf.concat([agu1[0][i],agu1[1][i]],axis=1)
            agu2_fw_bw = tf.concat([agu2[0][i],agu2[1][i]],axis=1)
            score_ij = tf.nn.softmax(
                tf.matmul(agu1_fw_bw, agu2_fw_bw, transpose_a=False, transpose_b=True))  # score_ij :[arg1_seq_length,arg2_seq_length]
            score_ij = tf.expand_dims(score_ij, axis=0, name='score_ij')

            if i == 0:
                self.A_matrix = score_ij
            else:
                self.A_matrix = tf.concat([self.A_matrix, score_ij], axis=0, name='A_matrix')

        # print('A_matrix:',self.A_matrix)
        return self.A_matrix

    def get_A_concat_matrix(self,A_matrix):

        zeros = tf.zeros([FLAGS.batch_size,self.arg1_max_len,self.arg1_max_len],dtype=tf.float32)
        print(zeros)
        tmp1 = tf.concat([zeros,A_matrix],axis=2)
        A_trans = tf.transpose(A_matrix,[0,2,1])
        zeros = tf.zeros([FLAGS.batch_size, self.arg2_max_len, self.arg2_max_len], dtype=tf.float32)
        tmp2 = tf.concat([A_trans,zeros],axis=2)
        concat = tf.concat([tmp1,tmp2],axis=1)
        return concat

    def get_D_matrix(self,A):
        indices = []
        # dense_shape = [FLAGS.batch_size,self.arg1_max_len+self.arg2_max_len,self.arg1_max_len+self.arg2_max_len]
        # print(dense_shape)
        v = tf.reduce_sum(A,axis=2)
        # v = tf.reshape(v,[1,-1])
        # v = tf.squeeze(v,axis=0)
        # v = tf.pow(v,-0.5)
        print(v)
        diag = tf.expand_dims(tf.matrix_diag(tf.pow(v[0],-0.5)), axis=0)
        for i in range(1,FLAGS.batch_size):
        #     for j in range(self.arg1_max_len+self.arg2_max_len):
        #         indices.append([i,j,j])
        # D = tf.SparseTensor(indices=indices, values=v, dense_shape=dense_shape)
        # D = tf.sparse_tensor_to_dense(D,
        #                           default_value=0,
        #                           validate_indices=True,
        #                           name=None)
            one_diag = tf.expand_dims(tf.matrix_diag(tf.pow(v[i],-0.5)), axis=0)
            diag = tf.concat([diag, one_diag], axis=0)
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

    # def _max_multi_f1(self, label, pre_pro):
    #     exp_pro = pre_pro[:, 0]
    #     con_pro = pre_pro[:, 1]
    #     com_pro = pre_pro[:, 2]
    #     tem_pro = pre_pro[:, 3]
    #
    #     def is_true(label_index, one_label):
    #         if label_index == one_label:
    #             return 1
    #         else:
    #             return 0
    #
    #     exp_label = [is_true(0, sample) for sample in label]
    #     con_label = [is_true(1, sample) for sample in label]
    #     com_label = [is_true(2, sample) for sample in label]
    #     tem_label = [is_true(3, sample) for sample in label]
    #
    #     self._max_f1(exp_label, exp_pro)

        # 计算各个类别的f1,鉴于label(y-true)存在多个label的特殊性，使用以后api效果欠佳
        # label-每个样本可能存在多个类别
        # distribution-各个类别的样本比例，list类型
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    para_dict = {'batch_size': FLAGS.batch_size, 'learning_rate': FLAGS.learning_rate, 'vocabulary_size': 72847, 'embedding_size': FLAGS.embedding_size,
                 'rnn_size': FLAGS.rnn_size, 'gcn_size':FLAGS.gcn_size,'clip_value': 5, 'epoch': FLAGS.epochs, 'iterations': 250, 'embedding': data.embedding,
                 'sess':sess, 'gd_op':"Adam", 'classes':FLAGS.classes}
    print('batch_size:',FLAGS.batch_size)
    print('rnn_size:',FLAGS.rnn_size)
    print('gcn_size:',FLAGS.gcn_size)
    print('lr:',FLAGS.learning_rate)
    print('seq_length:',FLAGS.seq_length)
    print()

    model = idr_base_model(**para_dict)

    print("time:%.1f(minute)" % ((time.time() - start_time) / 60))