# -*- coding: utf-8 -*-
# @Time    : 2018/9/7 9:50
# @Author  : WCheng

"""
IDR基础模型，双向lstm，arg1、arg2共享lstm参数
"""
import tensorflow as tf
import time
import os
from sklearn import metrics
import numpy as np
from lstm_pdtb_data import pddata

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('classes', 2, 'num-class classification:[2,4]')
# ['Expansion','Contingency','Comparison','Temporal',None]
flags.DEFINE_string('pos_class', 'Expansion', 'positive class in 2-class classification:')
flags.DEFINE_integer('embedding_size', 300, 'embedding size.')
flags.DEFINE_integer('rnn_size', 128, 'hidden_units_size of lstm')
flags.DEFINE_integer('gcn_size', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 64, 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 8, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, ' keep probability')
flags.DEFINE_float('weight_decay', 0.9,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4


class idr_base_model():

    def __init__(self, batch_size, learning_rate, vocabulary_size, embedding_size, rnn_size, clip_value, epoch,
                 iterations, embedding=None, dropout=0, sess=None, gd_op='Adam', classes=2):
        # embedding matrix
        self.lr = tf.Variable(learning_rate, trainable=False)

        self.trainable = tf.placeholder(tf.bool, None)
        self.lr_decay_factor = tf.placeholder(tf.float32, None)
        self.lr_decay_op = tf.assign(self.lr, self.lr * self.lr_decay_factor)
        # batch_size, steps
        self.arg1_ids = tf.placeholder(tf.int32, [None, None], "arg1_ids")
        self.arg1_len = tf.placeholder(tf.int32, [FLAGS.batch_size], "arg1_len")
        self.arg2_ids = tf.placeholder(tf.int32, [None, None], "arg2_ids")
        self.arg2_len = tf.placeholder(tf.int32, [FLAGS.batch_size], "arg2_len")
        self.labels = tf.placeholder(tf.int32, [FLAGS.batch_size], 'labels')

        # arg1_step = tf.shape(self.arg1_ids)[-1]
        # arg2_step = tf.shape(self.arg2_ids)[-1]

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

        #
        bi_arg1_final_states_c = tf.concat([arg1_final_states[0][0], arg1_final_states[1][0]], axis=1)
        bi_arg2_final_states_c = tf.concat([arg2_final_states[0][0], arg2_final_states[1][0]], axis=1)
        # [batch_size,2*hidden_size] => [batch_size,4*hidden_size]
        rnn_out = tf.concat([bi_arg1_final_states_c, bi_arg2_final_states_c], axis=1)

        rnn_out_drop = tf.layers.dropout(rnn_out, rate=dropout, training=self.trainable)

        dense1_out = tf.layers.dense(rnn_out_drop, 64, name='dense1', reuse=False)
        # dense1_out_bn = tf.layers.batch_normalization(dense1_out, trainable=trainable)
        dense1_out_ac = tf.nn.relu(dense1_out)
        dense1_out_drop = tf.layers.dropout(dense1_out_ac, rate=dropout, training=self.trainable)
        self.dense2_out = tf.layers.dense(dense1_out_drop, classes, name='dense2', reuse=False)
        # self.dense2_out_bn = tf.layers.batch_normalization(dense1_out, trainable=trainable)
        self.out = tf.nn.softmax(self.dense2_out)
        self.predict = tf.argmax(self.dense2_out, axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.dense2_out)
        # self.loss = self._loss(tf.one_hot(self.labels,2), tf.nn.softmax(self.dense2_out))

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
                    arg1, arg2, arg1_len, arg2_len, label= data.next_multi_rel(batch_size, 'train')
                else:
                    arg1, arg2, arg1_len, arg2_len, label= data.next_single_rel(batch_size, 'train')

                fd = {self.arg1_ids: arg1, self.arg2_ids: arg2, self.labels: label, self.arg1_len: arg1_len,
                      self.arg2_len: arg2_len, self.trainable: True}
                step = sess.run(self.global_step)
                # print(step)
                v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                # gd = sess.run([self.gd_pre, self.gd_pos], feed_dict=fd)

                loss, _ = sess.run([self.loss, self.train_op], feed_dict=fd)
                # print(loss)
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
                        arg1, arg2, arg1_len, arg2_len,label = data.next_multi_rel(None, 'test')
                    else:
                        arg1, arg2, arg1_len, arg2_len,label = data.next_single_rel(None, 'test')
                    label_multi = []
                    for one in label:
                        label_multi.append(one)
                    for i in range(len(arg1)//batch_size):
                        fd = {self.arg1_ids: arg1[i*batch_size:(i+1)*batch_size], self.arg2_ids: arg2[i*batch_size:(i+1)*batch_size], self.arg1_len: arg1_len[i*batch_size:(i+1)*batch_size],
                                self.arg2_len: arg2_len[i*batch_size:(i+1)*batch_size], self.trainable: False}
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
                          self.arg2_len: arg2_len[-batch_size:], self.trainable: False}
                    predict, dense2_out, pre_pro, lr = sess.run([self.predict, self.dense2_out, self.out, self.lr],
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

                        # fpr, tpr, thresholds = metrics.roc_curve(label, pre_pro[:, 1], pos_label=1)
                        # auc = metrics.auc(fpr, tpr)
                        auc = metrics.roc_auc_score(label, pre_proes[:, 1])
                        f1_max, f1_max_weight = self._max_f1(label, pre_proes)
                        # if abs(acc[1] - f1)>0.5:
                        #     out = sess.run([self.out], feed_dict=fd)
                        print(
                                'epoch:%d iter_num:%d train_loss:%.4f test_loss:%.4f acc:%.2f auc:%.3f f1:%.4f f1_max:%.2f f1_weight:%.2f lr: %.4f'
                                % (epoch, iteration, loss, test_loss, acc, auc, f1, f1_max, f1_max_weight, lr))
                    if classes == 4:
                        f1 = metrics.f1_score(label, predicts, average='macro')
                        print( 'epoch:%d iter_num:%d train_loss:%.4f test_loss:%.4f acc:%.2f f1:%.4f lr: %.4f'
                                % (epoch, iteration, loss, test_loss, acc, f1, lr))


    def train(self, sess, data):
        pass

    def test(self):
        pass

    def run(self):
        pass

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

    def _max_multi_f1(self, label, pre_pro):
        exp_pro = pre_pro[:, 0]
        con_pro = pre_pro[:, 1]
        com_pro = pre_pro[:, 2]
        tem_pro = pre_pro[:, 3]

        def is_true(label_index, one_label):
            if label_index == one_label:
                return 1
            else:
                return 0

        exp_label = [is_true(0, sample) for sample in label]
        con_label = [is_true(1, sample) for sample in label]
        com_label = [is_true(2, sample) for sample in label]
        tem_label = [is_true(3, sample) for sample in label]

        self._max_f1(exp_label, exp_pro)


if __name__ == '__main__':
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # ['Expansion', 'Contingency', 'Comparison', 'Temporal']
    rel = FLAGS.pos_class
    print(FLAGS.classes, '-class')
    if FLAGS.classes == 2:
        print('pos_class:', rel)
    data = pddata(rel)
    if FLAGS.classes == 4:
        data.gen_whole_data()
    else:
        data.gen_rel_data(rel)

    # 13046:7004    3340    1942    760
    # 14000 19400 22200 24600

    sess = tf.Session(config=config)
    para_dict = {'batch_size': FLAGS.batch_size, 'learning_rate': FLAGS.learning_rate, 'vocabulary_size': 72847, 'embedding_size': 300,
                 'rnn_size': FLAGS.rnn_size, 'clip_value': 5, 'epoch': FLAGS.epochs, 'iterations': 250, 'embedding': data.embedding,
                 'sess':sess, 'gd_op':"Adam", 'classes':FLAGS.classes}

    model = idr_base_model(**para_dict)

    print("time:%.1f(minute)" % ((time.time() - start_time) / 60))