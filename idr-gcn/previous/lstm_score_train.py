from __future__ import division
from __future__ import print_function

import time
from previous.metrics import f1_accuracy
from previous.lstm_score_utils import *
from previous.lstm_score_models import LSTM_GCN
import random
import os
import numpy as np


# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'lstm_gcn', 'Model string.')
flags.DEFINE_string('class_num', 2, 'num-class classification:[2,4]')
# ['Expansion','Contingency','Comparison','Temporal',None]
flags.DEFINE_string('pos_class', 'Expansion', 'positive class in 2-class classification:')
flags.DEFINE_string('embedding_size', 300, 'embedding size.')
flags.DEFINE_string('hidden_size', 256, 'hidden_units_size of lstm')
flags.DEFINE_string('batch_size', 100, 'Model string.')
flags.DEFINE_string('max_seq_length', 50, 'Model string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('num_epochs_per_decay', 1, 'for learning rate decay .')
flags.DEFINE_integer('iters', 4000, 'Number of epochs to train.')
flags.DEFINE_integer('gcn_hidden', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('embed_dropout', 0.7, ' keep probability.')
flags.DEFINE_float('rnn_dropout', 0.7, ' keep probability')
flags.DEFINE_float('dropout', 0.5, ' keep probability')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')


def train():

    mode = 'train'
    data, y_train, word2id, embed = load_corpus(mode,FLAGS.class_num,FLAGS.pos_class)
    dataset_size = len(data)
    vocab_size = len(word2id)
    model_func = LSTM_GCN
    # load PDTB_data
    placeholders = {
        'batch_agu1': tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None]),
        'batch_agu2': tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None]),
        'batch_labels': tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.class_num)),
        'agu1_seq_length':tf.placeholder(tf.int32,shape=(FLAGS.batch_size)),
        'agu2_seq_length': tf.placeholder(tf.int32, shape=(FLAGS.batch_size)),
        'embed': tf.placeholder(tf.float32, shape=(None,FLAGS.embedding_size)),
        'emb_dropout_keep_prob':tf.placeholder(tf.float32, name='emb_dropout_keep_prob'),
        'rnn_dropout_keep_prob' :tf.placeholder(tf.float32, name='rnn_dropout_keep_prob'),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'learning_rate':tf.placeholder(tf.float32, name='learning_rate')
    }

    # Initialize session
    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_conf)
    with tf.variable_scope('model'):
        model = model_func(placeholders, vocab_size, FLAGS.embedding_size,FLAGS.hidden_size, logging=True)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model

    for iter in range(FLAGS.iters):

        current_epoch = iter//(dataset_size//FLAGS.batch_size)
        batch_agu1,batch_agu2, batch_labels,agu1_seq_length,agu2_seq_length = get_batch(data, y_train, word2id,
                                                                                FLAGS.max_seq_length,
                                                                                FLAGS.batch_size,iter)
        t = time.time()
        learning_rate = tf.train.exponential_decay(0.001,
                                                   current_epoch,
                                                   decay_steps=FLAGS.num_epochs_per_decay,
                                                  decay_rate=0.95)
        lr =sess.run(learning_rate)
        feed_dict = construct_feed_dict(
            batch_agu1, batch_agu2,batch_labels, agu1_seq_length,agu2_seq_length,placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['emb_dropout_keep_prob']: FLAGS.embed_dropout})
        feed_dict.update({placeholders['rnn_dropout_keep_prob']: FLAGS.rnn_dropout})
        feed_dict.update({placeholders['learning_rate']: lr})
        feed_dict.update({placeholders['embed']: embed})

        # Training step

        outs = sess.run([model.embedding_init,model.opt_op,model.outputs, model.loss,
                         model.accuracy,model.agu1_embedding,model.rnn_outputs,model.predict()], feed_dict=feed_dict)
        # print('embedding:',outs[-1][0][2])
        # print('rnn_outputs:',outs[-2].shape,len(outs[-2][0:10]))

        # print(outs[-2])
        # print()
        # print(outs[-1])
        # print()

        print("epoch:",current_epoch,"iter:", '%04d' % (iter + 1), "train_loss=", "{:.5f}".format(outs[3]),
              "train_acc=", "{:.5f}".format(outs[4]),"time=", "{:.5f}".format(time.time() - t))

        saver = tf.train.Saver()
        # Validation
        if iter!=0 and iter%100 == 0 :
            save_path = saver.save(sess,'PDTB_data/model/batch32')
            run_valid_test(sess,model,placeholders,'valid',FLAGS.class_num,FLAGS.pos_class)
            run_valid_test(sess,model,placeholders,'test',FLAGS.class_num,FLAGS.pos_class)

def run_valid_test(sess,model,placeholders,mode,class_num,pos_class):

    print('run '+mode+' dataset')
    data, y, word2id, embed = load_corpus(mode,class_num,pos_class)
    dataset_size = len(data)
    preds = []
    for iter in range(dataset_size//FLAGS.batch_size+1):
        # print('iter:',iter)
        batch_agu1, batch_agu2, batch_labels, agu1_seq_length, agu2_seq_length = get_valid_test_batch(data,y,word2id,
                                                                                           FLAGS.max_seq_length,
                                                                                           FLAGS.batch_size,iter)


        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(
            batch_agu1, batch_agu2, batch_labels, agu1_seq_length, agu2_seq_length, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['emb_dropout_keep_prob']: FLAGS.embed_dropout})
        feed_dict.update({placeholders['rnn_dropout_keep_prob']: FLAGS.rnn_dropout})
        feed_dict.update({placeholders['learning_rate']: 0.001})
        feed_dict.update({placeholders['embed']: embed})

        res = sess.run(model.predict(),feed_dict=feed_dict)
        preds.append(res)
    print(preds[0])
    print(mode,'acc,f1', f1_accuracy(preds,y,dataset_size,FLAGS.batch_size))


if __name__ == '__main__':

    train()
