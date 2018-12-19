import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score,accuracy_score


def masked_softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss=tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=preds, pos_weight=0.2, name=None)
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels):
    """Accuracy with masking."""

    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)



    # F1_score


def f1_accuracy(preds, labels,dataset_size,batch_size):

    # for i in range(len(preds)):
        # print(i)
        # print(preds[i])
    all_preds = preds[0]
    # print(len(preds))
    for i in range(1,len(preds)):
        # print(i)
        # print(preds[i])
        if i == len(preds)-1:
            # print(preds[-(dataset_size % batch_size):])
            # print(all_preds.shape, preds[-1][-(dataset_size % batch_size):].shape)
            all_preds = np.vstack((all_preds,preds[-1][-(dataset_size%batch_size):]))
        else:
            all_preds = np.vstack((all_preds,preds[i]))

    # preds_pool = np.argmax(all_preds, axis=1).tolist()
    # print(preds_pool)
    labels_pool = np.argmax(labels, axis=1).tolist()
    unique_label = np.unique(labels_pool)

    f1s = []
    f1s_weight = []
    for boundary in np.arange(0.1, 0.91, 0.01):
        pre = []
        for pro in all_preds:
            if pro[0] >= boundary:
                pre.append(0)
            else:
                pre.append(1)
        f1s.append(f1_score(labels_pool, pre, average='macro') * 100)
    max_f1 = max(f1s)
    # max_f1_weight = max(f1s_weight)
    # boundary = 1-(f1s.index(max_f1)+10)*0.01

    acc = accuracy_score(labels_pool, pre, normalize=True, sample_weight=None)
    # 4-class
    # F1 = f1_score(labels_pool, preds_pool, labels=unique_label, average='macro')
    # F1 = f1_score(y_true, y_pred, pos_label=1,average='binary')
    return  acc,max_f1