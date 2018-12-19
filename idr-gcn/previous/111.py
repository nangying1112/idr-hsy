import pickle as pkl
import tensorflow as tf


'''
sess = tf.Session()
# 测试稀疏tensor生成
indices = [[0,0],[1,1],[2,2]]
a = tf.ones([1,2])
print('j',sess.run(tf.pow(a,2)))
# print(sess.run(a[0][0]+a[0][1]))
print(a)
b = tf.ones([1,1])
c = tf.ones([1,1])
values = tf.concat([a,b,c],axis=1)
print(values)
values = tf.squeeze(values,axis=0)
print(values)
print(type(indices))
print(sess.run(values))
dense_shape = [3,8]
sp = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
print(sess.run(sp))
print(sp)
'''


'''
# 测试softmax
ss = np.array([[1.,2.,8.],[4.,5.,9.]])
print(ss)
# ss = tf.convert_to_tensor(ss)
# print(ss)
ss = tf.nn.softmax(ss)
print(sess.run(ss))
'''


'''
# 把四个类别分开
for sample in PDTB_data[0]:
    if 'Comparison' in sample[0]:
        comp.append(sample)
    if 'Expansion' in sample[0]:
        exp.append(sample)
    if 'Contingency' in sample[0]:
        cont.append(sample)
    if 'Temporal' in sample[0]:
        temp.append(sample)
with open('PDTB_data/comp','wb') as f:
    pkl.dump(comp,f,pkl.HIGHEST_PROTOCOL)
with open('PDTB_data/exp','wb') as f:
    pkl.dump(exp,f,pkl.HIGHEST_PROTOCOL)
with open('PDTB_data/cont','wb') as f:
    pkl.dump(cont,f,pkl.HIGHEST_PROTOCOL)
with open('PDTB_data/temp','wb') as f:
    pkl.dump(temp,f,pkl.HIGHEST_PROTOCOL)
print(len(comp),len(exp),len(cont),len(temp))
'''

'''
dense_shape = [2,3,3]
indices = []
A = [[[1.,2.,3.],[2.,3.,4.],[3.,4.,5.]],
    [[4.,5.,6.],[5.,6.,7.],[6.,7.,8.]]]
print(A)
v = tf.reduce_sum(A,axis=2)
print(v)
with tf.Session() as sess:
    print(sess.run(v))
    v = tf.reshape(v,[1,-1])
    v = tf.squeeze(v,axis=0)
    v = tf.pow(v,-0.5)
    print(sess.run(v))
    for i in range(2):
        for j in range(3):
            indices.append([i,j,j])
    D = tf.SparseTensor(indices=indices, values=v, dense_shape=dense_shape)
    D = tf.sparse_tensor_to_dense(D,
                              default_value=0,
                              validate_indices=True,
                              name=None)
    print(sess.run(D))
'''

A = [[1,2],[2,3]]
B = [[3,4],[4,5]]
C = tf.matmul(A,B)
with tf.Session() as sess:
    print(sess.run(C))
