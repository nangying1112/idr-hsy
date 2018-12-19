import pickle

batch_size = 100
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
arg1 = ['there','was', 'general', 'feeling', 'that', 'we','would', 'seen', 'the', 'worst']
arg2 = ['the', 'resignation', 'came', 'as', 'great', 'surprise']

max_len = 40
def padding( ids, max_len):
    # 'pad':0
    # print(ids)
    if len(ids) > max_len:
        # print(ids[-max_len:])
        return ids[-max_len:]
    else:
        # print((ids + [0]*max_len)[:max_len])
        return (ids + [0] * max_len)[:max_len]




with open('voc2id.pickle','rb') as f:
    voc2id = pickle.load(f)
arg1_id = []
arg2_id = []
for word in arg1:
    id = voc2id[word]
    arg1_id.append(id)
for word in arg2:
    id = voc2id[word]
    arg2_id.append(id)
arg1_len = len(arg1_id)
arg2_len = len(arg2_id)
arg1_id = padding(arg1_id, max_len)
arg2_id = padding(arg2_id, max_len)
print(arg1_id)
print(arg2_id)
label = 1

case_pickle = []
case_arg1 = []
case_arg2 = []
case_arg1_len = []
case_arg2_len = []
case_label = []

for i in range(batch_size):
    case_arg1.append(arg1_id)
    case_arg2.append(arg2_id)
    case_arg1_len.append(arg1_len)
    case_arg2_len.append(arg2_len)
    case_label.append(label)
case_pickle.append(case_arg1)
case_pickle.append(case_arg2)
case_pickle.append(case_arg1_len)
case_pickle.append(case_arg2_len)
case_pickle.append(case_label)
print(len(case_label))

with open('case.pickle','wb') as f:
    pickle.dump(case_pickle,f,pickle.HIGHEST_PROTOCOL)
'''
# a = [1,2,3]
# a = np.array(a)
# a = a*2
# print(a)
arg1 = ['there','was', 'general', 'feeling', 'that', 'we','would', 'seen', 'the', 'worst']

arg2 = ['the', 'resignation', 'came', 'as', 'great', 'surprise']
with open('A_matirx.pickle','rb') as f:
    A_matrix = pickle.load(f)
M= A_matrix[40][1]
M = M[0:10]
new = []
for nums in M:
    new.append(nums[0:6])
new = np.array(new)
print(new)

y = np.array(new)
y = y*100
df = pd.DataFrame(y,columns=[x for x in arg2],index=[x for x in arg1])
# sns.heatmap(df,annot=True)
print(y)
f, ax = plt.subplots(figsize = (9, 6))
# cmap = sns.palplot(sns.color_palette("Blues"))

cmap = sns.cubehelix_palette(start = 1.5, rot =1.6, gamma=0.2, as_cmap = True)
# cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(df, cmap = cmap, linewidths = 0.5, ax = ax,cbar=False)
plt.show()
ax.tick_params(axis='y',labelsize=18)
ax.tick_params(axis='x',labelsize=18)

ax.set_title('Before GCN',fontsize=30)
ax.set_xlabel('arg2',fontsize=18)
ax.set_ylabel('arg1',fontsize=18)

f.savefig('before_gcn.pdf', bbox_inches='tight')