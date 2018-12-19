import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

arg1 = ['there','was', 'general', 'feeling', 'that', 'we','would', 'seen', 'the', 'worst']

arg2 = ['the', 'resignation', 'came', 'as', 'great', 'surprise']
print(len(arg1))
print(len(arg2))
#
# ny = np.random.randint(1,3,60)
# print(ny)
ny = [[0.02529644, 0.02326207 ,0.022909,   0.02374114, 0.02306403 ,0.02424928],
 [0.02860111, 0.02756062 ,0.02606343, 0.02650556, 0.0267643 , 0.02806335],
 [0.02804793, 0.02737791 ,0.02453158, 0.02601462 ,0.02571998, 0.02577537],
 [0.02910206, 0.02695408, 0.02501705, 0.02643219 ,0.02529199, 0.02592398],
 [0.02888523 ,0.02671066, 0.02486665, 0.02585432 ,0.02494474 ,0.0258748 ],
 [0.02926508 ,0.02850547 ,0.02640476, 0.02612566 ,0.02566665, 0.02794774],
 [0.03078582 ,0.03003959, 0.02678426, 0.02619871 ,0.02585264, 0.02763409],
 [0.0321291 , 0.03331739 ,0.02577996, 0.02309655, 0.02197365, 0.02343777],
 [0.02912227 ,0.02913403 ,0.02283064, 0.02169588, 0.02102751, 0.02114883],
 [0.0311951 , 0.0307133 , 0.02291423 ,0.02139364 ,0.0219319  ,0.02177548]]
y = np.array(ny)

df = pd.DataFrame(y,columns=[x for x in arg2],index=[x for x in arg1])
# sns.heatmap(df,annot=True)

f, ax = plt.subplots(figsize = (10, 4))
cmap = sns.cubehelix_palette(start = 1.5, rot =1.6, gamma=0.2, as_cmap = True)
# cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(df, cmap = cmap, linewidths = 0.05, ax = ax)
plt.show()

ax.set_title('heatmap')
ax.set_xlabel('arg1')
ax.set_ylabel('arg2')

f.savefig('sns_heatmap_normal.jpg', bbox_inches='tight')