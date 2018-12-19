from nltk.corpus import stopwords
import pickle
# nltk.download()
# stop_words = set(stopwords.words('english'))
# print(stop_words)

'''
stop_words =['by', 'ourselves', 'their', 'the', 'or', 'won', 'most',
             'needn', 'they', 'an', 'them', 'with', 'we', 'other', 'hasn',
             'between', 'now', 'over', 'what', 'mustn', 'his',
             'who', 'again', 'and', 'had', 'theirs',
             'to', 'it', 'is', 'few', 'i', 'd', 'having', 'y', 'any', 'when', 'of',
             'can', 'she', 'been', 'should', 'him', 'not', 'its', 'own',
             'are', 'this', 'shouldn', 'up', 'themselves', 'am', 'were',
             'here', 'wasn', 'then', 'll', 'out', 'being', 'until', 'ma', 'down', 'ain',
             'shan', 'wouldn', 'has', 'each', 'my', 'off', 'her', 'aren', 'myself',
             'herself', 'more', 'as', 'which', 'that', 'into', 'hadn', 'hers',
             'these', 'whom', 'our', 'for', 'about', 'both', 'there', 're', 'during',
             'in', 'haven', 'yours', 'below', 'yourselves', 'do', 'have', 'where',
             'than', 'weren', 'doing', 'isn', 'such', 'mightn', 'from', 'm', 'you',
             'just', 'those', 'me', 'did', 't', 'nor', 'your', 'so', 'o', 'some',
             'he', 'same', 'above', 'be', 'does', 'himself', 'under', 've',
             'couldn', 'on', 's', 'no', 'doesn', 'all', 'too', 'ours', 'at',
             'itself', 'was', 'don', 'didn', 'only', 'a', 'through', 'yourself']
'''


# train_dev_test_clean.PDTB_data  :  remove stopwords and split the sample who has more than one label

stop_words = ['!', ',' ,'.' ,'?' ,'-s' ,'-ly' ,'</s> ', 's']
with open('PDTB_data/train_dev_test.data','rb') as f:
    data = pickle.load(f)
print(data[0][3][1])
word_freq = {}  # to remove rare words
for i in range(3):
    for doc_content in data[i]:
        words = doc_content[1] +doc_content[2]
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
print(len(word_freq))
data_clean = []
for i in range(3):
    data_set = []
    for one in data[i]:
        for n in range(len(one[0])):
            term = []
            agu1 = []
            agu2 = []
            term.append(one[0][n])
            for j in range(len(one[1])):
                if one[1][j] not in stop_words and word_freq[one[1][j]]>1:
                    agu1.append(one[1][j])
            for k in range(len(one[2])):
                if one[2][k] not in stop_words and word_freq[one[2][k]]>1:
                    agu2.append(one[2][k])
            term.append(agu1)
            term.append(agu2)
            data_set.append(term)
    data_clean.append(data_set)
print(data_clean[0][3][1])
print(len(data_clean))
print(len(data_clean[0]))
print(len(data_clean[1]))

with open('PDTB_data/train_dev_test_clean.PDTB_data','wb') as f:
    pickle.dump(data_clean,f,pickle.HIGHEST_PROTOCOL)









