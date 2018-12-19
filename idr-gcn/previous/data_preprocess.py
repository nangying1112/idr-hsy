import pickle as pkl
from previous.lstm_score_utils import loadWord2Vec


# build vocab and word2id
word_freq = {}
word_set = set()
with open('PDTB_data/train_dev_test.data','rb') as f:
    data = pkl.load(f)
print(data[0][0:10])
for i in range(len(data)):
    for agus in data[i]:
        for word in agus[1]+agus[2]:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
vocab = list(word_set)
vocab = ['UNK','PAD']+vocab
vocab_size = len(vocab)
word2id = dict([(x,y) for (y,x) in enumerate(vocab)])

print('vocab_size:',vocab_size)
with open('PDTB_data/word2id','wb') as f:
    pkl.dump(word2id,f,pkl.HIGHEST_PROTOCOL)

# load glove vectors
word_vector_file = 'PDTB_data/glove.6B.300d.txt'
word_vector_map = loadWord2Vec(word_vector_file)

embed = []
for word in vocab:
    if word in word_vector_map:
        embed.append(word_vector_map[word])
    else:
        embed.append([0. for i in range(300)])

print('embed_size:',len(embed))

with open('PDTB_data/embed','wb') as f:
    pkl.dump(embed,f,pkl.HIGHEST_PROTOCOL)

