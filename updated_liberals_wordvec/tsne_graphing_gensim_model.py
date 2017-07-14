# import modules & set up logging
import os, sys, gzip, gensim, logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in gzip.open(os.path.join(self.dirname, fname), "rt"):
                yield line.split()

sentences = MySentences('./tweets_dump/') # a memory-friendly iterator
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, iter=1, min_count=10, workers=100)
model.save('./mymodel')


model = gensim.models.Word2Vec.load('./test_model/mymodel')
model['rt']
model.similarity('woman', 'man')
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

model = gensim.models.Word2Vec.load(sys.argv[1])
vectors = model[model.wv.vocab]

tsne = TSNE(n_components=2)
vector_tsne = tsne.fit_transform(vectors)

plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
plt.show()
"""

model = gensim.models.Word2Vec.load(sys.argv[1])
#words = [key for key in model.wv.vocab.keys()]

neighbors = model.most_similar(sys.argv[2], topn=100)
words = [entry[0] for entry in neighbors]

labels = words[0:100]
vectors = [model[labels[i]] for i in range(0, len(labels))]

tsne = TSNE(n_components=2)
data = tsne.fit_transform(vectors)

plt.subplots_adjust(bottom = 0.1)
plt.scatter(data[:, 0]*100, data[:, 1]*100, marker='o')

for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x*100, y*100), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()
