def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
            (With help from William. Thank you!)

    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.

    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(
        base_embed, other_embed, words=words)

    # get the embedding matrices
    base_vecs = in_base_embed.wv.syn0norm
    other_vecs = in_other_embed.wv.syn0norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.syn0norm = other_embed.wv.syn0 = (other_embed.wv.syn0norm).dot(ortho)
    return other_embed


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
            -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
            -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    
    if words:
        common_vocab &= set(words)

    print("Liberal, Conservative = ", (len(vocab_m1), len(vocab_m2)))

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(
        key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count, reverse=True)

    print("Common Vocabulary = ", len(common_vocab))
    print(common_vocab[0:10])
    
    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = m.wv.syn0#norm
        
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.syn0norm = m.wv.syn0 = new_arr

        print("Vector Shape = ", (new_arr.shape, m.wv.syn0.shape))

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
 
        m.index2word = common_vocab       
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, 
                    count=old_vocab_obj.count)
        
        m.wv.vocab = new_vocab
        print("Index2Word, Vocab = ", (len(m.index2word), len(m.wv.vocab)))

    return (m1, m2)


import sys, gensim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
np.random.seed(7)
#first_word2vec_model = Word2Vec.load('updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
#second_word2vec_model = Word2Vec.load('updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])

first_word2vec_model = gensim.models.Word2Vec.load(
        'updated_liberals_wordvec/model_visualize/model_liberal')#(sys.argv[0])
second_word2vec_model = gensim.models.Word2Vec.load(
        'updated_conservatives_wordvec/model_visualize/model_conservative')#(sys.argv[1])

second_aligned_model = smart_procrustes_align_gensim(first_word2vec_model, second_word2vec_model)
first_aligned_model = smart_procrustes_align_gensim(second_aligned_model, first_word2vec_model)
# Get the vocab for each model
vocab_m1 = set(first_aligned_model.wv.vocab.keys())
vocab_m2 = set(second_aligned_model.wv.vocab.keys())
# Find the common vocabulary
common_vocab = vocab_m1 & vocab_m2

assert(len(common_vocab) == len(vocab_m1) and len(common_vocab) == len(vocab_m2))
print("Words in Intersection: ", len(common_vocab), len(vocab_m1), len(vocab_m2))
#print(common_vocab)

#print("Liberal = ", first_word2vec_model.wv.most_similar(word, topn=10))
#print("Conservative = ", second_word2vec_model.wv.most_similar(word, topn=10))
#print("Aligned = ", second_aligned_model.wv.most_similar(word, topn=10))

'''
word = sys.argv[1]#'unamerican'
topcount = int(sys.argv[2])#100

first_neighbors = first_aligned_model.most_similar(word, topn=topcount)
first_labels = [entry[0] for entry in first_neighbors if entry[0] in first_aligned_model.wv.vocab]
first_labels.insert(0, word)
print("liberal model: ", word, first_labels)
first_vectors = [first_aligned_model[label] for label in first_labels]# if label in first_aligned_model.wv.vocab]

second_neighbors = second_aligned_model.most_similar(word, topn=topcount)
second_labels = [entry[0] for entry in second_neighbors if entry[0] in second_aligned_model.wv.vocab]
second_labels.insert(0, word)
print("conservative model: ", word, second_labels)
second_vectors = [second_aligned_model[label] for label in second_labels]# if label in second_aligned_model.wv.vocab]
'''
#exit()

first_labels = sorted(list(vocab_m1))[0:100]
second_labels = sorted(list(vocab_m2))[0:100]
labels = first_labels + second_labels

first_vectors = [first_aligned_model[label] for label in first_labels]
second_vectors = [second_aligned_model[label] for label in second_labels]
vectors = first_vectors + second_vectors

tsne = TSNE(n_components=2)
data = tsne.fit_transform(vectors)

scale = 10000
plt.subplots_adjust(bottom = 0.1)
plt.scatter(data[:, 0]*scale, data[:, 1]*scale, marker='o')

color = 'blue'
for label, x, y in zip(first_labels, data[0:len(first_labels), 0], data[0:len(first_labels), 1]):
            
    plt.annotate(
        label,
        xy=(x*scale, y*scale), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

color = 'red'
for label, x, y in zip(second_labels, data[len(first_labels):, 0], data[len(first_labels):, 1]):
    plt.annotate(
        label,
        xy=(x*scale, y*scale), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

fig = plt.gcf()
fig.set_size_inches(16, 12)
#plt.show()
plt.draw()
fig.savefig('visualize_two_models.png', dpi=100)
#fig.savefig('visualize_two_models.svg', format='svg', dpi=1200)

'''
labels = first_labels + second_labels
vectors = first_vectors + second_vectors
assert(len(labels) == len(vectors))
print("Labels, Vectors: ", len(labels), len(vectors))

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
'''
