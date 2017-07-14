# import modules & set up logging
import os, gzip, gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in gzip.open(os.path.join(self.dirname, fname), "rt"):
                yield line.split()

sentences = MySentences('./dataset_liberal/') # a memory-friendly iterator
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=10, workers=100)
model.save('./model_visualize/model_liberal')
