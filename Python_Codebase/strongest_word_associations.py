"""
Code to make a network out of the shortest N cosine-distances (or, equivalently, the strongest N associations)
between a set of words in a gensim word2vec model.

To use:
Set the filenames for the word2vec model.
Set `my_words` to be a list of your own choosing.
Set `num_top_dists` to be a number or a factor of the length of `my_words.`
Choose between the two methods below to produce distances, and comment-out the other one.
"""

# Import gensim and load the model
import sys, gensim

#model = gensim.models.Word2Vec.load(sys.argv[1])
#model = gensim.models.Word2Vec.load('updated_liberals_wordvec/model_visualize/model_liberal')
model = gensim.models.Word2Vec.load('updated_conservatives_wordvec/model_visualize/model_conservative')

# Set the the words we want to find connections between
my_words = sorted(model.wv.vocab)
my_words = ['god', 'jesus', 'islam', 'nature', 'weather', 'climate', 'freedom', 
            'lgbt', 'obama', 'hilary', 'trump', '#tcot', '#p2b ', 'unamerican']

my_words = [word for word in my_words if word in model] # filter out words not in model
# The number of connections we want: either as a factor of the number of words or a set number
num_top_conns = len(my_words) * 2 


'''
count = 0
max_count = 100#100000
my_words = []

file_read = open('pearson_pvalue_sameword_pairs.txt', 'r')
for line in file_read:
    array = line.strip()[1:-1].split(',')
    word = array[0][1:-1]
    if word.startswith('http'):
        continue
    count += 1
    print("Analyzing Word: ", count, word)
    try:
        assert(word in model.wv.vocab)
    except:
        continue
    my_words.append(word)
    
    if count >= max_count:
        break
num_top_conns = 200
#print(my_words)
'''
#######

# Make a list of all word-to-word distances [each as a tuple of (word1,word2,dist)]
dists=[]

## Method 1 to find distances: use gensim to get the similarity between each word pair
for i1,word1 in enumerate(my_words):
	for i2,word2 in enumerate(my_words):
		if i1>=i2: continue
		cosine_similarity = model.similarity(word1,word2)
		cosine_distance = 1 - cosine_similarity
		dist = (word1, word2, cosine_distance)
		dists.append(dist)
'''
## Or, Method 2 to find distances: use scipy (faster)
from scipy.spatial.distance import pdist,squareform
Matrix = np.array([model[word] for word in my_words])
dist = squareform(pdist(Matrix,'cosine'))
for i1,word1 in enumerate(my_words):
	for i2,word2 in enumerate(my_words):
		if i1>=i2: continue
		cosine_distance = Matrix[i1, i2]
		dist = (word1, word2, cosine_distance)
		dists.append(dist)

######
'''
# Sort the list by ascending distance
dists.sort(key=lambda _tuple_: _tuple_[-1])
# Get the top connections
top_conns = dists[:num_top_conns]

# Make a network
'''
import networkx as nx
graph = nx.Graph()
node_dict = {}
for word1,word2,dist in top_conns:
	weight = 1 - dist # cosine similarity makes more sense for edge weight
	if not word1 in node_dict:
		node1 = graph.add_node(word1)
		node_dict[word1] = node1
	node1 = node_dict[word1]

	if not word2 in node_dict:
		node2 = graph.add_node(word2)
		node_dict[word2] = node2
	node2 = node_dict[word2]

	graph.add_edge(word1, word2, weight=float(weight))

# Write the network
nx.write_graphml(graph, 'strongest_word_network.graphml')

import matplotlib.pyplot as plt
nx.draw(graph)
plt.show()
'''
'''
from pygraphml import GraphMLParser
parser = GraphMLParser()
gx = parser.parse('strongest_word_network.graphml')
gx.show()
'''

import matplotlib
import networkx as nx
#import networkx.drawing
import matplotlib.pyplot as plt
graph = nx.Graph()
#nx.draw_random(graph)
word_dict = {}
labels = {}
node_count = 1

for word1,word2,dist in top_conns:
    weight = 1 - dist # cosine similarity makes more sense for edge weight
    if not word1 in word_dict:
        graph.add_node(node_count, name=word1)
        print(graph.node[node_count])
        word_dict[word1] = node_count
        labels[node_count] = word1
        node_count += 1
    index1 = word_dict[word1]

    if not word2 in word_dict:
        graph.add_node(node_count, name=word2)
        print(graph.node[node_count])
        word_dict[word2] = node_count
        labels[node_count] = word2
        node_count += 1
    index2 = word_dict[word2]

    graph.add_edge(index1, index2, weight=float(weight))


# Write the network
#nx.draw(graph)
pos = nx.spring_layout(graph)
#print("Labels X,Y Locations: ", pos)
nx.draw_networkx(G=graph,labels=labels,font_size=12,node_shape='o',alpha=0.5)
#nx.draw_networkx_labels(G=graph, pos=pos, labels=labels, font_size=16)
nx.write_graphml(graph, 'strongest_word_network.graphml')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(8, 6)
plt.show()

#pos=nx.graphviz_layout(graph, prog="neato")
for  sub_graph in nx.connected_component_subgraphs(graph):
    xy_pos = nx.spring_layout(sub_graph)
    print("SubGraph: ", xy_pos)
    tags=dict((tag, labels[tag]) for tag in xy_pos)
    nx.draw_networkx(G=sub_graph,pos=xy_pos,labels=tags,font_size=12,node_shape='o',alpha=0.5)
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 9)
    plt.show()
    