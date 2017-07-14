import os, sys, re, gzip
from os import walk

from ttp import ttp
import matplotlib.pyplot as plt
import string, json, operator, math 
from collections import Counter

from nltk import bigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd, vincent
from vincent import AxisProperties, PropertySet, ValueRef

import time
from multiprocessing import Process, Lock, Manager

#import gensim

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']
#print "Stop Words: ", stop
 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
emoticons_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

 
regex_str = [
	emoticons_str,
	#r':=;', # Eye
	#r'oO\-?', # Nose
    	#r'D\)\]\(\]/\\OpP', # Mouth
    	r'<[^>]+>', # HTML tags
    	r'(?:@[\w_]+)', # @-mentions
    	r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    	r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    	r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    	r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    	r'(?:[\w_]+)', # other words
    	r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
	return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
	tokens = tokenize(s)
	if lowercase:
		tokens = [token if emoticons_re.search(token) else token.lower() for token in tokens]
	return tokens

 
def word_in_text(word, text):
	word = word.lower()
	text = text.lower()
	match = re.search(word, text)
	if match:
		return True
	return False


def extract_link(text):
	regex = r'https?|ftp?://?[^\s<>"]?|www\.?[^\s<>"]?|[^\s<>]?\.com?|\.gov?|\.org?|\.info?'
	match = re.search(regex, text)
	if match:
		return text
	#return match.group()
	return ''

"""
tweet = 'RT @marcobonzanini: just an example! x:D https://example.com #NLP'
print(preprocess(tweet))
print(extract_link("https://t.coxyz.co/abc=1&d=%20"))
print(extract_link("https://t.co/Ty5011oJjU"))
print(extract_link("www.google.com"))
print(extract_link("ftp://abc/kljk"))
print(extract_link("45.wh.gov/321S"))
# ['RT', '@marcobonzanini', ':', '!', ':D', 'http://example.com', '#NLP']
exit()
"""

liberal_tweet_list = ["p2", "liberals", "resist", "uniteblue", "democrats", "freespeech", "left", "p2b", "p21", "topprog", "votedem"]
liberal_tweet_dic = {}
for i in range(0, len(liberal_tweet_list)):
	liberal_tweet_dic[liberal_tweet_list[i]] = True


#model = gensim.models.Word2Vec.load('./test_model_2k/mymodel')


def main(link, file_write):

	#Reading Tweets
	
	print("Reading Tweets: ", link)
	tweets_data_path = link#sys.argv[1]#./twitter_data.txt'
	tweets_file = gzip.open(tweets_data_path, "rt")

	tweets_data = []
	for line in tweets_file:
		try:
			tweet = json.loads(line)
		except:
			continue
		if tweet.has_key(u'delete') == True or \
			not "en" in tweet["lang"]:
			continue
		#print "Sample Tweet: ", tweet['text']

		if tweet.has_key(u'text') == False:
			continue
	
		line = tweet['text']
		if len(line) == 0:
			continue
		tokenized_tweet = preprocess(line, True)
		#print "Tokenized: ", tokenized_tweet

		liberal_tweet = False
		"""
		entities = tweet["entities"]
		if len(entities["hashtags"]) > 0:
			print("hashtags list: ", type(entities["hashtags"]), entities["hashtags"])
			for entry in entities["hashtags"]:
				print("Entry: ", entry, entry[u'text'])
				if liberal_tweet_dic.has_key(entry[u'text']):
					liberal_tweet = True
					break
		"""
		for term in tokenized_tweet:
			if term.strip().startswith("#"):
				if liberal_tweet_dic.has_key(term.strip()[1:]):
					liberal_tweet = True
					break	
		
		if not liberal_tweet:
			continue


		line = " ".join(tokenized_tweet) + "\n"
		tweets_data.append(line)

	print("Total Tweets: " + str(len(tweets_data)))
	for i in range(0, len(tweets_data)):
		#print(tweets_data[i])
		file_write.write(tweets_data[i].encode("UTF-8"))

	return 
	"""
	#Analyzing Tweets by Country
	print 'Analyzing tweets by country\n'
	tweets_by_country = tweets['country'].value_counts()
	fig, ax = plt.subplots(figsize=(8,8))
	ax.tick_params(axis='x')#, labelsize=15)
	ax.tick_params(axis='y')#, labelsize=10)
	ax.set_xlabel('Countries')#, fontsize=15)
	ax.set_ylabel('Number of tweets')# , fontsize=15)
	ax.set_title('Top 5 countries')#, fontsize=15, fontweight='bold')
	tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')
	plt.savefig('tweet_by_country.png', format='png', bbox_inches='tight')
	"""



def input_function(start, end, process_no):
	process_status[process_no] = "running"
	file_write = gzip.open("./dataset_liberal/tweets_" + str(process_no + 1) + ".gz", "wt")

	for i in range(start, end):
		try:
			main(original_file_list[i], file_write)
		except:
			continue
	
	file_write.close()
	#append_lock.acquire()
	#append_lock.release()
	process_status[process_no] = "complete"



input_dir = sys.argv[1]
original_file_list = []
for (dirpath, dirnames, filenames) in walk(input_dir):
	for i in range(0, len(filenames)):
		if "onepercent" in filenames[i] and ".gz" in filenames[i]:
			original_file_list.append(input_dir + "/" + filenames[i])
	break

original_file_list = sorted(original_file_list)[0:]
print("Original File Count: ", len(original_file_list))


process_count = 200
process_array = []

append_lock = Lock()
manager = Manager()
process_status = manager.list()

division = int(math.ceil(1.0 * len(original_file_list) / process_count))

for i in range(0, process_count):
        start = i * division
        end = min((i + 1) * division, len(original_file_list))
        print("Multiprocess Start == ", (start, end, i))
	process = Process(target=input_function, args=(start, end, i))
        process_status.append("waiting")
       	process.start()
	process_array.append(process)


for i in range(0, len(process_array)):
        process_array[i].join()


count = 0
while(count < process_count):
	if process_status[count] == "complete":
		count += 1
	else:
		time.sleep(5)		

print("Done")
#---------------------------------------------------------------------------------------------------
"""
fig = plt.figure()
plt.bar(range(len(word_freq)), word_freq.values(), align='center')
plt.xticks(range(len(word_freq)), word_freq.keys())
fig.savefig('term_histogram.png', dpi=fig.dpi)
plt.show()
"""
