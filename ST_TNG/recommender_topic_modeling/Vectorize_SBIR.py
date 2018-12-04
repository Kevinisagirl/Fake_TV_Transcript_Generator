from gensim.models import ldaseqmodel
from gensim.models import LdaSeqModel
from gensim.corpora import bleicorpus
from gensim.models import TfidfModel
import numpy
from gensim.matutils import hellinger
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec


def training_vectorize(holder):
	#Vector uses BOW to store features of the corpus. Uses dictionary
	#for facilitating this operation. This is an important part of the 
	#sequential vectorization

	# split the data
	holder.content = holder['content'].apply(lambda row: row.split())
	# make a dictionary
	dictionary = Dictionary(holder.content.tolist())

	# filter the dictionary
	dictionary.filter_extremes(no_above=0.8, no_below=5)
	dictionary.compactify()

	# transform the data with the dictionary
	holder["content"] = holder["content"].apply(lambda row: dictionary.doc2bow(row))
	
	# transform with tf-idf
	# tfidf = TfidfModel(holder["content"].tolist())
	# holder["content"] = holder["content"].apply(lambda col: tfidf[col])
	return holder, dictionary #, tfidf

def test_vectorize(holder, dictionary):
	holder["content"] = holder["content"].apply(lambda row: dictionary.doc2bow(row.split()))
	#transform with tf-idf
	# holder["content"] = holder["content"].apply(lambda col: tfidf[col])
	return holder

def target_vectorize(holder, dictionary):
	holder["content"] = holder["content"].apply(lambda row: dictionary.doc2bow(row.split())) 
	return holder

