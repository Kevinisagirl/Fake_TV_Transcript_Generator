import re 
import pandas as pd 
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import numpy as np

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

stopwords = []


def clean_sbir(holder):
	# delete duplicate rows based off abstracts
	holder = holder.drop_duplicates('content')

	#Save old abstract col
	holder["original_abstract"] = holder.content

	#Clean body text strings
	regex = re.compile('[^A-Za-z]') #Stick with alphabetical for now
	holder.content = holder.content.str.replace(regex, ' ')

	#holder.content = holder.content.str.replace('\d+','') #Strip numbers
	holder.content = holder.content.str.lower()	#lowercase
	holder.content = holder.content.astype(str)  # convert everything to a string

	#Load stopwords, strip from body text
	with open('../data/list_stopwords.txt') as f:
		stopwords = f.read().splitlines()
	holder.content = holder.content.apply(lambda x: ' '.join([get_lemma2(word) for word in x.split() if word not in stopwords]))
	
	#Fill all empty strings in the entire dataframe with NaN
	holder = holder.replace('', np.nan)
	# holder['sbir_award_year'] = holder['sbir_award_year'].replace('0', np.nan)

	# Drop NaN so that the model doesn't expect the extra documents
	# Consider testing with inplace=true to avoid the slice copy
	# holder = holder.dropna(subset=['sbir_award_year', 'content'])

	# Convert the date column to a pandas datetime format
	# holder.sbir_award_year = pd.to_datetime(holder.sbir_award_year, format="%Y").apply(lambda x: x.year)
	# Sort the dataframe by date
	# holder = holder.sort_values(by=['sbir_award_year'])
	
	return holder
	
	

	