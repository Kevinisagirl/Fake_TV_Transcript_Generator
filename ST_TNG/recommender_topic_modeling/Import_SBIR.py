from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import re
import os 
import tempfile 
import operator
import codecs
import json
#from pandas.io.json import json_normalize

def split_target_training_test_df(sbir_data):
	author_ids_per_doc = sbir_data.sbir_company.tolist()
	author_freqs = Counter(author_ids_per_doc)
	# let's use the first company as our hold out set
	# should probably do this sooner to better mimic new data coming in
	target_author = author_freqs.most_common(1)[0][0]
	target_df = sbir_data[sbir_data.sbir_company == target_author]
	remaining_df = sbir_data[sbir_data.sbir_company != target_author]

	# randomly split the training_df into training and test df
	np.random.seed(42)
	msk = np.random.rand(len(remaining_df)) < 0.8
	training_df = remaining_df[msk].copy()
	test_df = remaining_df[~msk]

	return target_df, training_df, test_df

def import_sbir(train_file):
	sbir_data = pd.read_pickle(train_file)
	# sbir_data['combined_lines'] = sbir_data.Character + ': ' + sbir_data.Line
	# combine the lines into a single script per episode
	sbir_data = sbir_data.groupby(['Episode'])[['Episode','Line']].transform(lambda x: '\n'.join(x)).drop_duplicates()
	sbir_data.Episode = sbir_data.Episode.apply(lambda x: x.split('\n')[0])
	sbir_data.Line = sbir_data.Line.apply(lambda x: x.replace('\n', ' '))
	# write lines out to csv
	with open('lines.txt', 'w') as f:
		for item in sbir_data.Line.tolist():
			f.write("%s\n" % item)
	return sbir_data

	