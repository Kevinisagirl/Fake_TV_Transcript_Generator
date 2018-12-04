import logging
import sys
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import import_SBIR
import clean_SBIR
import vectorize_SBIR
import models
import similarity_SBIR
import plotting_SBIR

import re
import time
import pandas as pd 
from collections import defaultdict
import matplotlib.pyplot as plt
import pyLDAvis 
import pyLDAvis.gensim
from gensim.models import HdpModel, LdaModel, LdaSeqModel

from collections import Counter

'''
SBIR_TopicModelling: Provides methods to sculpt a gensim model by batching documents and updating an existing model, or creating a new one using the entire corpus. 
'''
class SBIR_TopicModelling():
    def __init__(self):
        logging.basicConfig(filename="../data/SBIR_run.log", filemode='w', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        pd.options.display.max_rows=80
        pd.options.display.max_colwidth=80
    
    '''
    Used for logging
    '''
    def start_timer(self, message):
        self.start_time = time.time()
        logging.info(message)

    '''
    Used for logging
    '''
    def stop_timer(self):
        delt = time.time() - self.start_time
        logging.info("Time taken: " + str(delt) + " seconds.")

    '''
    This returns a tuple containing the model itself, along with the corpus and dictionary, which can be used for vis/updating purposes.
    TODO: I need to load the dict and corpus vec too.
    '''
    def load(self, model_path):
        try:
            model = LdaModel.load(model_path)
            return model 
        except OSError as e:
            logging.info("No pre-trained model found.")
            return None

    '''
    Provide a model with an associated corpus_vector and dictionary to save to a file (file name may be provided, but the default will be the under ../data/saved/sbir+)
    '''
    def save(self, model, corpus_vector, dictionary, dir_path=''):
        # Just a nice default
        if dir_path == '':
            dir_path = "../data/saved/sbir"

        if (model is None) or (corpus_vector is None) or (dictionary is None) :
            logging.warn("Unable to save model. NoneType arguments given.")
            return False
        else:
            model.save(dir_path + ".ldamodel")
            dictionary.save(dir_path + ".dict")
            corpus_vector.to_pickle(dir_path + ".corpusvec")
        return True

    '''
    Call this method only to update this model with more documents. Call run to make a new model from scratch. 
    '''
    def update(self, new_docs_path, existing_model_path):
        
        model = self.load(existing_model_path)

        self.start_timer("Importing new document batch...")
        new_docs = import_SBIR.import_sbir(new_docs_path)
        self.stop_timer()

        self.start_timer("Cleaning new document batch...")
        cleaned_new_docs = clean_SBIR.clean_sbir(new_docs)
        self.stop_timer()

        self.start_timer("Vectorizing new document batch...")
        new_vec, new_dict = vectorize_SBIR.training_vectorize(cleaned_new_docs)
        self.stop_timer()

        self.start_timer("Updating LDA model with new document batch...")
        model.update(corpus=new_vec.content.tolist())
        self.stop_timer()

    '''
    This will create a new model from scratch.
    '''
    def train_corpus(self, corpus_path, save_path=''):

        # Import data
        self.start_timer("Importing SBIR_data...")
        sbir_data = import_SBIR.import_sbir(corpus_path)
        target_data, training_data, test_data = import_SBIR.split_target_training_test_df(sbir_data)
        self.stop_timer()
        
        # Clean data
        self.start_timer("Cleaning SBIR data...")
        holder = clean_SBIR.clean_sbir(training_data.copy(deep=True))
        target_data_holder = clean_SBIR.clean_sbir(target_data)
        test_data_holder = clean_SBIR.clean_sbir(test_data.copy())
        self.stop_timer()
        
        # Vectorize data
        self.start_timer("Vectorizing SBIR data...")
        corpus_vector, dictionary = vectorize_SBIR.training_vectorize(holder.copy())
        self.vec_target_data_holder = vectorize_SBIR.target_vectorize(target_data_holder.copy(), dictionary)
        self.vec_test_data_holder = vectorize_SBIR.test_vectorize(test_data_holder.copy(), dictionary)
        self.stop_timer()

        # Get Similarities
        #matches = similarity_SBIR.get_similarities(vector, vec_target_data_holder)
        #print(matches.head(10))
        
        # Train a model
        trainer = models.ModelTrainer(corpus=corpus_vector, dictionary=dictionary)
        model = trainer.lda_model(num_topics=7)

        self.save(model, corpus_vector, dictionary, save_path)

    '''
    Finds the similarity between 
    TODO: Add document args so I can give variability to the classification test documents
    '''
    def classify(self):
        # Which of the topics do the test documents get placed in?

        # grab the topics for each document
        self.doc_tops = list(self.model[self.vec_test_data_holder.content.tolist()])
        # convert the resulting tuples to dictionaries
        self.doc_tops = list(map(lambda doc: dict((x,y) for x,y in doc), self.doc_tops))
        # add the dictionary as a column to the df
        self.vec_test_data_holder["topics"] = self.doc_tops
        # grab the topic with highest probability for each document
        self.doc_tops = list(map(lambda doc: max(doc, key=doc.get), self.doc_tops))
        # add a column to the df with the topic classification
        self.vec_test_data_holder["topic"] = self.doc_tops

    '''
    Right now this will save two html files to visualize the model.
    1) pyLDAvis - this visualization shows the document topic clusters and the word frequencies for each topic. 
    2) Plotly - This shows the topics over time, and how many documents fit into each topic. 
    '''
    def vis(self, model):
        # pyLDAvis
        # pyLDAvis.enable_notebook()
        p = pyLDAvis.gensim.prepare(self.model, self.vec_holder.content.tolist(), self.dictionary)
        pyLDAvis.save_html(p, 'pyLDAvis_LDA.html')

        # Topic vis
        topics = self.model.print_topics(num_topics=-1)
        topic_counts = dict(Counter(self.doc_tops))

        # pretty print
        print("Topic", " ", "\t", "Count", "\t", "Topic Words")
        for tup in topics:
            print("Topic", tup[0], "\t", topic_counts.get(tup[0]), "\t", " ".join(re.findall('"([^"]*)"', tup[1])))


'''
Entry point
'''
if __name__ == '__main__':
    topic_modeller = SBIR_TopicModelling()
    topic_modeller.train_corpus(corpus_path="../data/data_sets/SBIR-All-Depts-00001.json", save_path='../data/saved/sbir')
    model = topic_modeller.load('../data/saved/sbir.ldamodel')
    
    #Log topic results to run log
    model.print_topics()

    topic_modeller.update("../data/data_sets/SBIR-All-Depts-00000.json", "../data/saved/sbir.ldamodel")
    
    #topic_modeller.classify()
    #topic_modeller.vis()



