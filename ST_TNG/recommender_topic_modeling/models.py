from gensim.models import HdpModel, LdaModel, LdaSeqModel
import pandas as pd
import pickle as pickle
import logging
import matplotlib.pyplot as plt

'''
Just a wrapper for the gensim function. The intention is that this will be more scalable with different models and data sets.
Each instance of the model trainer must contain the corpus and dictionary. 
'''
class ModelTrainer():
    def __init__(self, corpus, dictionary, test, num_topics=0):
        self.corpus = corpus
        self.dictionary = dictionary
        self.model = None
        self.test = test
        self.perplexity = 100000000
        self.num_topics = num_topics

    def lda_model(self):
        self.model = LdaModel(corpus=self.corpus['content'].tolist(), id2word=self.dictionary, num_topics=self.num_topics) # try the distributed parameter
        return self.model
    
    def best_lda_model(self):
        tuple_list = []
        for n in range(3,50):
            test_model = LdaModel(corpus=self.corpus['content'].tolist(), id2word=self.dictionary, num_topics=n) # try the distributed parameter
            tperplexity = test_model.log_perplexity(self.test.content.tolist(), total_docs=None)
            tuple_list.append((n,tperplexity))
            # if tperplexity < self.perplexity:
            #     self.model = test_model
            #     self.perplexity = tperplexity
            #     print("New lower log_perplexity with",n,"topics")
            if n%10 == 0:
                print(n)
        plt.scatter(*zip(*tuple_list))
        plt.show()
        # return self.model

