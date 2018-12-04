from gensim import similarities
import numpy as np
import pandas as pd

def get_similarities(training_df, target_df):
    indices = similarities.MatrixSimilarity(training_df.content.tolist())
    target_df["similar_articles"] = target_df.content.apply(lambda col: indices[col])
    similarity_to_corpus = target_df.similar_articles.sum(axis=0)/len(target_df)
    ind = np.argpartition(similarity_to_corpus, -10)[-10:]
    matches = training_df.iloc[ind].copy()

    return(matches)