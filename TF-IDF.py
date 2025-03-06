from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def ToTFIDF(text,desired_word):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    features= vectorizer.get_feature_names_out()
    if desired_word in features:
        word_index = np.where(features == desired_word)[0][0]  
        word_scores = X[:, word_index].toarray().flatten()
        print('word_scores:',word_scores)
    return X

a=ToTFIDF(text=["this is text, not file or filename file"],desired_word="file")
