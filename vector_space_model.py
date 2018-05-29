import utils
import json
import math
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from scipy.spatial.distance import cosine as cos_distance
import numpy as np

class VectorSpaceModel:
    def __init__(self,corpus, bm25_parameter=[1.5,0.75,0]):
        self.corpus = corpus
        self.tf = self.build_tf_model()
        self.build_vocabulary()
        self.build_inverted_index()
        self.tfidf = self.build_tfidf_model()
        self.bm25_parameter = bm25_parameter

    def build_vocabulary(self):
        self.vocabulary = self.count_vectorizer.vocabulary_
        self.inverted_vocabulary = []
        for term, _ in sorted(self.vocabulary.items(), key=lambda x: x[1]):
            self.inverted_vocabulary.append(term)
        return self.vocabulary

    def build_inverted_index(self):
        transpose_tf = self.tf.T
        self.inverted_index = []
        for t in sorted(self.vocabulary.values(), key=lambda x: x):
            nonzero_doc = transpose_tf[t].nonzero()
            self.inverted_index.append([(d,transpose_tf[t][d])for d in nonzero_doc[0]])

    def build_tf_model(self):
        self.count_vectorizer = CountVectorizer(analyzer=utils.preprocess)
        self.tf = self.count_vectorizer.fit_transform(self.corpus).toarray()
        return np.array(self.tf)

    def build_tfidf_model(self):
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf = self.tfidf_transformer.fit_transform(self.tf).toarray()
        return np.array(self.tfidf)

    def get_tf_vector(self,doc):
        return self.count_vectorizer.transform(doc).toarray()

    def get_tfidf_vector(self,doc):
        return self.tfidf_transformer.transform(self.get_tf_vector(doc)).toarray()

    def get_top_k_doc(self, query, k=5, method='bm25'):

        doc_similarities={}
        if method == 'bm25':

            k1 = self.bm25_parameter[0]
            b = self.bm25_parameter[1]
            k3 = self.bm25_parameter[2]

            # Get tf vector of query
            query_vector = self.get_tf_vector([query])[0]
            if len(query_vector.nonzero()[0]) == 0:
                return [(0,0)]*k

            N = len(self.corpus)
            Lavg = np.mean([len(d) for d in self.corpus])

            for term in query_vector.nonzero()[0]:

                fqt = query_vector[term]
                ft = np.count_nonzero(self.tf, axis=0)[term]
                # idf
                w1 = np.log(N/ft) # Prevent BM25 from being negative

                # Only calculate the BM25 for the document which contain query token
                for doc,fdt in self.inverted_index[term]:
                    Ld = len(self.corpus[doc])
                    # tf in doc
                    w2 = (k1 + 1) * fdt / (k1 * ((1 - b) + b * Ld / Lavg) + fdt)
                    # tf in query
                    w3 = (k3 + 1) * fqt / (k3 + fqt)
                    w = w1 * w2 * w3
                    doc_similarities[doc] = doc_similarities.get(doc,0) + w

        elif method == 'tfidf':

            # Get tfidf vector of query
            query_vector = self.get_tfidf_vector([query])[0]
            if len(query_vector.nonzero()[0]) == 0:
                return [(0,0)]*k

            for term in query_vector.nonzero()[0]:
                # Only calculate the cosine distance for the document which contain query token
                for doc,_ in self.inverted_index[term]:
                    if doc not in doc_similarities:
                        doc_similarities[doc] = 1 - cos_distance(self.tfidf[doc], query_vector)

        elif method == 'tf':

            # Get tf vector of query
            query_vector = self.get_tf_vector([query])[0]
            if len(query_vector.nonzero()[0]) == 0:
                return [(0,0)]*k

            for term in query_vector.nonzero()[0]:

                # Only calculate the cosine distance for the document which contain query token
                for doc,_ in self.inverted_index[term]:
                    if doc not in doc_similarities:
                        doc_similarities[doc] = 1 - cos_distance(self.tf[doc], query_vector)

        rank = sorted(doc_similarities.items(), key=lambda x: x[1], reverse=True)
        return rank[:k]
