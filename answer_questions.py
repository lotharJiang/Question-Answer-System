import csv
import gensim
from gensim.models import Word2Vec
import utils
import json
import math
import numpy as np
import spacy
import re
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, sent_tokenize
import sys, getopt
from sklearn.metrics import classification_report as c_report
from sklearn.utils import shuffle
from vector_space_model import VectorSpaceModel
from nltk.corpus import stopwords


def main(argv):
    method = 'bm25'
    k1 = 1.5
    b = 0.5
    k3 = 0
    try:
        opts, args = getopt.getopt(argv, "m:b:", ["method=", "bm25="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-m", "--method"):
            if arg == '1':
                method = 'bm25'
            elif arg == '2':
                method = 'tfidf'
            elif arg == '3':
                method = 'tf'
        elif opt in ("-b", "--bm25"):
            parameters = arg.split(',')
            k1 = float(parameters[0].strip())
            b = float(parameters[1].strip())
            k3 = float(parameters[2].strip())

    doc_path = 'data/documents.json'
    training_path = 'data/training.json'
    develop_path = 'data/devel.json'
    testing_path = 'data/testing.json'
    doc_set = utils.get_dataset(doc_path)
    dataset = utils.get_dataset(testing_path)

    questions = []
    for query in dataset:
        questions.append(query['question'])

    questions_classes = utils.question_classify(questions)
    nlp = spacy.load('en_core_web_sm')

    results = []
    for i in range(len(dataset)):
        query = dataset[i]
        candidates = []
        should_have_candidate = False

        # If the question class is not 'other', there should be candidate answer entities in retrieved sentences
        if questions_classes[i] != 'other':
            should_have_candidate = True

        question = query['question']
        answer = None

        # Build vector space model to retrieve the top 3 relevant paragraphs
        para_corpus = doc_set[query['docid']]['text']
        para_vsm = VectorSpaceModel(para_corpus, [k1, b, k3])
        match_paras = para_vsm.get_top_k_doc(question, k=3, method=method)

        # Build vector space model to retrieve the top 3 relevant sentences
        sent_corpus = []
        for para in match_paras:
            sent_corpus += sent_tokenize(doc_set[query['docid']]['text'][para[0]])
        sent_vsm = VectorSpaceModel(sent_corpus, [k1, b, k3])
        match_sents = sent_vsm.get_top_k_doc(question, k=3, method=method)

        if should_have_candidate:
            # Find entities with expected types in candidate sentences
            for sent in match_sents:
                match_sent = sent_corpus[sent[0]]
                entities = utils.entity_recognition(match_sent, nlp)

                if questions_classes[i] == 'hum':
                    for e in entities:
                        # 'Human' questions mostly start by 'Who', so they may also expect organization as answer
                        if (e[1] == 'PERSON' or e[1] == 'ORG') and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'date':
                    for e in entities:
                        if (e[1] == 'DATE' or e[1] == 'TIME') and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'money':
                    for e in entities:
                        if e[1] == 'MONEY' and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'gpe':
                    for e in entities:
                        if e[1] == 'GPE' and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'loc':
                    for e in entities:
                        if (e[1] == 'LOC' or e[1] == 'FAC') and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'num':
                    for e in entities:
                        if (e[1] == 'CARDINAL' or e[1] == 'QUANTITY') and e[
                            0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'ord':
                    for e in entities:
                        if (e[1] == 'ORDINAL') and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'percent':
                    for e in entities:
                        if (e[1] == 'PERCENT') and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'event':
                    for e in entities:
                        if (e[1] == 'EVENT') and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'art':
                    for e in entities:
                        if (e[1] == 'WORK_OF_ART') and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'lang':
                    for e in entities:
                        if (e[1] == 'LANGUAGE') and e[0].lower() not in question.lower():
                            candidates.append(e)
                elif questions_classes[i] == 'org':
                    for e in entities:
                        if (e[1] == 'ORG') and e[0].lower() not in question.lower():
                            candidates.append(e)

                if len(candidates) > 0:
                    answer = ''
                    for candidate in candidates:
                        answer += ' ' + candidate[0]
                    break

        # If question is tagged with no expected answer type.
        # Or there are no expected entities in top 3 relevant sentences.
        if answer is None:

            # Consider the answer is included in the top 1 relevant sentence
            match_sent = sent_corpus[match_sents[0][0]]
            non_candidates = utils.preprocess(question)
            punctuation = r'[\s+\.\!\/_,$%^*()+:;\-\[\]\"\'`]+|[+——！，。？、~@#￥%……&*（）]+'

            # Remove the tokens in question, punctuations, replicate tokens and tokens which are not noun or adjective.
            rough_answer = utils.preprocess(match_sent, True)
            for word in rough_answer[:]:
                if word[0] in non_candidates or word[1] in candidates or re.match(punctuation, word[0]) or not (
                        word[2].startswith('J') or word[2].startswith('N')):
                    continue
                candidates.append(word[1])
            answer = ' '.join(w for w in candidates).strip()
        answer = answer.lower().strip()
        print (question+'\n'+questions_classes[i]+'\n'+answer+'\n')
        results.append([query['id'], answer])

    # Save all results
    utils.save_csv("sphinx_bane.csv", ["id", "answer"], results)

if __name__ == "__main__":
    main(sys.argv[1:])
