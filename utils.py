from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import spacy
import json
import re
from sklearn.utils import shuffle
from collections import Counter
import string
import csv
import tensorflow as tf
from spacy.symbols import nsubj, VERB
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from question_classifier import QuestionClassifier

def read_csv(filename):
    csvfile = open(filename, "r")
    read = csv.reader(csvfile)
    data = []
    for line in read:
        data.append(line)
    csvfile.close()
    return data


def save_csv(filename, header, data):
    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)


def get_dataset(path):
    with open(path) as f:
        dataset_str = f.read()
    dataset = json.loads(dataset_str)
    return dataset


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def preprocess(text, with_raw_info=False):
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    corpus = []

    for sent in sent_tokenize(text):

        # Tokenization & POS Tagging
        tagged_sentence = pos_tag(word_tokenize(sent))

        for word, pos in tagged_sentence:

            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN

            # Lemmatization
            preprocessed_token = lemmatizer.lemmatize(word.lower(), pos=wordnet_pos)

            # Removal of Stop Words
            if preprocessed_token not in stopWords:
                # Return the processed token with raw token and its POS tags
                if with_raw_info:
                    corpus.append((preprocessed_token, word, pos))

                else:
                    corpus.append(preprocessed_token)
    return corpus

# NER Tagging using Spacy
def entity_recognition(text, nlp):
    entity_names = []
    doc = nlp(text)
    for ent in doc.ents:
        entity_names.append((ent.text, ent.label_, (ent.start, ent.end)))
    return entity_names

# Tag the expected entity in answer
# It is used for the generation of training set for CNN question classifier
def ner_tag_answer(para, ans, nlp):
    entities = entity_recognition(para, nlp)
    s, e = find_raw_answer_position(para, ans)
    if s is None or e is None:
        return 'Other'
    possible_entity = []
    for ent in entities:
        if max(s, ent[2][0]) < min(e, ent[2][1]):
            possible_entity.append(ent[1])
    if len(possible_entity) == 0:
        return 'Other'
    return possible_entity[0]

# Normalize sentences for question classifier
def normalize_sentence(sents):
    sents = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sents)
    sents = re.sub(r"\'s", " \'s", sents)
    sents = re.sub(r"\'ve", " \'ve", sents)
    sents = re.sub(r"n\'t", " n\'t", sents)
    sents = re.sub(r"\'re", " \'re", sents)
    sents = re.sub(r"\'d", " \'d", sents)
    sents = re.sub(r"\'ll", " \'ll", sents)
    sents = re.sub(r",", " , ", sents)
    sents = re.sub(r"!", " ! ", sents)
    sents = re.sub(r"\(", " \( ", sents)
    sents = re.sub(r"\)", " \) ", sents)
    sents = re.sub(r"\?", " \? ", sents)
    sents = re.sub(r"\s{2,}", " ", sents)
    return sents.strip().lower()

# Acquire training dataset for question classifier
def get_trainingset(question_classes, data_path_prefix):
    x_train = []
    y_train = []
    for c in range(len(question_classes)):
        label = [0] * len(question_classes)
        label[c] = 1
        data = list(open(data_path_prefix + question_classes[c] + '.txt', "r").readlines())
        data = [ques.strip() for ques in data]
        x_train += data
        y_train += [label] * len(data)
    return (x_train, y_train)


# Build vocabulary
def build_vocabulary(sents):
    word_count = 0
    vocabulary = dict()
    for sent in sents:
        for word in sent:
            if word not in vocabulary:
                vocabulary[word] = word_count
                word_count += 1
    return vocabulary

# Preprocess questions by padding
def processed_sentences(sents, max_length=100, vocabulary=None):
    sents = [s.strip() for s in sents]
    sents = [normalize_sentence(sent) for sent in sents]
    sents = [s.split(" ") for s in sents]
    processed_sents = []
    pad_token = "<PAD/>"
    # Training
    if vocabulary is None:
        max_length = max([len(sent) for sent in sents])
        processed_sents = [sent + [pad_token] * (max_length - len(sent)) for sent in sents]

    else:
        for sent in sents:
            for word in sent[:]:
                if word not in vocabulary:
                    # Omit the words not in vocabulary
                    sent.remove(word)

            # Omit the words over length restriction
            if len(sent) > max_length:
                sent = sent[:max_length]
            processed_sents.append(sent + [pad_token] * (max_length - len(sent)))
    return processed_sents


# Get training dataset and process data
def get_training_dataset_info():
    # Totally 13 question classes
    question_classes = ['num', 'percent', 'money', 'ord', 'date', 'gpe', 'loc', 'hum', 'org', 'event', 'art', 'lang', 'other']
    x_train, y_train = get_trainingset(question_classes, 'data/qustion_classifier_training/train_')
    processed_x_train = processed_sentences(x_train)
    vocabulary = build_vocabulary(processed_x_train)
    processed_x_train = np.array([[vocabulary[word] for word in sent] for sent in processed_x_train])
    y_train = np.array(y_train)
    return (processed_x_train, y_train, vocabulary, question_classes)

# Generates a batch iterator for a dataset.
def batch_generater(data, batch_size, num_epochs, to_shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for _ in range(num_epochs):
        if to_shuffle:
            shuffled_data = shuffle(data)
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def question_classify(dataset):
    x_train, _ , vocabulary, classes = get_training_dataset_info()
    max_length = max([len(sent) for sent in x_train])
    x_test = processed_sentences(dataset, max_length, vocabulary)
    x_test = np.array([[vocabulary[word] for word in sent] for sent in x_test])

    checkpoint_file = tf.train.latest_checkpoint('classifier_model/checkpoints/')
    tf.reset_default_graph()
    sess = tf.Session()
    with sess.as_default():
        classifier = QuestionClassifier(
            input_size=x_train.shape[1],
            n_classes=len(classes),
            vocabulary_size=len(vocabulary))
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)
        batches = batch_generater(x_test, 100, 1, to_shuffle=False)

        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(classifier.predictions, {classifier.question: x_test_batch, classifier.dropout_rate: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

    y_predict = []
    for predict in all_predictions:
        y_predict.append(classes[int(predict)])

    return y_predict

# Following functions are used on the evaluation
def evaluate_CNN_question_classfier():
    loss = np.load('loss_dev.npy')
    accuracy = np.load('accuracy_dev.npy')
    precision = np.load('precision_dev.npy')
    recall = np.load('recall_dev.npy')
    f1score = np.load('f1score_dev.npy')

    x = [i for i in range(len(loss)) if i % 10 == 0]
    xtick = [(str((x_) * 100)) for x_ in x]
    best_checkpoint = np.argmax(accuracy)
    plt.plot(loss, 'r', linestyle='-', label='Cross Entropy Loss')
    plt.plot(accuracy, 'g', linestyle='-', label='Accuracy')
    plt.plot(precision, 'y', linestyle='-', label='Precision')
    plt.plot(recall, 'b', linestyle='-', label='Recall')
    plt.plot(f1score, 'brown', linestyle='-', label='F1-Score')
    plt.axhline(y=accuracy[best_checkpoint], color='grey', ls="--", linewidth=1)
    plt.text(len(loss), accuracy[best_checkpoint], round(accuracy[best_checkpoint], 4), ha='center', va='bottom')
    plt.xticks(x, xtick)
    plt.xlabel('Training Steps')
    plt.legend()
    plt.show()


def evaluate_information_retrieval():
    bm25 = [0.71, 0.68, 0.69]
    now1 = [0.73, 0.72, 0.72]
    nonnegative = [0.77, 0.75, 0.75]
    originalidf = [0.77, 0.75, 0.75]
    tfidf_cos = [0.73, 0.70, 0.71]
    tf_cos = [0.55, 0.51, 0.52]

    plt.barh([1], bm25[-1], 0.5)
    plt.barh([2], now1[-1], 0.5)
    plt.barh([3], nonnegative[-1], 0.5)
    plt.barh([4], originalidf[-1], 0.5)
    plt.barh([5], tfidf_cos[-1], 0.5)
    plt.barh([6], tf_cos[-1], 0.5)
    plt.text(bm25[-1] + 0.035, 1 - 0.1, bm25[-1], ha='center', va='bottom')
    plt.text(now1[-1] + 0.035, 2 - 0.1, now1[-1], ha='center', va='bottom')
    plt.text(nonnegative[-1] + 0.035, 3 - 0.1, nonnegative[-1], ha='center', va='bottom')
    plt.text(originalidf[-1] + 0.035, 4 - 0.1, originalidf[-1], ha='center', va='bottom')
    plt.text(tfidf_cos[-1] + 0.035, 5 - 0.1, tfidf_cos[-1], ha='center', va='bottom')
    plt.text(tf_cos[-1] + 0.035, 6 - 0.1, tf_cos[-1], ha='center', va='bottom')
    plt.xlim(0, 1)
    plt.yticks([1, 2, 3, 4, 5, 6], ['BM25', 'BM25\n(remove idf)', 'Max(BM25,0)', 'BM25\n(idf = log(N/ft))',
                                    'Cosine diatance\n(Tfidf vectors)', 'Cosine diatance\n(Tf vectors)'])
    plt.xlabel('f1-score')
    plt.show()


def get_f1_score(predictions, true_answers, average='macro'):
    TP = 0
    FP = 0
    FN = 0
    precision = 0
    recall = 0
    f1 = 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        answer = true_answers[i]
        prediction_tokens = prediction.split()
        answer_tokens = answer.split()
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if average == 'micro':
            TP += num_same
            FP += len(prediction_tokens) - num_same
            FN += len(answer_tokens) - num_same
        if average == 'macro':
            TP = num_same
            FP = len(prediction_tokens) - num_same
            FN = len(answer_tokens) - num_same
            if num_same == 0:
                continue
            precision += 1.0 * TP / (TP + FP)
            recall += 1.0 * TP / (TP + FN)
            f1 += (2 * (1.0 * TP / (TP + FP)) * (1.0 * TP / (TP + FN))) / (
                        (1.0 * TP / (TP + FP)) + (1.0 * TP / (TP + FN)))
    if average == 'micro':
        precision = 1.0 * TP / (TP + FP)
        recall = 1.0 * TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)
    if average == 'macro':
        precision /= len(predictions)
        recall /= len(predictions)
        f1 /= len(predictions)

    return precision, recall, f1


# Following functions are previously developed for different design of the system.
# However, these methods are not adopted in the final version.
# Including: relationship extraction, dependency parsing, feature processing for attention-based neural network, etc.

def noun_chunk_extraction(text, nlp):
    chunks = []
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        chunks.append((chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text))

    return chunks


def get_root_verb(text, nlp):
    # chunks = []
    doc = nlp(text)
    for token in doc:
        if token.dep_ == 'ROOT':
            return (token.text, token.dep_, token.head.pos_, [child for child in token.children])

    return doc


def relations_extraction(text, nlp, types):
    doc = nlp(text)
    # merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    for span in spans:
        span.merge()
    relations = []
    for entity in filter(lambda w: w.ent_type_ in types, doc):
        if entity.dep_ in ('attr', 'dobj') or entity.dep_ in ('attr', 'iobj'):
            subject = [w for w in entity.head.lefts if w.dep_ == 'nsubj' or w.dep_ == 'nsubjpass']
            if subject:
                subject = subject[0]
                relations.append((subject, entity.head, entity, 2))
        elif entity.dep_ in ('attr', 'nsubj'):
            object = [w for w in entity.head.rights if w.dep_ == 'dobj' or w.dep_ == 'iobj']
            if object:
                object = object[0]
                relations.append((entity, entity.head, object, 0))
        elif entity.dep_ == 'pobj':  # and money.head.dep_ == 'prep':
            verb = entity
            count = 0
            while verb.pos != VERB:
                verb = verb.head
                count += 1
                if count > 5:
                    verb = entity.head.head
                    break

            subject = [w for w in verb.lefts if w.dep_ == 'nsubj' or w.dep_ == 'nsubjpass']
            if subject:
                subject = subject[0]
            object = [w for w in verb.rights if w.dep_ == 'dobj' or w.dep_ == 'iobj']
            if object:
                object = object[0]
            relations.append((subject, verb, object, entity, 3))
        elif entity.dep_ == 'compound':
            verb = entity
            count = 0
            while verb.pos != VERB:
                verb = verb.head
                count += 1
                if count > 5:
                    verb = entity.head.head
                    break
            subject = [w for w in verb.lefts if w.dep_ == 'nsubj' or w.dep_ == 'nsubjpass']
            if subject:
                subject = subject[0]
            object = [w for w in verb.rights if w.dep_ == 'dobj' or w.dep_ == 'iobj']
            if object:
                object = object[0]
            relations.append((subject, verb, object, entity, 3))
    return relations


def build_word2vec_and_pos2vec(doc_path='data/documents.json'):
    lemmatizer = WordNetLemmatizer()
    doc_set = get_dataset(doc_path)
    word_corpus = []
    pos_corpus = []
    for doc in doc_set:
        for para in doc['text']:
            for sent in sent_tokenize(para):
                tagged_sentence = pos_tag(word_tokenize(sent))
                sent_words = []
                sent_pos = []
                for word, pos in tagged_sentence:
                    wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
                    sent_words.append(lemmatizer.lemmatize(word.lower(), pos=wordnet_pos))
                    sent_pos.append(pos)
                word_corpus.append(sent_words)
                pos_corpus.append(sent_pos)
    word_model = Word2Vec(word_corpus, min_count=1)
    word_model.save('data/word2vec.pkl')
    pos_model = Word2Vec(pos_corpus, size=15)
    pos_model.save('data/pos2vec.pkl')
    return word_model, pos_model


def create_metadata(doc_path='./data/documents.json', training_path='./data/training.json',
                    develop_path='./data/devel.json'):
    doc_set = get_dataset(doc_path)
    i_para = 0
    meta = {}
    para_lengths = {}
    for doc in doc_set:
        for para in doc['text']:
            para_lengths[i_para] = sum([len(word_tokenize(sent)) for sent in sent_tokenize(para)])
            i_para += 1
    meta['para_lengths'] = para_lengths
    dataset = get_dataset(training_path) + get_dataset(develop_path)
    i_ques = 0
    question_lengths = {}
    answer_lengths = {}
    for question in dataset:
        question_lengths[i_ques] = len(word_tokenize(question['question']))
        answer_lengths[i_ques] = len(word_tokenize(question['text']))
        i_ques += 1
    meta['question_lengths'] = question_lengths
    meta['answer_lengths'] = answer_lengths
    with open("./data/metadata.json", "w") as f:
        json.dump(meta, f)
    f.close()


# Return the lemmatized tokens in text with their POS tags, but never remove stop words.
def tokenize_and_pos(text):
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for sent in sent_tokenize(text):
        tagged_sentence = pos_tag(word_tokenize(sent))
        for word, pos in tagged_sentence:
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            corpus.append((lemmatizer.lemmatize(word.lower(), pos=wordnet_pos), pos))
    return corpus

def preprocess_feature(word_model, pos_model, para, question, para_length=400, ques_length=50, ans_length=10):
    para = tokenize_and_pos(para)
    question = tokenize_and_pos(question)
    # Omit the context over limited length
    if len(para) > para_length:
        para = para[:para_length]

    if len(question) > ques_length:
        question = question[:ques_length]

    para_word = np.zeros((para_length, word_model.vector_size), dtype=np.float32)
    para_pos = np.zeros((para_length, pos_model.vector_size), dtype=np.float32)
    for i in range(len(para)):
        if para[i][0] in word_model.wv:
            para_word[i] = word_model.wv[para[i][0]]
        if para[i][1] in pos_model.wv:
            para_pos[i] = pos_model.wv[para[i][1]]

    ques_word = np.zeros((ques_length, word_model.vector_size), dtype=np.float32)
    ques_pos = np.zeros((ques_length, pos_model.vector_size), dtype=np.float32)
    for i in range(len(question)):
        if question[i][0] in word_model.wv:
            ques_word[i] = word_model.wv[question[i][0]]
        if question[i][1] in pos_model.wv:
            ques_pos[i] = pos_model.wv[question[i][1]]
    return para_word, para_pos, ques_word, ques_pos


def preprocess_answer(para, answer):
    para_token = word_tokenize(para)
    ans_token = word_tokenize(answer)
    para_token = [w.lower() for w in para_token]
    ans_token = [w.lower() for w in ans_token]

    start = [i for i, v in enumerate(para_token) if v == ans_token[0]]
    for s in start:
        for e in range(0, len(ans_token)):
            if para_token[s + e] != ans_token[e]:
                break
            if e == len(ans_token) - 1:
                return s, s + e + 1
    return None, None


def find_raw_answer_position(para, answer):
    start = [i for i, v in enumerate(para) if v.lower() == answer[0].lower()]
    try:
        for s in start:
            for e in range(0, len(answer)):
                if para[s + e].lower() != answer[e].lower():
                    break

                if e == len(answer) - 1:
                    return s, s + e + 1
        return None, None
    except:
        return None, None

def keywords_extraction(ques):
    keywords = []
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    for word, pos in pos_tag(word_tokenize(ques)):
        if pos.startswith('V') or pos.startswith('J') or pos.startswith('R'):
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            preprocess_keyword = lemmatizer.lemmatize(word.lower(), pos=wordnet_pos)
            if preprocess_keyword not in stopWords:
                keywords.append(preprocess_keyword)
    return keywords
