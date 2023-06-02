import os
import sys
import nltk
import scipy
import re
import numpy
import json
import pickle
import datetime
import random
import spacy
from nltk import FreqDist
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET
import spacy_udpipe

# TODO: atrast LV lemmatizēšanas bibliotēku

# def lemmatize_text(text):
#     nlp = spacy_udpipe.load("lv")
#     doc = nlp(text)
#     lemmas = [token.lemma_ for token in doc]
#     return " ".join(lemmas)

def start(fileName):
    # Nolasīt visus datus no XML faila
    tree = ET.parse('LVMED-Transcripts-900.xml')
    root = tree.getroot()

    # Izvilkt katra <doc> atribūtu un tekstu kopu
    data = []
    for doc in root.findall('doc'):
        modality = doc.attrib.get('modality')
        domain = doc.attrib.get('domain')
        text = doc.text.strip() if doc.text else ''
        ## lemmatized_text = lemmatize_text(text)  # Teksta lemmatizācija, kura nestrādā, jo nesekmīgi atrasta LV bibliotēka.
        data.append((modality, domain, text)) ## Ja izdara TODO, tad: text ---> (jāaizstāj ar) lemmatized_text

    # Izveidot jaunu .tsv ar UTF-8 kodējumu LV burtiem
    with open(fileName, 'w', encoding='utf-8') as file:
        # Galvenes izveidošana
        file.write('Modality-Domain\tText\n')

        # Katru xml tagu kā jaunu rindu
        for modality, domain, text in data:
            # Noņemt nost simbolus (normalizēšana)
            text = re.sub(r'\W+', ' ', text)  
            # Visus burtu samazināt lielumu(lowercase)
            text = text.lower()  
            # Ja ir daudz tukšās vietas, aizstāt ar vienu
            text = re.sub(r'\s+', ' ', text) 
            # Ierakstīt rindu failā, bet modality un domain top par vienu klasifikatoru
            file.write(f'{modality} {domain}\t{text}\n')

            tokens = nltk.word_tokenize(text)
            
            # Izrēķināt tokenu absolūto biežumu
            freq_dist = FreqDist(tokens)
            
            # Ierakstīt tokenu absolūto biežumu fail formātā 'FREQUENCY_VALUE'  'TOKEN'
    with open('frequency.tsv', 'w', encoding='utf-8') as freq_file:
        for token, freq in freq_dist.items():
            freq_file.write(f'{freq}\t{token}\n')

    print("TSV fails veiksmīgi izveidots")

def read_data(file):
    data_set = []

    with open(file, encoding='utf-8') as data:
        for entry in data:
            topics, text = entry.strip().split("\t")

            for topic in topics.split('.'):
                topic = topic.strip()
                featureset = vectorize_text(text)
                label = topic
                data_set.append((featureset, label))

    return data_set

def normalize_text(text):
	text = text.lower()
	text = re.sub(r"\d+", "100", text)

	return text.strip()


def normalize_vector(vector):
	words = list(vector.keys())

	# for w in words:
	# 	if w in stoplist or len(w) == 1 or w not in whitelist:
	# 		vector.pop(w)

	return vector


def vectorize_text(text):
	return normalize_vector({word: True for word in nltk.word_tokenize(normalize_text(text))})

def run_validation(data_path, k, n):
    print("\n\t" + str(k) + "-fold cross-validation:\n")

    iterations = []
    gold_total = []
    silver_total = []

    start_time = datetime.datetime.now().replace(microsecond=0)

    for i in range(n):
        data_set = read_data(data_path)
        if len(data_set) < k:
            print("Insufficient samples for", k, "folds of cross-validation. Using holdout validation instead.")
            train_data, test_data = splitDataSets(data_set)
            validations, gold, silver = validate_accuracy(train_data, test_data)
        else:
            kf = KFold(n_splits=k, shuffle=True)
            validations = []
            gold = []
            silver = []
            for train_index, test_index in kf.split(data_set):
                train_data = [data_set[i] for i in train_index]
                test_data = [data_set[i] for i in test_index]
                fold_validations, fold_gold, fold_silver = validate_accuracy(train_data, test_data)
                validations.extend(fold_validations)
                gold.extend(fold_gold)
                silver.extend(fold_silver)

        iterations.append(numpy.mean(validations))

        gold_total += gold
        silver_total += silver

        print("\t{0}.\t".format(i+1), end='')
        for v in validations:
            print("{0:.2f}  ".format(v), end='')
        print("\t{0:.0%}".format(numpy.mean(validations)))

    end_time = datetime.datetime.now().replace(microsecond=0)
    print("\n\tTotal validation time: " + str(end_time - start_time))

    print("\n\tAverage accuracy in {0} iterations: {1:.0%}\n".format(n, numpy.mean(iterations)))

    print(classification_report(gold_total, silver_total))

    print("Confusion matrix:")
    print(nltk.ConfusionMatrix(gold_total, silver_total))

def splitDataSets(data_set, train_ratio=0.6):
    train_data = []
    test_data = []

    # Samiksēt datus, lai treniņu/testa dati būtu dažādāki
    random.shuffle(data_set)

    # Izrēķināt indeksa lielumu, ko datus skaldīt
    split_index = int(len(data_set) * train_ratio)

    # Saskaldīt datus treniņam/testam.
    train_data = data_set[:split_index]
    test_data = data_set[split_index:]

    return train_data, test_data

def validate_accuracy(train_data, test_data):
    nb = nltk.NaiveBayesClassifier.train(train_data)
    
    # Atdalīt funkciju komplektu (featureset) un birkas no treniņu datiem
    test_featuresets = [t[0] for t in test_data]
    test_labels = [t[1] for t in test_data]
    
    # Izrēķināt precizitāti funkciju komplektam (featureset)
    accuracy = nltk.classify.accuracy(nb, zip(test_featuresets, test_labels))

    gold_result = test_labels
    silver_result = [nb.classify(t) for t in test_featuresets]

    return [accuracy], gold_result, silver_result