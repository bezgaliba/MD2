#!/usr/local/bin/python3

import os
import sys
import nltk
import scipy
import re
import numpy
import json
import pickle
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import csv


def initialise(stop_txt, freq_tsv):
    global stoplist
    stoplist = set()

    with open(stop_txt) as txt:
        for word in txt:
            stoplist.add(normalize_text(word.strip()))

    print("[I] Word stoplist is read (" + str(len(stoplist)) + ").")

    global whitelist
    whitelist = set()

    with open(freq_tsv) as tsv:
        for entry in tsv:
            freq, word = entry.strip().split("\t")

            if int(freq) < 3:  # TODO: experiment with the threshold
                # Ignore the long tail - 2/3 of words occure less than N times
                continue

            whitelist.add(normalize_text(word))

    print("[I] Word whitelist is read (" + str(len(whitelist)) + ").")


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "100", text)

    return text.strip()


def normalize_vector(vector):
    words = list(vector.keys())

    for w in words:
        if w in stoplist or len(w) == 1 or w not in whitelist:
            vector.pop(w)

    return vector


def vectorize_text(text):
    return normalize_vector({word: True for word in nltk.word_tokenize(normalize_text(text))})


def read_data(file_path):
    # Atlasīt kolonas, kas satur saturīgu informāciju
    columns = ['map', 'att_team', 'att_side', 'hitbox', 'wp', 'wp_type', 'round_type']
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Saglabāt atslēga-vērtība pārus un ielikt sarakstā.
            entry = {column: row[column] for column in columns}
            data.append(entry)
    return data


def cleanTokens(word_tokenize):
    # Atrast vardus, piemeram, LeftArm un CounterTerrorist,
    # kur 2 vārdi ir sapludināti vienā un tos atdalīt ar atstarpi.
    # Tiek atdalīta tā vieta, kur parādas 2. lielais burts

    # Apstrādāt gadījumus, kur pareizi tiek lietoti vārdi ar vairākiem lielajiem burtiem tajā
    preservedWords = [
        'AK47', 'AUG', 'AWP', 'CZ', 'ECO', 'FORCE_BUY', 'GORZALATEAM', 'IBUYPOWER',
        'CS:GO', 'M4A1', 'M4A4', 'G3SG1', 'MP7', 'MP9', 'NORMAL', 'SG556', 'SSG08',
        'ESports', 'SMG', 'UMP', 'USP', 'XM1014'
        ]
    cleanTokens = re.sub('([a-z0-9])([A-Z])', r'\1 \2', word_tokenize).split()

    # Apstrādāt gadījumus, kur simboli - . un skaitļus
    cleanTokens = [re.sub(r'[.:/_-]', ' ', word) if word not in preservedWords else word for word in cleanTokens]

    # Apstrādāt gadījumus, kur lieliem burtiem iepriekš seko cipars, piem., M4A4
    cleanTokens = [re.sub(r'(\d)\s+([A-Z])', r'\1\2', word_tokenize) for word_tokenize in cleanTokens]

    # Izņemt ārā skaitļus
    cleanTokens = [re.sub(r'^[0-9]*$', '', word_tokenize) for word_tokenize in cleanTokens]

    # Izņemt ārā prefiksus "cs" and "de" datu tīrībai, tie ir 'gaming' nepienesīgi termini.
    cleanTokens = [re.sub(r'^(cs|de)\s', '', word_tokenize) for word_tokenize in cleanTokens]

    #Izņemt ārā tukšās rindas
    cleanTokens = [word for word in cleanTokens if word.strip() != '']

    #Izņemt ārā daudzu vārdu atstarpes, piem., 'Silver Bullets Gaming' ==> 'Silver /n Bullets /n Gaming /n'
    cleanTokens = [re.sub(r' ', '\n', word) if word not in preservedWords else word for word in cleanTokens]


    # Izvadīt katru token (kam ir atstarpes) savā rindā
    return '\n'.join(cleanTokens)


def extract_values(data, output_file):
    values = set()
    for entry in data:
        values.update(entry.values())

    # Izņemt ārā tokenu 'None', kas ir datu trūkumu apzīmējums.
    values = [value for value in values if value is not None]

    # Sakārtot alfabētiski
    values = sorted(values)
    
    values = [cleanTokens(value) for value in values]

    # Ierakstīt vērtības stoplist.txt
    with open(output_file, "w") as file:
        file.write('\n'.join(values))


def join_data(data_set):
    union = []

    for cat in data_set:
        union += data_set[cat]

    return union


def validate_accuracy(data_set, k):
    kfold = KFold(n_splits=k, shuffle=True)

    data_split = {}

    for cat in data_set:
        # K-Fold split for each class to ensure balanced training and test data sets
        folds = []

        for train, test in kfold.split(data_set[cat]):      # k loops
            train_data = numpy.array(data_set[cat])[train]  # vs. data[train]
            test_data = numpy.array(data_set[cat])[test]    # vs. data[test]
            folds.append({"train": train_data, "test": test_data})

        data_split[cat] = folds

    validations = []

    gold_result = []
    silver_result = []

    for i in range(k):
        # Join the training and test data into two respective sets
        train_data = numpy.array([])
        test_data = numpy.array([])

        for cat in data_split:
            if len(train_data) > 0:
                train_data = numpy.append(train_data, data_split[cat][i]["train"], axis=0)
            else:
                train_data = data_split[cat][i]["train"]

            if len(test_data) > 0:
                test_data = numpy.append(test_data, data_split[cat][i]["test"], axis=0)
            else:
                test_data = data_split[cat][i]["test"]

        # Naive Bayes classifier: training and evaluation
        nb = nltk.NaiveBayesClassifier.train(train_data)
        validations.append(nltk.classify.accuracy(nb, test_data))

        for t in test_data:
            gold_result.append(t[1])
            silver_result.append(nb.classify(t[0]))

    return (validations, gold_result, silver_result)


def run_validation(data_path, k, n):
    print("\n\t" + str(k) + "-fold cross-validation:\n")

    iterations = []
    gold_total = []
    silver_total = []

    start_time = datetime.datetime.now().replace(microsecond=0)

    for i in range(n):
        validations, gold, silver = validate_accuracy(read_data(data_path), k)
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
    #print(confusion_matrix(gold_total, silver_total))


def run_training(data_path, verbose):
    print("[I] Training an NB classifier...")

    start_time = datetime.datetime.now().replace(microsecond=0)

    # The final (production) model is trained by using all available data (train+test)
    nb = nltk.NaiveBayesClassifier.train(join_data(read_data(data_path)))

    end_time = datetime.datetime.now().replace(microsecond=0)
    print("[I] Training time: " + str(end_time - start_time))

    if verbose:
        nb.show_most_informative_features(n=100)

    dmp = open("nb_classifier.pickle", "wb")
    pickle.dump(nb, dmp)
    dmp.close()

    print("[I] NB classifier trained and serialised in a file.")


def run():
    dmp = open("nb_classifier.pickle", "rb")
    nb = pickle.load(dmp)
    dmp.close()

    print("[I] NB classifier loaded from a file.")

    while True:
        message = input("\nEnter a text:\n")

        if len(message) == 0:
            break

        features = vectorize_text(message)
        topic = nb.prob_classify(features)

        print("\n{0}\n".format(list(features.keys())))

        for t in topic.samples():
            print("{0}: {1:.3f}".format(t, topic.prob(t)))

        print("\nGuess: " + nb.classify(features))
