#script to merge FiQA dataset with Sentiment140 datset

import csv
import xml.etree.ElementTree as ET
import codecs
from random import shuffle

#loading FiQA dataset
def load_data_from_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    sentences = list()
    score= list()
    for row in root:
        score.append(float(row[1].text))
        sentences.append(row[2].text)

    labels = [1 if x >= 0 else 0 for x in score]
    return sentences,labels
FiQA_sentences,FiQA_labels = load_data_from_xml('dataset/financial_posts_ABSA_train.xml')

#loading sentiment 140 dataset
def load_data_from_csv(filename):
    labels = []
    sentences = []
    with codecs.open(filename, "r",encoding='utf-8', errors='ignore') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if int(row[0])!=2:
                labels.append(int(row[0]))
                sentences.append(row[-1])
    return sentences,labels

senti140_test_sentences,senti140_test_labels = load_data_from_csv('dataset/testdata.manual.2009.06.14.csv')
senti140_train_sentences,senti140_train_labels = load_data_from_csv('dataset/training.1600000.processed.noemoticon.csv')
senti140_test_labels = [1 if x == 4 else 0 for x in senti140_test_labels]
senti140_train_labels = [1 if x == 4 else 0 for x in senti140_train_labels]

sentences = FiQA_sentences+senti140_test_sentences+senti140_train_sentences
labels = FiQA_labels+senti140_test_labels+senti140_train_labels

def shuffle_data(sentences,labels):
    numbers = [i for i in range(len(sentences))]
    shuffle(numbers)
    temp_text = sentences
    temp_labels = labels
    for i in numbers:
        sentences[i] = temp_text[i]
        labels[i]=temp_labels[i]
    print(len(sentences))
    print(len(labels))
    return sentences,labels
sentences,labels = shuffle_data(sentences,labels)
train_sentences,dev_sentences,test_sentences = sentences[:int(len(sentences)*0.98)],sentences[int(len(sentences)*0.98):int(len(sentences)*0.99)],sentences[int(len(sentences)*0.99):]
train_labels,dev_labels,test_labels = labels[:int(len(labels)*0.98)],labels[int(len(labels)*0.98):int(len(labels)*0.99)],labels[int(len(labels)*0.99):]
print(len(train_sentences),' ',len(dev_sentences),' ',len(test_sentences))
print(len(train_labels),' ',len(dev_labels),' ',len(test_labels))
print('writing file')
with open('dataset/test.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in zip(sentences,labels):
        spamwriter.writerow(i)

