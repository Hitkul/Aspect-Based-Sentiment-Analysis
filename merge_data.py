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
    temp_text = []
    temp_labels = []
    for i in numbers:
        temp_text.append(sentences[i])
        temp_labels.append(labels[i])
    return temp_text,temp_labels


sentences,labels = shuffle_data(sentences,labels)

# print(labels.count(0))
# print(labels.count(1))

train_sentences,dev_sentences,test_sentences = sentences[:int(len(sentences)*0.98)],sentences[int(len(sentences)*0.98):int(len(sentences)*0.99)],sentences[int(len(sentences)*0.99):]
train_labels,dev_labels,test_labels = labels[:int(len(labels)*0.98)],labels[int(len(labels)*0.98):int(len(labels)*0.99)],labels[int(len(labels)*0.99):]

# print(train_labels.count(0),dev_labels.count(0),test_labels.count(0))
# print(train_labels.count(1),dev_labels.count(1),test_labels.count(1))


def write_to_file(filename,l1,l2):
    print('writing ',filename)
    with open('dataset/'+filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in zip(l1,l2):
            spamwriter.writerow(i)
write_to_file('final_train.csv',train_sentences,train_labels)
write_to_file('final_dev.csv',dev_sentences,dev_labels)
write_to_file('final_test.csv',test_sentences,test_labels)


