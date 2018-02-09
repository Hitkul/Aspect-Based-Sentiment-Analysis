#script to merge FiQA dataset with Sentiment140 datset
import csv
import json
import codecs
from random import shuffle

#loading FiQA dataset
def load_data_from_json(filename):
    with open(filename,'r') as f:
        foo = json.load(f)
    sentences = []
    labels = []
    for key in foo.keys():
        sentences.append(foo[key]['sentence'])
        labels.append(float(foo[key]['info'][0]['sentiment_score']))
    
    labels = [1 if x>=0 else 0 for x in labels]
    return sentences,labels

FiQA_sentences,FiQA_labels = load_data_from_json('dataset/master.json')

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


def shuffle_data(sentences,labels):
    numbers = [i for i in range(len(sentences))]
    shuffle(numbers)
    temp_text = []
    temp_labels = []
    for i in numbers:
        temp_text.append(sentences[i])
        temp_labels.append(labels[i])
    return temp_text,temp_labels

sentences = senti140_test_sentences+senti140_train_sentences
labels = senti140_test_labels+senti140_train_labels

sentences,labels = shuffle_data(sentences,labels)

sentences = sentences[:10000]
labels = labels[:10000]


sentences += FiQA_sentences
labels+=FiQA_labels

sentences,labels = shuffle_data(sentences,labels)





train_sentences,dev_sentences,test_sentences = sentences[:int(len(sentences)*0.98)],sentences[int(len(sentences)*0.98):int(len(sentences)*0.99)],sentences[int(len(sentences)*0.99):]
train_labels,dev_labels,test_labels = labels[:int(len(labels)*0.98)],labels[int(len(labels)*0.98):int(len(labels)*0.99)],labels[int(len(labels)*0.99):]


print(labels.count(0))
print(labels.count(1))

print(train_labels.count(0),dev_labels.count(0),test_labels.count(0))
print(train_labels.count(1),dev_labels.count(1),test_labels.count(1))

print(len(train_sentences),len(dev_sentences),len(test_sentences))
print(len(train_labels),len(dev_labels),len(test_labels))

print(type(train_sentences),type(dev_sentences),type(test_sentences))
print(type(train_labels),type(dev_labels),type(test_labels))

train_dict = {'sentence':train_sentences,'labels':train_labels}
dev_dict = {'sentence':dev_sentences,'labels':dev_labels}
test_dict = {'sentence':test_sentences,'labels':test_labels}

def write_to_json(filename,_dict):
    print('writing ',filename)
    with open('dataset/'+filename,'w') as fout:
        json.dump(_dict,fout,indent=4)

# def write_to_file(filename,l1,l2):
#     print('writing ',filename)
#     with open('dataset/'+filename, 'w', newline='') as csvfile:
#         spamwriter = csv.writer(csvfile, delimiter=',',
#                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         for i in zip(l1,l2):
#             spamwriter.writerow(i)
write_to_json('final_train.json',train_dict)
write_to_json('final_dev.json',dev_dict)
write_to_json('final_test.json',test_dict)


