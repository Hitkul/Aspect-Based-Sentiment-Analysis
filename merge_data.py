#script to merge FiQA dataset with Sentiment140 datset

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

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

sentences,labels = load_data_from_xml('dataset/financial_posts_ABSA_train.xml')
print(sentences[:5])
print(labels[:5])
print(len(sentences),len(labels))
