
import json
import re
import nltk
from string import punctuation
import numpy as np
import networkx as nx
import spacy
from math import exp

def is_post(s):
    if len(re.findall(r'\$([a-zA-Z_]+)',s))>0:
        return True
    return False


def prepare_sentence(s):
    sentences = nltk.sent_tokenize(s)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    return chunked_sentences



def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names




def get_ners(s):
    chunked_sentences = prepare_sentence(s)
    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))
    entity_names = [x.split()[0] for x in list(set(entity_names))]
    return entity_names


def get_index_of_targets(s,is_post_flag = True):
    if is_post_flag:
        targets = ['$'+x for x in re.findall(r'\$([a-zA-Z_]+)',s)]
        index = [i for i,j in enumerate(s.split()) if j in targets]
        return index
    else:
        targets = get_ners(s)
        index = [i for i,j in enumerate(s.split()) if j in targets]
        return index


def get_sentence_dependency_tree(s):
    nlp = spacy.load('en')
    document = nlp(s)

    # Load spacy's dependency tree into a networkx graph
    edges = []
    for token in document:
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_,token.i),
                          '{0}-{1}'.format(child.lower_,child.i)))
    graph = nx.DiGraph(edges)
    for node in graph.in_degree():
        if node[1] == 0:
            root = node[0]
            break
    nodes = graph.node()
    depth = 0
    for node in nodes:
        try:
            temp = nx.shortest_path_length(graph, source=root,target=node)
        except:
            continue
        if temp > depth:
            depth = temp
    return graph, depth


def get_distance_between_two_words(graph,node1,node1_index,node2,node2_index,depth):
    node1 = node1.lower().replace('$','')
    node1 = node1+'-'+str(node1_index)
    node2 = node2.lower().replace('$','')
    node2 = node2+'-'+str(node2_index)
    try:
        return nx.shortest_path_length(graph.to_undirected(), source=node1, target=node2)
    except:
        return 10*depth 


def get_sentence_tokens_prob(s):
    s_prob_vectors = []
    tokens = s.split()
    prob_target = np.zeros(len(tokens))
    if is_post(s):
        target_index = get_index_of_targets(s)
    else:
        target_index = get_index_of_targets(s,is_post_flag=False)
    
    if len(target_index) == 0:
        s_prob_vectors.append(np.zeros(len(tokens)))
        return s_prob_vectors
    prob_each_target = 1.0/len(target_index)
    graph,depth = get_sentence_dependency_tree(s.replace('$',''))
    for i in target_index:
        prob_target[i] = prob_each_target
    for i in range(len(prob_target)):
        if prob_target[i]!=0:
            sentence_prob = np.zeros(len(tokens))
            for j in range(len(sentence_prob)):
                if j==i:
                    sentence_prob[j]+=1+prob_target[i]
                else:
                    sentence_prob[j]+=prob_target[i]*exp(-((get_distance_between_two_words(graph,tokens[i],i,tokens[j],j,depth)**2)/(2.0*depth)))
            s_prob_vectors.append(sentence_prob)
    return s_prob_vectors


def renormalize_series(series):
    mean_series = np.mean(series)
    std_series = np.std(series)
    if std_series == 0:
        return [1 for x in series]
    series_normalized = [(x-mean_series)/std_series for x in series]
    return [x+1 for x in series_normalized]

def get_normalized_sentence_relation_vector(s):
    sentence_relation_vector = get_sentence_tokens_prob(s)
    sentence_relation_vector = [renormalize_series(x) for x in sentence_relation_vector]
    return sentence_relation_vector
