#multi channel CNN for sentiment analysis
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import numpy as np
import re
import json
import codecs
import word2vecReader as godin_embedding
import pickle
from random import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import keras.backend as K
# import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from math import sqrt
from gensim.models import KeyedVectors
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args



#loading data
def load_data_from_file(filename):
    print("loading file = ",filename)
    with open(filename,'r') as f:
        foo = json.load(f)
    return foo['sentence'],foo['labels']

    

# sentences,score = load_data_from_xml('dataset/financial_posts_ABSA_train.xml')
trainX,trainY = load_data_from_file('dataset/final_train.json')
devX,devY = load_data_from_file('dataset/final_dev.json')

print('train data len')
print(len(trainX),len(trainY))

print('dev data len')
print(len(devX),len(devY))

print('train and dev count 1')
print(trainY.count(1),devY.count(1))

print('train and dev count 0')
print(trainY.count(0),devY.count(0))


# #only using subset of data for testing code
trainX = trainX[:100]
trainY = trainY[:100]
devX = devX[:10]
devY = devY[:10]


# turn a sentence into clean tokens
def clean_sentence(sentence):
    #remove multiple repeat non num-aplha char !!!!!!!!!-->!
    sentence = re.sub(r'(\W)\1{2,}', r'\1', sentence) 
    #removes alpha char repeating more than twice aaaa->aa
    sentence = re.sub(r'(\w)\1{2,}', r'\1\1', sentence)
    #removes links
    sentence = re.sub(r'(?P<url>https?://[^\s]+)', r'', sentence)
    # remove @usernames
    sentence = re.sub(r"\@(\w+)", "", sentence)
    #removing stock names to see if it helps
#     sentence = re.sub(r"(?:\$|https?\://)\S+", "", sentence)
    #remove # from #tags
    sentence = sentence.replace('#','')
    # split into tokens by white space
    tokens = sentence.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation.replace('$',''))
    tokens = [w.translate(table) for w in tokens]
#     remove remaining tokens that are not alphabetic
#     tokens = [word for word in tokens if word.isalpha()]
#no removing non alpha words to keep stock names($ZSL)
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens



# extract sentences out of df and cleaning it
print('cleaning train set')
trainX = [clean_sentence(x) for x in trainX]
print('cleaning dev set')
devX = [clean_sentence(x) for x in devX]
# sentences

# print(len(trainX),len(trainY))
# print(len(devX),len(devY))

#converting output matrix [-ve,+ve]

devY = to_categorical(devY,2)
trainY = to_categorical(trainY,2)


lengths = [len(s.split()) for s in trainX]
max_length = max(lengths)


# plt.subplots(figsize=(12,10))
# plt.hist(lengths, normed=True,edgecolor='black')



#loading Google Word2Vec
def load_google_word2vec(file_name):
    print("Loading word2vec model, this can take some time...")
    return KeyedVectors.load_word2vec_format(file_name, binary=True)


word2vec_model= load_google_word2vec('word_embeddings/GoogleNews-vectors-negative300.bin')


#loading godin word embedding
def load_godin_word_embedding(path):
    print("Loading goding model, this can take some time...")
    return godin_embedding.Word2Vec.load_word2vec_format(path, binary=True)



godin_model = load_godin_word_embedding("word_embeddings/word2vec_twitter_model.bin")


def get_embedding_matrix(model,sentence,godin_flag = False):
    tokens = sentence.split()[:max_length]
    if godin_flag:
        embedding_matrix = np.zeros((max_length,400))
    else:
        embedding_matrix = np.zeros((max_length,300))
    for i,word in enumerate(tokens):
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix


print("bulding word2vec matrix of train set")
train_word2vec = np.asarray([get_embedding_matrix(word2vec_model,x) for x in trainX])
print("bulding godin matrix of train set")
train_godin = np.asarray([get_embedding_matrix(godin_model,x,godin_flag=True) for x in trainX])
print("bulding word2vec matrix of dev set")
dev_word2vec = np.asarray([get_embedding_matrix(word2vec_model,x) for x in devX])
print("bulding godin matrix of dev set")
dev_godin = np.asarray([get_embedding_matrix(godin_model,x,godin_flag=True) for x in devX])




para_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',name='learning_rate')



para_dropout = Real(low=0.4, high=0.9,name = 'dropout')

para_n_dense = Categorical(categories=[100,200,300,400], name='n_dense')
para_n_filters = Categorical(categories=[100,200,300,400],name='n_filters')


para_filter_size_c1 = Integer(low=1,high=6,name = 'filter_size_c1')
para_filter_size_c2 = Integer(low=1,high=6,name = 'filter_size_c2')
para_filter_size_c3 = Integer(low=1,high=6,name = 'filter_size_c3')


para_em_c1 = Categorical(categories=['embedding_matrix_godin','embedding_matrix_word2vec'],name='em_c1')
para_em_c2 = Categorical(categories=['embedding_matrix_godin','embedding_matrix_word2vec'],name='em_c2')
para_em_c3 = Categorical(categories=['embedding_matrix_godin','embedding_matrix_word2vec'],name='em_c3')


para_batch_size = Categorical(categories=[8,16,32,64],name='batch_size')


para_epoch = Categorical(categories=[10,15,20,30],name='epoch')


parameters = [para_learning_rate,para_dropout,para_n_dense,para_n_filters,para_filter_size_c1,para_filter_size_c2,para_filter_size_c3,para_em_c1,para_em_c2,para_em_c3,para_batch_size,para_epoch]

default_parameters = [1e-4,0.5,100,100,2,4,6,'embedding_matrix_word2vec','embedding_matrix_godin','embedding_matrix_word2vec',16,10]


def define_model(length,n_dense,dropout,learning_rate,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3):
    # channel 1
    if em_c1 == 'embedding_matrix_word2vec':
        inputs1 = Input(shape=(length,300))
    else:
        inputs1 = Input(shape=(length,400))

    conv1 = Conv1D(filters=n_filters, kernel_size=filter_size_c1, activation='relu')(inputs1)
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    if em_c2 == 'embedding_matrix_word2vec':
        inputs2 = Input(shape=(length,300))
    else:
        inputs2 = Input(shape=(length,400))
    conv2 = Conv1D(filters=n_filters, kernel_size=filter_size_c2, activation='relu')(inputs2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    if em_c3 == 'embedding_matrix_word2vec':
        inputs3 = Input(shape=(length,300))
    else:
        inputs3 = Input(shape=(length,400))
    conv3 = Conv1D(filters=n_filters, kernel_size=filter_size_c3, activation='relu')(inputs3)
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(n_dense, activation='relu')(merged)
    outputs = Dense(2, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # summarize
    # print(model.summary())
#     plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model



#dict to store performance of all models
record = dict()
key=0




@use_named_args(dimensions=parameters)
def fitness(learning_rate,dropout,n_dense,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3,batch_size,epoch):
# n_dense,dropout,learning_rate,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3,free_em_dim,em_trainable_flag
    # Print the hyper-parameters.
    global key
    global record
    print('-----------------------------combination no={0}------------------'.format(key))
    print('learning rate ==>',learning_rate)
    print('dropout==>',dropout)
    print('n_dense==>',n_dense)
    print('n_filters==>',n_filters)
    print('filter_size_c1',filter_size_c1)
    print('filter_size_c2',filter_size_c2)
    print('filter_size_c3',filter_size_c3)
    print('em_c1==>',em_c1)
    print('em_c2==>',em_c2)
    print('em_c3==>',em_c3)
    print('batch_size==>',batch_size)
    print('epocs==>',epoch)

    # Create the neural network with these hyper-parameters.
    model = define_model(length = max_length,
                         n_dense=n_dense,
                         dropout=dropout,
                         learning_rate=learning_rate,
                         n_filters=n_filters,
                         filter_size_c1=int(filter_size_c1),
                         filter_size_c2=int(filter_size_c2),
                         filter_size_c3=int(filter_size_c3),
                         em_c1=em_c1,
                         em_c2=em_c2,
                         em_c3=em_c3)
    input_train_array = [train_word2vec if x=='embedding_matrix_word2vec' else train_godin for x in [em_c1,em_c2,em_c3]]
    input_dev_array = [dev_word2vec if x=='embedding_matrix_word2vec' else dev_godin for x in [em_c1,em_c2,em_c3]]
    
  
    history_object = model.fit(input_train_array, trainY,epochs=epoch, batch_size=batch_size,validation_data=(input_dev_array,devY))

    accuracy = history_object.history['val_acc'][-1]

    print("Accuracy: {0:.2%}".format(accuracy))
    
    
    record[key] = {'parameters':[learning_rate,dropout,n_dense,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3,batch_size,epoch],'val_acc':accuracy}
    
    model.save('models/'+str(key)+'.h5')
    
    with open('models/record.pickle', 'wb') as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    key+=1
    
 
    del model
    
    K.clear_session()
    return -accuracy


search_result = gp_minimize(func=fitness,
                            dimensions=parameters,
                            acq_func='EI',
                            n_calls=11,
                            x0=default_parameters)



with open('models/record.pickle', 'wb') as handle:
    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
