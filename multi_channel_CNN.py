
#multi channel CNN for sentiment analysis
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from string import punctuation
from os import remove
import pandas as pd
import numpy as np
import re
import fasttext
import csv
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
import matplotlib.pyplot as plt
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
    sentences = []
    label = []
    with codecs.open(filename, "r",encoding='utf-8', errors='ignore') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            try:
                sentences.append(row[0])
                label.append(row[1])
            except:
                print(row)
    return sentences,label


# sentences,score = load_data_from_xml('dataset/financial_posts_ABSA_train.xml')
trainX,trainY = load_data_from_file('dataset/final_train.csv')
devX,devY = load_data_from_file('dataset/final_dev.csv')


#only using 1%data for testing code
trainX = trainX[:5]
trainY = trainY[:5]
devX = devX[:2]
devY = devY[:2]


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
    stop_words = set(nltk.corpus.stopwords.words('english'))
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


#copying sentences for fastext traning 
tranLines = list(trainX)
devLines = list(devX)



print(len(trainX),len(trainY))
print(len(devX),len(devY))



#converting output matrix [-ve,+ve]
devY = to_categorical(devY,2)
trainY = to_categorical(trainY,2)

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# create tokenizer
tokenizer = create_tokenizer(trainX)
# calculate max document length
lengths = [len(s.split()) for s in trainX]
max_length = max(lengths)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % max_length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainX, max_length)
devX = encode_text(tokenizer, devX, max_length)
print(trainX.shape,devX.shape)




# plt.subplots(figsize=(12,10))
# plt.hist(lengths, normed=True,edgecolor='black')


# considring only few sentences have len >20, we can also take max_len = 20

#loading GloVe embedding
def load_GloVe_embedding(file_name):
    embeddings_index = dict()
    f = open(file_name)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


# create a weight matrix for words in training docs
def get_GloVe_embedding_matrix(embeddings_index):
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

print('loading Glove embedding')
embeddings_index_glove = load_GloVe_embedding('word_embeddings/glove.6B.300d.txt')
embedding_matrix_glove = get_GloVe_embedding_matrix(embeddings_index_glove)



#loading Google Word2Vec
def load_google_word2vec(file_name):
    return KeyedVectors.load_word2vec_format(file_name, binary=True)



def get_word2vec_embedding_matrix(model):
    embedding_matrix = np.zeros((vocab_size,300))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix

print('loading word2vec embedding')
word2vec_model= load_google_word2vec('word_embeddings/GoogleNews-vectors-negative300.bin')
embedding_matrix_word2vec = get_word2vec_embedding_matrix(word2vec_model)



#fast text word embedding
def load_fast_text_model(sentences):
    try:
        m = fasttext.load_model('fast_text_model.bin')
        print("trained model loaded")
        return m
    except:
        print("traning new model")
        with open('temp_file.txt','w') as temp_file:
            for sentence in sentences:
#                 sentence = sentence.encode('UTF-8')
#                 print(sentence)
                temp_file.write(sentence)
        m = fasttext.cbow('temp_file.txt','fast_text_model')
        remove('temp_file.txt')
        print('model trained')
        return m



def get_fast_text_matrix(model):
    embedding_matrix = np.zeros((vocab_size,100))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix


#need to fix this
fast_text_model = load_fast_text_model(tranLines+devLines)
embedding_matrix_fast_text = get_fast_text_matrix(fast_text_model)


#loading godin word embedding
def load_godin_word_embedding(path):
    print("Loading the model, this can take some time...")
    return godin_embedding.Word2Vec.load_word2vec_format(path, binary=True)


def get_godin_embedding_matrix(model):
    embedding_matrix = np.zeros((vocab_size,400))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix


print('loading godin embedding')
godin_model = load_godin_word_embedding("word_embeddings/word2vec_twitter_model.bin")
embedding_matrix_godin = get_godin_embedding_matrix(godin_model)


# ## Hyper Parameters

para_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',name='learning_rate')

para_dropout = Categorical(categories=[0.4,0.5,0.6,0.7,0.8,0.9],name='dropout')
# para_n_dense = Integer(low=100, high=400, name='n_dense')
para_n_dense = Categorical(categories=[100,200,300,400], name='n_dense')


# para_n_filters = Integer(low=100,high=400,name='n_filters')
para_n_filters = Categorical(categories=[100,200,300,400],name='n_filters')

para_filter_size_c1 = Integer(low=1,high=6,name = 'filter_size_c1')

para_filter_size_c2 = Integer(low=1,high=6,name = 'filter_size_c2')

para_filter_size_c3 = Integer(low=1,high=6,name = 'filter_size_c3')

# 'embedding_matrix_fast_text',
para_em_c1 = Categorical(categories=['embedding_matrix_fast_text','embedding_matrix_godin','embedding_matrix_word2vec','embedding_matrix_glove','free'],name='em_c1')

para_em_c2 = Categorical(categories=['embedding_matrix_fast_text','embedding_matrix_godin','embedding_matrix_word2vec','embedding_matrix_glove','free'],name='em_c2')

para_em_c3 = Categorical(categories=['embedding_matrix_fast_text','embedding_matrix_godin','embedding_matrix_word2vec','embedding_matrix_glove','free'],name='em_c3')

para_em_trainable_flag = Categorical(categories=[True,False],name='em_trainable_flag')

para_free_em_dim = Categorical(categories=[100,300,400],name='free_em_dim')

para_batch_size = Categorical(categories=[50,100,150],name='batch_size')

# ,50,100,200,300,400,500
para_epoch = Categorical(categories=[10,50,100,200,300,400,500],name='epoch')

parameters = [para_learning_rate,para_dropout,para_n_dense,para_n_filters,para_filter_size_c1,para_filter_size_c2,para_filter_size_c3,para_em_c1,para_em_c2,para_em_c3,para_em_trainable_flag,para_free_em_dim,para_batch_size,para_epoch]

default_parameters = [1e-3,0.8,100,100,1,5,2,'embedding_matrix_word2vec','embedding_matrix_glove','embedding_matrix_fast_text',False,100,50,10]


# ## Model

# In[111]:


# define the model
def define_model(length,vocab_size,n_dense,dropout,learning_rate,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3,free_em_dim,em_trainable_flag):
    # channel 1
    inputs1 = Input(shape=(length,))
    if em_c1 == 'free':
        embedding1 = Embedding(vocab_size, free_em_dim)(inputs1)
    else:
        embedding1 = Embedding(vocab_size, len(eval(em_c1)[0]), weights = [eval(em_c1)],input_length=length,trainable = em_trainable_flag)(inputs1)

    conv1 = Conv1D(filters=n_filters, kernel_size=filter_size_c1, activation='relu')(embedding1)
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
#     embedding2 = Embedding(vocab_size, 400, weights = [embedding_matrix_godin],input_length=length,trainable = em_trainable_flag)(inputs2)
    if em_c2 == 'free':
        embedding2 = Embedding(vocab_size, free_em_dim)(inputs2)
    else:
        embedding2 = Embedding(vocab_size, len(eval(em_c2)[0]), weights = [eval(em_c2)],input_length=length,trainable = em_trainable_flag)(inputs2)
    conv2 = Conv1D(filters=n_filters, kernel_size=filter_size_c2, activation='relu')(embedding2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
#     embedding3 = Embedding(vocab_size, 400)(inputs3)
    if em_c3 == 'free':
        embedding3 = Embedding(vocab_size, free_em_dim)(inputs3)
    else:
        embedding3 = Embedding(vocab_size, len(eval(em_c3)[0]), weights = [eval(em_c3)],input_length=length,trainable = em_trainable_flag)(inputs3)
    conv3 = Conv1D(filters=n_filters, kernel_size=filter_size_c3, activation='relu')(embedding3)
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
#     print(model.summary())
#     plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

#dict to store performance of all models
record = dict()
key=0

@use_named_args(dimensions=parameters)
def fitness(learning_rate,dropout,n_dense,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3,em_trainable_flag,free_em_dim,batch_size,epoch):
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
    print('em_trainable_flag ==>',em_trainable_flag)
    print('free_em_dim==>',free_em_dim)
    print('batch_size==>',batch_size)
    print('epocs==>',epoch)

    # Create the neural network with these hyper-parameters.
    model = define_model(length = max_length,
                         vocab_size=vocab_size,
                         n_dense=n_dense,
                         dropout=dropout,
                         learning_rate=learning_rate,
                         n_filters=n_filters,
                         filter_size_c1=int(filter_size_c1),
                         filter_size_c2=int(filter_size_c2),
                         filter_size_c3=int(filter_size_c3),
                         em_c1=em_c1,
                         em_c2=em_c2,
                         em_c3=em_c3,
                         free_em_dim=free_em_dim,
                         em_trainable_flag=em_trainable_flag)

    
    # Use Keras to train the model.
    history_object = model.fit([trainX,trainX,trainX], trainY,epochs=epoch, batch_size=batch_size,validation_data=([devX,devX,devX],devY))

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history_object.history['val_acc'][-1]

    # Print the classification accuracy.
    print("Accuracy: {0:.2%}".format(accuracy))
    
    
    record[key] = {'parameters':[learning_rate,dropout,n_dense,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3,em_trainable_flag,free_em_dim,batch_size,epoch],'val_acc':accuracy}
    
    model.save('models/'+str(key)+'.h5')
    
    key+=1
    
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy



search_result = gp_minimize(func=fitness,
                            dimensions=parameters,
                            acq_func='EI',
                            n_calls=11,
                            x0=default_parameters)


sorted(zip(search_result.func_vals, search_result.x_iters))



# record
# fit model
# history_object = model.fit([trainX,trainX,trainX], trainY,epochs=10, batch_size=16)



with open('models/record.pickle', 'wb') as handle:
    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)