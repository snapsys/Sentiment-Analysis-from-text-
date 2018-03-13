from __future__ import absolute_import
from __future__ import print_function
import numpy as np
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv1D, Dense, Flatten, Activation
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.optimizers import Adam, Adagrad, SGD 
from keras import backend as K
import scipy.io as sio 
import numpy as np
import scipy.sparse as sparse
from keras import regularizers
from keras.constraints import maxnorm, non_neg
import keras
import sys 
import time
import h5py, pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
#import memory_profiler
#from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics 
import tensorflow as tf
import random
import argparse 


def Siamese_lik_ratio(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Info: This computes binary cross entropy for verification loss
    Inputs: y_true -- ground truth, obtained from batch genrator
            y_pred -- output of cosine_distance function. It is dot product matrix of document embeeddings
    '''
    
    y_pred = [y_pred[i,j] for i in range(total_no_samples) for j in range(i+1, total_no_samples)]
    y_pred = tf.stack(y_pred, axis=0)
    y_true_pair = [tf.cast(tf.not_equal(y_true[i], y_true[j]), tf.float32) for i in range(total_no_samples) for j in range(i+1, total_no_samples)]
    y_true_pair = tf.stack(y_true_pair, axis=0)
    no_neg_pairs = tf.reduce_sum(y_true_pair)
    no_pos_pairs = tf.reduce_sum(1-y_true_pair)
    neg_loss_wt = tf.cond(no_neg_pairs > 0, lambda: tf.divide(no_pos_pairs, no_neg_pairs), lambda: tf.Variable(0.0))
    return K.mean(-(1-y_true_pair) * K.log(y_pred+1e-20) - neg_loss_wt * (y_true_pair) * K.log(1-y_pred+1e-20))


def cosine_distance(vects):
    ''' calculates dot product between every other document embedding in the minibatch '''
    x = vects[0]
    x = K.l2_normalize(x, axis=-1)
    dot_prod = K.dot(x, K.transpose(x))
    return K.sigmoid(dot_prod)


def cos_dist_output_shape(shapes):
    ''' returns shape of lambda function. Used within keras classes '''
    shape1, shape2 = shapes
    return (total_no_samples, total_no_samples)


def conv_filt_v2(input_a, no_filters, filt_width):
    ''' base module for text data
    Inputs: input_a -- input tensor
            no_filters -- number of filter maps used in conv layer (details)
            filt_width -- convolution window width
    outputs: output tensor of base module
    '''
    # 20000 is the vocabulary size. It should be greater than or equal to your tokenizer vocabulary size
    embed = Embedding(vocab_size, embed_size)(input_a)
    conv = Conv1D(no_filters, filt_width, strides=1)(embed)
    conv = Activation('relu')(conv)
    conv = AveragePooling1D(pool_size=2)(conv)
    conv = Dropout(0.5)(conv)
    conv = GlobalAveragePooling1D()(conv)
    return conv


def doc_represent_embedding(tokenizer, data):
    ''' uses one hot encoding representaion for each word. List of lists is returned for a document 
        Inputs:  tokenizer -- used to turn the text into appropriate input format. Usually
                              converting the text into sequence of word indices
                              (https://keras.io/preprocessing/text/#tokenizer)
                 data -- text input, output of text_file_load() function
        outputs: input_rep -- In theory, Matrix of one-hot encoding vectors for the given document
                              Here for practical purposes, it returns a list. In this list, 
                              each element is also a list which contains non-zeros location in the corresponding vector
    '''
    x = keras.preprocessing.text.text_to_word_sequence(data)
    temp = tokenizer.texts_to_sequences(x)
    sequences = [list(np.concatenate(temp))]
    input_rep = pad_sequences(sequences, maxlen=MaxLen)
    input_rep = np.reshape(input_rep, (1,-1))
    return input_rep


def doc_represent_BOW(tokenizer, data):
    ''' Used to calculate Bag of words representation for the given document.
        Inputs: tokenizer -- used to turn the text into appropriate input format. Usually   
                             converting the text into sequence of word indices
                data -- text input, output of text_file_load() function
        outputs: input_rep -- Bag of words vector for the given document -- Vector of count of each word in the document
    '''

    x = keras.preprocessing.text.text_to_word_sequence(data)
    temp = tokenizer.texts_to_matrix(x, mode='count')
    input_rep = np.sum(temp,axis=0)
    input_rep = np.reshape(input_rep, (-1, 1))
    return input_rep


def text_file_load(file_id, transcripts_dir='/var/users/raghavendra/CSAT_scripts/transcripts_mod/'):
    ''' returns sequence of words of a document removing new line character. Expects input file to have only relevant text
        file_id -- name of file to be used
    '''
    with open(transcripts_dir + file_id + '.csv', 'r') as f:
        conversation = f.readlines()
        conversation = [sentence.strip() for sentence in conversation]
        conversation = ' '.join(conversation)
    return conversation


def allpairs_gen_joint(utt2label, label2utt, tokenizer, no_classesPerBatch, sampPerClass, no_classes):
    ''' minibatch generator for model training. It yields a batch of samples everytime using yield statemnet
        Inputs:  utt2label -- dictionary to map utterance to correspomnding label. This dictionary contains training and validation                              utterances combined as I am using single generator for both training and validation data. label2utt 
                              is different for train and val data
                 label2utt -- dictionary to map label to utterances. This is different for training and dev data sets
                 tokenizer -- used to turn the text into appropriate input format. Usually converting the text into
                              sequence of word indices (https://keras.io/preprocessing/text/#tokenizer)
                 no_classesPerBatch -- number of classes to choose minibatch samples from
                 sampPerClass -- number of samples to choose per each class
                 no_classes -- total number of classes. Number of output neurons in classification 
    '''
    while 1:
        # choose classes randomly 
        classesPerBatch = random.sample(range(no_classes), no_classesPerBatch)
        # choose samples (utterance index or utterance name) randomly from each class
        samples = [np.random.choice(label2utt[str(i)], size=sampPerClass) for i in classesPerBatch]
        samples = np.concatenate(samples)   # indices of set of samples in the batch

        # sample to label matrix
        batch_ind_set = np.array([[i, utt2label[i]] for i in samples])

        # data, classification labels and verification labels for those samples
        batch_data = np.vstack([doc_represent_embedding(tokenizer, text_file_load(utt)) for utt in batch_ind_set[:, 0]])
        batch_labels = np.vstack(keras.utils.to_categorical(int(j), no_classes) for j in batch_ind_set[:, 1])
        aux_labels = np.vstack(int(j) for j in batch_ind_set[:, 1])
        yield({'input1':batch_data},{'main_output_1':batch_labels, 'aux_output':aux_labels})


def data_dict_gen(data_dir, fold_no):
    ''' creating utterance lists and dictionaries to map utterance to labels. Used when you are testing on existing data.
            We can remove this function later if we do not need. I kept it to test on training data also
            Input arguments and output argumnets are self-explanatory. 
    Inputs:
        data_dir -- data directory where CV_folds.pkl, file_id_labels.txt exist
        fold_no -- Cross validation fold no to test on. It is used to get utterance list
    outputs:
        utt2label -- dictiionary with utterances as keys and labels as values, contains train+test
        label2utt_train -- dictiionary with labels as keys and utterances as values, only for train uttearances
        label2utt_test -- same as label2utt_train but only for test utterances
        test_uttlist -- list of test utterances
        no_train_utt -- number of training utterances in the data
    '''

    CV_folds = pickle.load(open(data_dir + '/CV_folds.pkl','rb'))
    fild_id_list = CV_folds['file_id']
    
    utt2label = {}
    with open(data_dir + '/file_id_labels.txt', 'r') as f:
        for i in f.readlines():
            file_id = i.strip().split()[0]
            file_class = i.strip().split()[1]
            utt2label[file_id] = file_class
    no_classes = len(np.unique([utt2label[i] for i in utt2label.keys()]))    

    label2utt_train = {}
    for i in range(no_classes):
        label2utt_train[str(i)] = []  
    for file_ind in CV_folds['train' + str(fold_no)]:
        file_id = fild_id_list[int(file_ind)]
        file_class = utt2label[file_id]
        label2utt_train[file_class].append(file_id)

    label2utt_test = {}
    for i in range(no_classes):
        label2utt_test[str(i)] = []
    for file_ind in CV_folds['test' + str(fold_no)]:
        file_id = fild_id_list[int(file_ind)]
        file_class = utt2label[file_id]
        label2utt_test[file_class].append(file_id)
    
    no_train_utt = np.sum([len(label2utt_train[i]) for i in label2utt_train.keys()])
    no_test_utt = np.sum([len(label2utt_test[i]) for i in label2utt_test.keys()])
    
    return utt2label, label2utt_train, label2utt_test, no_classes, no_train_utt, no_test_utt
   

def get_args():
    """gets command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='data dir which should have tokenizer')
    parser.add_argument('no_epochs', type=int, help='no of epochs to train')
    parser.add_argument('expt_dir', help='expt dir path')
    parser.add_argument('no_classesPerBatch', type=int, help='no of classes to choose from for minibatch samples' )
    parser.add_argument('sampPerClass', type=int, help='no of samples per each class in the minibatch. sampPerClass should be greater than 2')
    parser.add_argument('maxlen', type=int, help='max no of words to consider in each document for training')
    parser.add_argument('max_no_filt', type=int, help='number of parallel conv layers')
    parser.add_argument('wt', type=int, help='verification loss factor. It is equal to lambda/minibatch_size')
    parser.add_argument('fold_no', type=int, help='CV fold no to be trained')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='1 if you want to see training progress for each minibatch, 2 if per epoch')

    parser.add_argument('--run_no', default=1, help='use if you want to run multiple times same expt in the same dir')
    args = parser.parse_args()
    return args


def main():
    '''
    Inputs: Obtained from get_args() function
    Outputs: Outputs trained models in expt_dir
    Format: 
    CUDA_VISIBLE_DEVICES=" " python train.py data_dir no_epochs expt_dir no_classesPerBatch sampPerClass maxlen max_no_filt wt fold_no -v verbose --run_no run_no    

    Inputs description:
        data_dir -- data dir which should have tokenizer
        no_epochs -- no of epochs to train 
        expt_dir -- expt dir path
        no_classesPerBatch -- no of classes to choose from for minibatch samples. It is just to control minibatch_size = no_classesPerBatch * sampPerClass. Choose no_classesPerBatch as high as possible and make sure minibatch size is managebale. 
        sampPerClass -- no of samples per each class in the minibatch. sampPerClass should be greater than 2
        maxlen - maximum length of document you want to consider. I used 1000  most of the times
        max_no_filt -- Number of parallel convolution layers   
        wt -- verification loss factor. It is equal to lambda/minibatch_size
        fold_no -- CV fold no to be trained. Used to get data for that CV fold
        verbose -- 1 if you want to see training progress for each minibatch, 2 if per epoch
        run_no -- use if you want to run multiple times same expt in the same dir. It is just an indicator to differentiate your stored models from each run
    ''' 
    args = get_args()

    print("args")
    if not os.path.isdir(args.expt_dir):
        os.makedirs(args.expt_dir)
    
    # data related parameters
    global total_no_samples 
    total_no_samples = args.no_classesPerBatch * args.sampPerClass # minibatch size
    
    # getting utt and label related info
    utt2label, label2utt_train, label2utt_test, no_classes, no_train_utt, no_test_utt = data_dict_gen(args.data_dir, args.fold_no)
    tokenizer = pickle.load(open(args.data_dir + '/tokenizer_None_CV_' + str(args.fold_no) + 'fold.pkl', 'rb'))
    
    global vocab_size, embed_size
    vocab_size, embed_size = len(tokenizer.word_index)+1, 300 
    
    # generators for trainig and validation data
    train_gen = allpairs_gen_joint(utt2label, label2utt_train, tokenizer, args.no_classesPerBatch, args.sampPerClass, no_classes)
    dev_gen = allpairs_gen_joint(utt2label, label2utt_test, tokenizer, args.no_classesPerBatch, args.sampPerClass, no_classes)
    steps_train = no_train_utt/total_no_samples
    steps_dev = no_test_utt/total_no_samples
    
    # Network parametwrs
    global MaxLen
    MaxLen = args.maxlen
    
    input_dim = MaxLen # input dim for word emebedding layer
    
    #global init
    #init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    
    # create model
    input_a = Input(shape=(input_dim,), name='input1')
    filt_width_spacing = 3
    no_filters = 10
    filt_models = []
    
    # creating all base modules and concatenating them
    for i in range(args.max_no_filt):
        filt_width = filt_width_spacing*i + 1
        print(filt_width) 
        base_op_stream = conv_filt_v2(input_a, no_filters, filt_width)
        filt_models.append(base_op_stream)
    
    if args.max_no_filt > 1:
        base_op = concatenate(filt_models)
    else:
        base_op = filt_models[0]
    
    # final base module encapsulated into one big model. It contains set of parallel base modules
    base_model = Model(input=[input_a],  output=[base_op]) 
    conv_processed_a = base_model(input_a)
    
    drop_layer = Dropout(0.3)
    conv_processed_a_drop = drop_layer(conv_processed_a)
    
    dense_layer_1 = Dense(30) 
    dense_processed_a = dense_layer_1(conv_processed_a_drop)
    dense_processed_a = Activation('relu')(dense_processed_a)
    
    dense_layer = Dense(no_classes)
    classif_processed_a = dense_layer(dense_processed_a)
    main_out_1 = Activation('softmax', name='main_output_1')(classif_processed_a)
    aux_output = Lambda(cosine_distance, output_shape=cos_dist_output_shape, name='aux_output')([dense_processed_a])
    model = Model(input=[input_a], output=[main_out_1, aux_output])
    
    # creating test model seperately here itself just because there is some error when I load "model" at test time
    test_model = Model(input=[input_a], output=[main_out_1])
    
    adagrad = Adagrad()
    
    period = 10  # how frequent you want to store the model. once in "period" epochs
    model.compile(loss={'main_output_1':'categorical_crossentropy', 'aux_output':Siamese_lik_ratio},
                optimizer=adagrad, metrics={'main_output_1':'accuracy'},
                loss_weights={'main_output_1':1, 'aux_output':args.wt})
    
    c1 = ModelCheckpoint(args.expt_dir + '/run_' + str(args.run_no) + 'SPOKEN_Siamese_' + str(args.wt) + '_foldno_' + str(args.fold_no) + '_{epoch:02d}-{val_loss:.2f}.hdf5',
                monitor='val_loss', verbose=0, save_best_only=False, mode='auto', period=period)
    
    # storing the model architecture
    yaml_string = model.to_yaml()
    with open(args.expt_dir + '/run_' + str(args.run_no) + 'model_SPOKEN_Siamese' +  '.yaml','w') as f:
        f.write(yaml_string)
    
    yaml_string_test = test_model.to_yaml() 
    with open(args.expt_dir + '/run_' + str(args.run_no) + 'test_model_SPOKEN_Siamese' +  '.yaml','w') as f:
        f.write(yaml_string_test)
    
    
    print(model.summary())
    # training the model
    History = model.fit_generator(train_gen, validation_data=dev_gen, validation_steps=steps_dev, steps_per_epoch=steps_train, epochs=args.no_epochs, max_queue_size=16, workers=2, initial_epoch=0, verbose=args.verbose, pickle_safe=True, callbacks=[c1])
    
    # storing the histroy of the model. Used ot check loss progress after training is done
    with open(args.expt_dir + '/run_' + str(args.run_no) + 'History_' + str(args.wt) + '.pkl', 'wb') as f:
        pickle.dump(History.history, f)
        #f = pickle.load(open(expt_dir + '/History_0.1.pkl','rb'))            
    
    #model.save_weights(expt_dir + '/run_' + str(run_no) + 'weights_SPOKEN_Siamese_' + str(wt) + '_foldno_' + str(fold_no) + '.h5')


if __name__ == "__main__":
    main()   



