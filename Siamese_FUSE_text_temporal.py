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
import pandas
#from sklearn.model_selection import cross_val_score
#from sklearn import neighbors,linear_model, metrics, preprocessing, model_selection
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import accuracy_score
from utils import *
import global_vars


def conv_filt(input_dim, no_filters, filt_sizes):
    ''' returns a list of parallel convolutonal layers. We will concatenate them in concat_conv_filt() '''
    filt_models = []
    print(no_filters, filt_sizes)
    repeat_threshold = int(len(no_filters)/2)
    for filt_ind, filt in enumerate(no_filters):
        model = Sequential()
        # Anything more than vocabulary size to be considered is fine. I used 20000 here
        model.add(Embedding(20000, 300, input_length=global_vars.MaxLen))
        model.add(Conv1D(no_filters[filt_ind], filt_sizes[filt_ind], strides=1))
        model.add(Activation('relu'))
        if filt_ind < repeat_threshold:
            pool_sizes = 2
        else:
            pool_sizes = 7
        print(pool_sizes)
        model.add(AveragePooling1D(pool_size=pool_sizes))
        model.add(Dropout(0.5))
        model.add(GlobalAveragePooling1D())
        filt_models.append(model)
    return filt_models


def concat_conv_filt(conv_net_filters, no_filters, input_a):
    ''' concatenates a list of parallel layers '''
    conv_net_out_1 = []
    for i in range(len(no_filters)):
        processed_a = conv_net_filters[i](input_a)
        conv_net_out_1.append(processed_a)
    conv_processed_a = concatenate(conv_net_out_1)
    return conv_processed_a


def AF_CNN(input_a_AF):
    ''' convoltional layer for temporal features. AF meaning Audio features '''
    out = Conv1D(20, 3, strides=1)(input_a_AF)
    out = Flatten()(out)
    out = Dense(30)(out)
    out = Activation('relu')(out)
    return out


def allpairs_gen_joint(utt2label, label2utt, features, no_classesPerBatch = 2, data_dir = 'data/', no_classes = 2, post_string='train'):   
    print('Loading data after randomizing')
    tokenizer = pickle.load(open(data_dir + '/tokenizer_None_CV_' + str(fold_no) + 'fold.pkl', 'rb'))
    cuid =['cuid']
    target = ['binaryNumber']
    dataframe_features = pandas.read_csv("/var/users/purvak/csat/walgreen/features_target_balanced_v2.csv", usecols=features)
    dataframe_cuid = pandas.read_csv("/var/users/purvak/csat/walgreen/features_target_balanced_v2.csv", usecols=cuid)
    dataframe_target = pandas.read_csv("/var/users/purvak/csat/walgreen/features_target_balanced_v2.csv", usecols=target)
    dataframe_cuid = dataframe_cuid.to_dict()['cuid']
    utt2ind = dict(zip(dataframe_cuid.values(), dataframe_cuid.keys()))
    audio_feat = dataframe_features.values
    audio_feat = audio_feat.reshape(audio_feat.shape + (1,))

    task = 'verification'
    while 1:
        #sampPerClass = 4
        #no_classesPerBatch = 8
        classesPerBatch = np.random.choice(range(no_classes), size=no_classesPerBatch)
        samples = []
        for i in classesPerBatch:
            samples.append(np.random.choice(label2utt[str(i)], size=sampPerClass))
        samples = np.concatenate(samples)   # indices of set of samples in the batch

        # indices of set of samples in the batch with their labels
        batch_ind_set = np.array([[i, utt2label[i]] for i in samples])        

        # data at those indices
        batch_data = np.vstack([doc_represent_embedding(tokenizer, fisher_text_file_load(utt)) for utt in batch_ind_set[:, 0]])
        batch_data_audio_feat = np.array([audio_feat[utt2ind[utt]] for utt in batch_ind_set[:, 0]]) 

        if task == 'classification':
            batch_labels = np.vstack(keras.utils.to_categorical(int(j), no_classes) for j in batch_ind_set[:, 1])
            yield({'input1':batch_data},{'main_output_1':batch_labels})
        else:
            pairs_labels = np.vstack(np.array([1*(i1!=i2), int(i1), int(i2)]) for j,i1 in enumerate(batch_ind_set[:, 1]) for i2 in batch_ind_set[j+1:,1] )
            ind_local = range(batch_ind_set.shape[0])
            pairs_local = np.vstack(np.array([i1,i2]) for j,i1 in enumerate(ind_local) for i2 in ind_local[j+1:])

            input1_batch = batch_data[pairs_local[:,0]]
            input2_batch = batch_data[pairs_local[:,1]]

            label_aux_batch = pairs_labels[:, 0]
            label1_batch = np.vstack( keras.utils.to_categorical(j, no_classes) for j in pairs_labels[:, 1] )
            label2_batch = np.vstack( keras.utils.to_categorical(j, no_classes) for j in pairs_labels[:, 2] )

            input1_batch_AF = batch_data_audio_feat[pairs_local[:,0]]
            input2_batch_AF = batch_data_audio_feat[pairs_local[:,0]] 

            yield({'input1':input1_batch, 'input2':input2_batch, 'input1_AF':input1_batch_AF, 'input2_AF':input2_batch_AF}, {'main_output_1':label1_batch, 'main_output_2':label2_batch, 'aux_output':label_aux_batch})


def data_dict_gen(data_dir, no_classes):
    ''' Creates dictionaries required to map labels and uterances '''

    CV_folds = pickle.load(open(data_dir + '/CV_folds.pkl','rb'))  
    fild_id_list = CV_folds['file_id']
    
    utt2label = {}
    label2utt_train = {}
    for i in range(no_classes):
        label2utt_train[str(i)] = []
    
    with open(data_dir + '/file_id_labels.txt', 'r') as f:
        for i in f.readlines():
            file_id = i.strip().split()[0]
            file_class = i.strip().split()[1]
            utt2label[file_id] = file_class
    
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

    return utt2label, label2utt_train, label2utt_test


def main():
    data_dir = sys.argv[1] # /var/users/raghavendra/CSAT_scripts/data/
    no_epochs = int(sys.argv[2]) 
    run_no = int(sys.argv[3])  # useful if multiple runs need to be run for the same experiment. If it is first time, it is 1
    expt_dir = sys.argv[4]
    no_classesPerBatch = int(sys.argv[5]) # number of classes from which batch samples are sampled. Always > 2
    maxlen = int(sys.argv[6]) # # maximum number of words to consider for a document. We are using 1000 so there is a hope for improvement in performance here.
    max_no_filt = int(sys.argv[7])  # number of parllel layers to have in CNN arch
    wt = float(sys.argv[8])  # wt factor for verification loss function
    fold_no_arg = int(sys.argv[9])  # cross-validation fold no. It is 1,2,3,4,5
    TaskComp_flag = int(sys.argv[10]) # 1 if task completion is included in feature set otherwise 0
    
    global fold_no 
    fold_no = fold_no_arg
    global MaxLen
    MaxLen = maxlen 
    global sampPerClass
    sampPerClass = 4 # number of samples per class in a batch

    ## weighting factor calculation in verification loss function    
    no_samples = no_classesPerBatch * sampPerClass
    batch_size_calc = no_samples * (no_samples-1)/2
    pos_pairs = no_classesPerBatch * sampPerClass * (sampPerClass - 1)/2
    neg_pairs = batch_size_calc - pos_pairs
    global_vars.neg_loss_wt = pos_pairs/neg_pairs
    
    global init
    init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    
    no_classes = 2
    input_dim = MaxLen
    print("Input dim and no classes are {}, {}".format(input_dim, no_classes))
    
    if TaskComp_flag == 1:
        features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate','q3']
    elif TaskComp_flag == 0:
        features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate']
    else:
        print("Invalid argument for TaskComp_flag")
        sys.exit()
    
    utt2label, label2utt_train, label2utt_test = data_dict_gen(data_dir, no_classes)
    print(data_dir, run_no, expt_dir)
    
    if not os.path.isdir(expt_dir):
        os.makedirs(expt_dir)
    
    no_filters = []
    max_no_filt = int(max_no_filt/2)
    for i in range(max_no_filt):
        no_filters.append([3, 3])
    
    no_filters = list(np.hstack(no_filters))
    filt_sizes = list(range(1, 200, 3))
    filt_sizes = filt_sizes[:max_no_filt]*2

    audio_feat_dim = len(features)   
    # network definition
    input_a = Input(shape=(input_dim,), name='input1')
    input_b = Input(shape=(input_dim,), name='input2')
    input_a_AF = Input(shape=(audio_feat_dim,1), name='input1_AF')
    input_b_AF = Input(shape=(audio_feat_dim,1), name='input2_AF') 
    
    conv_net_filters = conv_filt(input_dim, no_filters, filt_sizes)
    base_op = concat_conv_filt(conv_net_filters, no_filters, input_a)
    base_model = Model(input=[input_a],  output=[base_op])
    
    conv_processed_a_text = base_model(input_a)
    conv_processed_b_text = base_model(input_b)
    
    AF_processed = AF_CNN(input_a_AF)
    AF_CNN_model = Model(input=[input_a_AF], output=[AF_processed])
    AF_processed_a_relu = AF_CNN_model(input_a_AF)
    AF_processed_b_relu = AF_CNN_model(input_b_AF)
    
    conv_processed_a = concatenate([conv_processed_a_text, AF_processed_a_relu])
    conv_processed_b = concatenate([conv_processed_b_text, AF_processed_b_relu])
    
    dense_layer = Dense(no_classes)
    classif_processed_a = dense_layer(conv_processed_a)
    main_out_1 = Activation('softmax', name='main_output_1')(classif_processed_a)
    classif_processed_b = dense_layer(conv_processed_b)
    main_out_2 = Activation('softmax', name='main_output_2')(classif_processed_b)
    aux_output = Lambda(cosine_distance, output_shape=cos_dist_output_shape, name='aux_output')([conv_processed_a_text, conv_processed_b_text])
    
    model = Model(input=[input_a, input_b, input_a_AF, input_b_AF], output=[main_out_1, main_out_2, aux_output])
    
    adagrad = Adagrad()
    model.compile(loss={'main_output_1':'categorical_crossentropy', 'main_output_2':'categorical_crossentropy', 'aux_output':lik_ratio},
                optimizer=adagrad, metrics={'main_output_1':'accuracy', 'main_output_2':'accuracy'},
                loss_weights={'main_output_1':1, 'main_output_2':1, 'aux_output':wt})
    
    c1 = ModelCheckpoint(expt_dir + '/run_' + str(run_no) + 'SPOKEN_Siamese_' + str(wt) + '_foldno_' + str(fold_no) + '_{epoch:02d}-{val_main_output_2_fmeasure:.2f}.hdf5',
                monitor='val_main_output_2_acc', verbose=0, save_best_only=False, mode='auto', period=3)
    
    yaml_string = model.to_yaml()
    with open(expt_dir + '/run_' + str(run_no) + 'model_SPOKEN_Siamese' +  '.yaml','w') as f:
        f.write(yaml_string)
    
    train_gen = allpairs_gen_joint(utt2label, label2utt_train, features, no_classesPerBatch, data_dir, no_classes, post_string='train')
    dev_gen = allpairs_gen_joint(utt2label, label2utt_test, features, no_classesPerBatch, data_dir, no_classes, post_string='dev')
    
    steps_train = 120
    steps_dev = 50
    print(model.summary())
    
    History = model.fit_generator(train_gen, validation_data=dev_gen, validation_steps=steps_dev, steps_per_epoch=steps_train, epochs=no_epochs, max_queue_size=64, workers=5, initial_epoch=0, verbose=1, pickle_safe=True, callbacks=[c1])
    
    with open(expt_dir + '/run_' + str(run_no) + 'History_' + str(wt) + '.pkl', 'wb') as f:
        pickle.dump(History.history, f)
        #f = pickle.load(open(expt_dir + '/History_0.1.pkl','rb'))            
    
    model.save_weights(expt_dir + '/run_' + str(run_no) + 'weights_SPOKEN_Siamese_' + str(wt) + '_foldno_' + str(fold_no) + '.h5')


if __name__ == "__main__":    
    main() 


