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
from sklearn.model_selection import cross_val_score
from sklearn import neighbors,linear_model, metrics, preprocessing, model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def cosine_distance(vects):
    x, y = vects
    print(x,y)
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    dot_prod = K.sum((x * y), axis=-1, keepdims=True)
    return K.sigmoid(dot_prod)
    #return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    #print("Shapes are {}".format(shapes))
    #print(shape1[0], 1)  
    return (shape1[0], 1)

def lik_ratio(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    print(y_true, y_pred)
    return K.mean(-(1-y_true) * K.log(y_pred+1e-20) - neg_loss_wt * (y_true) * K.log(1-y_pred+1e-20))


def lik_ratio_simi(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    print(y_true, y_pred)
    return K.mean(-(1-y_true) * K.log(y_pred+1e-20)) #- (y_true) * K.log(1-y_pred+1e-20))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 2
    print(y_true, y_pred)
    return K.mean((1 - y_true) * K.square(y_pred) +
                 neg_loss_wt * y_true * K.square(K.maximum(margin - y_pred, 0)))


def contrastive_loss_simi(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    return K.mean((1 - y_true) * K.square(y_pred))


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def conv_filt(input_dim, no_filters, filt_sizes, pool_sizes):

    filt_models = []
    strides_list = [1] * len(no_filters)
    print(no_filters, filt_sizes, strides_list)
    repeat_threshold = int(len(no_filters)/2)
    for filt_ind, filt in enumerate(no_filters):
        model = Sequential()
        model.add(Embedding(20000, 300, input_length=MaxLen))
        model.add(Conv1D(no_filters[filt_ind], filt_sizes[filt_ind], strides=strides_list[filt_ind]))
        model.add(Activation('relu'))
        if filt_ind < repeat_threshold:
            pool_sizes = 2
        else:
            pool_sizes = 7
        print(pool_sizes)
        model.add(AveragePooling1D(pool_size=pool_sizes))
        model.add(Dropout(0.5))
        #model.add(Conv1D(no_filters[filt_ind], filt_sizes[filt_ind], strides=1, dilation_rate=2))
        #model.add(Activation('relu'))
        #model.add(AveragePooling1D(pool_size=2))
        #model.add(Dropout(0.5))
        model.add(GlobalAveragePooling1D())
        filt_models.append(model)
    return filt_models


def concat_conv_filt(conv_net_filters, no_filters, input_a):

    conv_net_out_1 = []
    for i in range(len(no_filters)):
        processed_a = conv_net_filters[i](input_a)
        conv_net_out_1.append(processed_a)
    conv_processed_a = concatenate(conv_net_out_1)
    return conv_processed_a

def create_classif_network(no_classes, last_layer_neurons = 100):
    model_classif = Sequential()
    model_classif.add(Activation('linear', input_shape=(last_layer_neurons,)))
    model_classif.add(Dense(256, kernel_initializer=init))
    model_classif.add(Activation('relu'))
    model_classif.add(Dense(256, kernel_initializer=init))
    model_classif.add(Activation('relu'))
    model_classif.add(Dense(no_classes, kernel_initializer=init))
    return model_classif


def doc_represent_embedding(tokenizer, data):
    x = keras.preprocessing.text.text_to_word_sequence(data)
    #tokenizer.fit_on_texts(x)
    temp = tokenizer.texts_to_sequences(x)
    sequences = [list(np.concatenate(temp))]
    input_rep = pad_sequences(sequences, maxlen=MaxLen)
    input_rep = np.reshape(input_rep, (1,-1))
    return input_rep


def doc_represent_embedding_NEEDtoCHECK(tokenizer, data):
    #x = keras.preprocessing.text.text_to_word_sequence(data) 
    mode_ip_rep = 'count'
    mat = tokenizer.texts_to_matrix([data], mode=mode_ip_rep)
    input_rep = mat
    #input_rep = np.sum(mat, axis=0)
    input_rep = np.reshape(input_rep, (1,-1))
    return input_rep


def fisher_text_file_load(file_id, transcripts_dir='/var/users/raghavendra/CSAT_scripts/transcripts_mod/'):
    #file_id = str(file_id).zfill(5)
    #channel_id = ['A', 'B']
    #channel_id = np.random.choice(channel_id, 1)[0]
    #with open(transcripts_dir + file_id + '-' + channel_id + '.txt', 'r') as f: 
    with open(transcripts_dir + file_id + '.csv', 'r') as f:
        conversation = f.readlines()
        conversation = [sentence.strip() for sentence in conversation]
        conversation = ' '.join(conversation)
    return conversation


def allpairs_gen_joint(label2utt, no_classesPerBatch = 8, suffix = "", data_dir = 'data/', no_classes = 2, post_string='train'):   
    print('Loading data after randomizing')
    tokenizer = pickle.load(open(data_dir + '/tokenizer_None_CV_' + str(fold_no) + 'fold.pkl', 'rb'))
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


data_dir = sys.argv[1] # /var/users/raghavendra/CSAT_scripts/data/
no_epochs = int(sys.argv[2])
run_no = int(sys.argv[3])
expt_dir = sys.argv[4]
no_classesPerBatch = int(sys.argv[5])
maxlen = int(sys.argv[6]) #2713 
max_no_filt = int(sys.argv[7])
wt = float(sys.argv[8])
fold_no_arg = int(sys.argv[9])
TaskComp_flag = int(sys.argv[10])

global fold_no 
fold_no = fold_no_arg
global MaxLen
MaxLen = maxlen
global sampPerClass
sampPerClass = 4

no_samples = no_classesPerBatch * sampPerClass
batch_size_calc = no_samples * (no_samples-1)/2
pos_pairs = no_classesPerBatch * sampPerClass * (sampPerClass - 1)/2
neg_pairs = batch_size_calc - pos_pairs
global neg_loss_wt
neg_loss_wt = pos_pairs/neg_pairs
print("Neg Loss wt is {}".format(neg_loss_wt))

global init
init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)

if TaskComp_flag == 1:
    features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate','q3']
elif TaskComp_flag == 0:
    features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate']
else:
    print("Invalid argument for TaskComp_flag")
    sys.exit()

cuid =['cuid']
target = ['binaryNumber']
dataframe_features = pandas.read_csv("/var/users/purvak/csat/walgreen/features_target_balanced_v2.csv", usecols=features)
dataframe_cuid = pandas.read_csv("/var/users/purvak/csat/walgreen/features_target_balanced_v2.csv", usecols=cuid)
dataframe_target = pandas.read_csv("/var/users/purvak/csat/walgreen/features_target_balanced_v2.csv", usecols=target)
dataframe_cuid = dataframe_cuid.to_dict()['cuid']
utt2ind = dict(zip(dataframe_cuid.values(), dataframe_cuid.keys()))
audio_feat = dataframe_features.values
audio_feat = audio_feat.reshape(audio_feat.shape + (1,))
audio_feat_dim = audio_feat.shape[1]
#X_cuid = dataframe_cuid.values


CV_folds = pickle.load(open(data_dir + '/CV_folds.pkl','rb'))  
fild_id_list = CV_folds['file_id']

utt2label = {}
label2utt_train = {}
for i in range(2):
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
for i in range(2):     
    label2utt_test[str(i)] = []

for file_ind in CV_folds['test' + str(fold_no)]:
    file_id = fild_id_list[int(file_ind)]
    file_class = utt2label[file_id]
    label2utt_test[file_class].append(file_id)   

print(data_dir, run_no, expt_dir)

if not os.path.isdir(expt_dir):
    os.makedirs(expt_dir)

#input_dim, no_classes = dim_fisher()
no_classes = 2
input_dim = MaxLen
no_classes = int(no_classes)
print("Input dim and no classes are {}, {}".format(input_dim, no_classes))

#no_filters = list(range(3, max_no_filt, 1))
no_filters = []

max_no_filt = int(max_no_filt/2)
for i in range(max_no_filt):
    no_filters.append([3, 3])

no_filters = list(np.hstack(no_filters))
filt_sizes = list(range(1, 200, 3))
filt_sizes = filt_sizes[:max_no_filt]*2
pool_sizes = 10

# network definition
input_a = Input(shape=(input_dim,), name='input1')
input_b = Input(shape=(input_dim,), name='input2')
input_a_AF = Input(shape=(audio_feat_dim,1), name='input1_AF')
input_b_AF = Input(shape=(audio_feat_dim,1), name='input2_AF') 


conv_net_filters = conv_filt(input_dim, no_filters, filt_sizes, pool_sizes)
base_op = concat_conv_filt(conv_net_filters, no_filters, input_a)
base_model = Model(input=[input_a],  output=[base_op])

def AF_CNN(input_a_AF):
    out = Conv1D(20, 3, strides=1)(input_a_AF)
    out = Flatten()(out)
    out = Dense(30)(out)
    out = Activation('relu')(out)
    return out

#AF_dense = Dense(30)
#AF_processed_a = AF_dense(input_a_AF)
#AF_processed_b = AF_dense(input_b_AF)
#relu_layer = Activation('relu')
#AF_processed_a_relu = relu_layer(AF_processed_a)
#AF_processed_b_relu = relu_layer(AF_processed_b) 

conv_processed_a_text = base_model(input_a)
conv_processed_b_text = base_model(input_b)

AF_processed = AF_CNN(input_a_AF)
AF_CNN_model = Model(input=[input_a_AF], output=[AF_processed])
AF_processed_a_relu = AF_CNN_model(input_a_AF)
AF_processed_b_relu = AF_CNN_model(input_b_AF)

conv_processed_a = concatenate([conv_processed_a_text, AF_processed_a_relu])
conv_processed_b = concatenate([conv_processed_b_text, AF_processed_b_relu])

#drop_layer = Dropout(0.3)

#conv_processed_a_drop = drop_layer(conv_processed_b)
#conv_processed_b_drop = drop_layer(conv_processed_b)

dense_layer = Dense(no_classes)
classif_processed_a = dense_layer(conv_processed_a)
main_out_1 = Activation('softmax', name='main_output_1')(classif_processed_a)
classif_processed_b = dense_layer(conv_processed_b)
main_out_2 = Activation('softmax', name='main_output_2')(classif_processed_b)
aux_output = Lambda(cosine_distance, output_shape=eucl_dist_output_shape, name='aux_output')([conv_processed_a_text, conv_processed_b_text])

model = Model(input=[input_a, input_b, input_a_AF, input_b_AF], output=[main_out_1, main_out_2, aux_output])
#model = Model(input=[input_a], output=[main_out_1])

adagrad = Adagrad()

model.compile(loss={'main_output_1':'categorical_crossentropy', 'main_output_2':'categorical_crossentropy', 'aux_output':lik_ratio},
            optimizer=adagrad, metrics={'main_output_1':fmeasure, 'main_output_2':fmeasure},
            loss_weights={'main_output_1':1, 'main_output_2':1, 'aux_output':wt})

c1 = ModelCheckpoint(expt_dir + '/run_' + str(run_no) + 'SPOKEN_Siamese_' + str(wt) + '_foldno_' + str(fold_no) + '_{epoch:02d}-{val_main_output_2_fmeasure:.2f}.hdf5',
            monitor='val_main_output_2_fmeasure', verbose=0, save_best_only=False, mode='auto', period=3)

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)

yaml_string = model.to_yaml()
with open(expt_dir + '/run_' + str(run_no) + 'model_SPOKEN_Siamese' +  '.yaml','w') as f:
    f.write(yaml_string)


suffix = ""
train_gen = allpairs_gen_joint(label2utt_train, no_classesPerBatch, suffix, data_dir, no_classes, post_string='train')
dev_gen = allpairs_gen_joint(label2utt_test, no_classesPerBatch, suffix, data_dir, no_classes, post_string='dev')

steps_train = 120
steps_dev = 50
print(model.summary())

History = model.fit_generator(train_gen, validation_data=dev_gen, validation_steps=steps_dev, steps_per_epoch=steps_train, epochs=no_epochs, max_queue_size=64, workers=5, initial_epoch=0, verbose=2, pickle_safe=True, callbacks=[c1])

with open(expt_dir + '/run_' + str(run_no) + 'History_' + str(wt) + '.pkl', 'wb') as f:
    pickle.dump(History.history, f)
    #f = pickle.load(open(expt_dir + '/History_0.1.pkl','rb'))            

model.save_weights(expt_dir + '/run_' + str(run_no) + 'weights_SPOKEN_Siamese_' + str(wt) + '_foldno_' + str(fold_no) + '.h5')




