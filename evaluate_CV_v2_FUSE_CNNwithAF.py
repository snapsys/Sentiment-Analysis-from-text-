from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv1D, Dense, MaxPooling1D, Flatten, Activation, TimeDistributed
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
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
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras.models import model_from_yaml
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from keras.preprocessing.text import  text_to_word_sequence

import pandas
from sklearn.model_selection import cross_val_score


def load_model(arch_file):
    #arch_file = "joint_verif_ident_expts/wt_expts_AEstyle_linearlayer_extraConvReluLayer_dropout/run_3model_fisher_Siamese_0.7.yaml"
    yaml_file = open(arch_file, 'r')
    yaml_string = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(yaml_string)
    return model

def doc_represent_embedding(tokenizer, data):
    x = text_to_word_sequence(data)
    #tokenizer.fit_on_texts(x)
    temp = tokenizer.texts_to_sequences(x)
    sequences = [list(np.concatenate(temp))]
    input_rep = pad_sequences(sequences, maxlen=MaxLen)
    input_rep = np.reshape(input_rep, (1,-1))
    return input_rep

def fisher_text_file_load(file_id, transcripts_dir='/var/users/raghavendra/CSAT_scripts/transcripts_mod/'):
    with open(transcripts_dir + file_id + '.csv', 'r') as f:
        conversation = f.readlines()
        conversation = [sentence.strip() for sentence in conversation]
        conversation = ' '.join(conversation)
    return conversation


def allpairs_gen_joint(utt2label, features, data_dir = 'data/', no_classes = 2, post_string='train'):

    print('Loading data after randomizing')
    tokenizer = pickle.load(open(data_dir + '/tokenizer_None_CV_' + str(fold_no) + 'fold.pkl', 'rb'))
    cuid =['cuid']
    target = ['binaryNumber']
    AF_data_csv = "/var/users/purvak/csat/walgreen/features_target_balanced_v2.csv"
    dataframe_features = pandas.read_csv(AF_data_csv, usecols=features)
    dataframe_cuid = pandas.read_csv(AF_data_csv, usecols=cuid)
    dataframe_target = pandas.read_csv(AF_data_csv, usecols=target)
    dataframe_cuid = dataframe_cuid.to_dict()['cuid']
    utt2ind = dict(zip(dataframe_cuid.values(), dataframe_cuid.keys()))
    audio_feat = dataframe_features.values
    audio_feat = audio_feat.reshape(audio_feat.shape + (1,))     

    task = 'verif_test'
    batch_no = 0    
    while 1:
        k = 0
        if (batch_no+1) * batch_size < len(test_uttlist):
            samples = test_uttlist[batch_no * batch_size: (batch_no+1) * batch_size]
        else:
            print('last batch')
            samples = test_uttlist[batch_no * batch_size:]
             
        # indices of set of samples in the batch with their labels
        batch_ind_set = np.array([[i, utt2label[i]] for i in samples])

        # data at those indices
        batch_data = np.vstack([doc_represent_embedding(tokenizer, fisher_text_file_load(utt)) for utt in batch_ind_set[:, 0]])
        batch_data_audio_feat = np.array([audio_feat[utt2ind[utt]] for utt in batch_ind_set[:, 0]])

        if task == 'classification':
            batch_labels = np.vstack(keras.utils.to_categorical(int(j), no_classes) for j in batch_ind_set[:, 1])
            yield({'input1':batch_data},{'main_output_1':batch_labels})
        elif task == 'verif_test':
            batch_labels = np.vstack(keras.utils.to_categorical(int(j), no_classes) for j in batch_ind_set[:, 1])
            input1_batch_AF = batch_data_audio_feat
            yield({'input1':batch_data, 'input1_AF':input1_batch_AF,},{'main_output_1':batch_labels}) 
            #yield({'input1':batch_data, 'input2':batch_data},{'main_output_1':batch_labels, 'main_output_2':batch_labels}) 

        else:
            pairs_labels = np.vstack(np.array([1*(i1!=i2), int(i1), int(i2)]) for j,i1 in enumerate(batch_ind_set[:, 1]) for i2 in batch_ind_set[j+1:,1] )
            ind_local = range(batch_ind_set.shape[0])
            pairs_local = np.vstack(np.array([i1,i2]) for j,i1 in enumerate(ind_local) for i2 in ind_local[j+1:])
            input1_batch = batch_data[pairs_local[:,0]]
            input2_batch = batch_data[pairs_local[:,1]]
            label1_batch = np.vstack( keras.utils.to_categorical(j, no_classes) for j in pairs_labels[:, 1] )
            label2_batch = np.vstack( keras.utils.to_categorical(j, no_classes) for j in pairs_labels[:, 2] )
            yield({'input1':input1_batch, 'input2':input2_batch}, {'main_output_1':label1_batch, 'main_output_2':label2_batch})

        batch_no += 1


def sub_model_v2(input_a, input_a_AF, model, layer_ind):
    out_AF = model.layers[5](input_a_AF)
    #out_AF = model.layers[6](out_AF)

    out_word = model.layers[4](input_a)
    out = model.layers[6]([out_word, out_AF])

    out = model.layers[8](out)
    out = model.layers[9](out)

    #for layer in layer_ind:
    #    out = model.layers[int(layer)](input_a)
    #    input_a = out
    return out

def data_dict_gen(data_dir):
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
    
    test_uttlist = []
    for file_ind in CV_folds['test' + str(fold_no)]:
        file_id = fild_id_list[int(file_ind)]
        file_class = utt2label[file_id]
        label2utt_test[file_class].append(file_id)
        test_uttlist.append(file_id)
    return utt2label, label2utt_train, label2utt_test, test_uttlist


def main():
    arch_file = sys.argv[1]  # architecture file. Its is yaml file stored in training stage.
    model_path = sys.argv[2] # model file (.hdf5 file). It has weights
    data_dir = sys.argv[3]  
    post_string = str(sys.argv[4]) # here it is "test"
    out_dir = sys.argv[5]   # 
    print(arch_file, model_path, data_dir, post_string, out_dir)
    wt = float(sys.argv[6])  # wt factor for verification loss function. Used only to store the results file
    fold_no_arg = int(sys.argv[7]) # cross validation fold no
    suffix = sys.argv[8] # suffix to give to results file
    layer_ind = sys.argv[9].split(',') # right now not used. Used to control feature extraction
    print(layer_ind)
    task_comp_flag = int(sys.argv[10]) # task completion flag

    global fold_no
    fold_no = fold_no_arg
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    print("arch file is {}".format(arch_file))
    print("Model path is {}".format(model_path))

    global test_uttlist
    utt2label, label2utt_train, label2utt_test, test_uttlist = data_dict_gen(data_dir) 

    model = load_model(arch_file)
    model.load_weights(model_path)
    global MaxLen
    MaxLen = 1000
    global batch_size
    batch_size = 64
    test_label = []
    test_p = []
    print("Post_string is {}".format(post_string))

    if task_comp_flag == 0:
        features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate']
    elif task_comp_flag == 1:
        features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate', 'q3']
    elif task_comp_flag == 2: 
        features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate','q3_1','q3_2','q3_3']

    data_gen = allpairs_gen_joint(utt2label, features, data_dir = data_dir, no_classes = 2, post_string='train')
    #no_lines = file_line_count(data_dir + '/' + post_string + '.txt')
    no_lines = len(test_uttlist)
    no_steps = int(np.ceil(no_lines/batch_size))

    audio_feat_dim = len(features)
    input_dim = MaxLen
    input_a = Input(shape=(input_dim,), name='input1')
    input_a_AF = Input(shape=(audio_feat_dim, 1), name='input1_AF')

    d2 = sub_model_v2(input_a, input_a_AF, model, layer_ind)
    test_model = Model(input=[input_a, input_a_AF], output=[d2])  
    print("Original model is ")
    print(model.summary())
    print("Test model is ")
    print(test_model.summary())
    print("no_steps are {}".format(no_steps))
    for i in range(no_steps):
        pair = next(data_gen)
        #feat = np.vstack(pair[0])
        #label = np.vstack(pair[1])
        #print(feat.shape, label)    
        x = test_model.predict(pair[0])
        temp = np.argmax(x, axis=1)
        test_p.append(temp)
        temp = np.argmax(pair[1]['main_output_1'], axis=1)
        test_label.append(temp)
    
    test_label = np.hstack(test_label)
    test_p = np.hstack(test_p)
    test_acc = np.mean(test_label == test_p) * 100.0
    print("Number of samples processed is {}".format(len(test_p)))
    
    f1_score_20news = metrics.f1_score(test_p, test_label, average='macro')
    precision, recall, fscore, support = score(test_label, test_p)
    #print("Train Acc is {}".format(train_acc))
    print("Test Acc and f1-score are {}, {}".format(test_acc, f1_score_20news))
    print("precision, recall, fscore, support are {}, {}, {}, {}".format( precision, recall, fscore, support))
    res_dict = {}
    res_dict['precision'] = precision
    res_dict['recall'] = recall
    res_dict['fscore'] = fscore
    res_dict['support'] = support
    
    #np.savetxt('GT_fisher.txt',test_label)
    #np.savetxt('pred_labels_fisher.txt',test_p)
    with open(out_dir + '/result_' + suffix + '.txt', 'a') as f:
        f.write(str(fold_no) + '\t' + str(wt) + '\t' +   str(test_acc) + '\t'  +  str(f1_score_20news)+ '\t'  +  str(fscore[0]) + '\t' + str(fscore[1]) + '\n' )


if __name__ == "__main__":
    main()





