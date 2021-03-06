from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv1D, Dense, MaxPooling1D, Flatten, Activation
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from keras import backend as K
import numpy as np
import keras
import sys
import time
import h5py, pickle
from keras.models import model_from_yaml
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from keras.preprocessing.text import  text_to_word_sequence
import pandas
from sklearn.model_selection import cross_val_score
from hdf_utils import *
from utils_TextOnly import *
import argparse


### audio_data_load(uttlist_train, uttlist_test) is extra function from earlier version evaluate_FUSE_AF_git.py ###


def allpairs_gen_joint(tokenizer, test_uttlist, utt2label, no_classes, transcripts_dir, MaxLen):
    ''' Generates data to be used for feed forwarding
    Inputs:
    tokenizer -- tokenizer used in the training stage
    test_uttlist -- utterance list to be evaluated
    utt2label -- utterance to label maaping dictionary
    no_classes -- number of possible classes for each utterance
    transcripts_dir -- dir where transcripts exists
    MaxLen -- maximum number of words considered for training 

    Outputs: it yields the batch data (features and labels) in a format required for DNN
    '''
   
     
    batch_no = 0    
    while 1:
        # getting utterance list for each batch
        if (batch_no+1) * batch_size < len(test_uttlist):
            samples = test_uttlist[batch_no * batch_size: (batch_no+1) * batch_size]
        else:
            print('last batch')
            samples = test_uttlist[batch_no * batch_size:]
             
        # indices of set of samples in the batch with their labels
        batch_ind_set = np.array([[i, utt2label[i]] for i in samples])

        # data at those indices
        batch_data = np.vstack([doc_represent_embedding(tokenizer, text_file_load(utt, transcripts_dir), MaxLen) for utt in batch_ind_set[:, 0]])
        #batch_data_audio_feat = np.array([f_AF[utt] for utt in batch_ind_set[:, 0]])
        #batch_data_audio_feat = np.array([audio_feat_test_norm_dict[utt] for utt in batch_ind_set[:, 0]])

        batch_labels = np.vstack(keras.utils.to_categorical(int(j), no_classes) for j in batch_ind_set[:, 1])
        yield({'input1':batch_data,},{'main_output_1':batch_labels})
        #yield({'input1':batch_data, 'input1_AF':batch_data_audio_feat},{'main_output_1':batch_labels}) 

        batch_no += 1


def data_dict_gen(data_dir, no_classes, fold_no):
    ''' Goal of this function is to return utt2label, label2utt_test, test_uttlist. 
        Probably you may need to write your own function depending on your data prganization preference

        Task: creating utterance lists and dictionaries to map utterance to labels. Used when you are testing on existing data. We can remove this function later if we do not need. I kept it to test on training data also

        Input arguments and output argumnets are self-explanatory. 
    Inputs:
        data_dir -- data directory where CV_folds.pkl, file_id_labels.txt exist
        no_classes -- Possible number of classes each document can be classified into 
        fold_no -- Cross validation fold no to test on. It is used to get utterance list
    outputs:
        utt2label -- dictiionary with utterances as keys and labels as values, contains train+test
        label2utt_train -- dictiionary with labels as keys and utterances as values, only for train uttearances
        label2utt_test -- same as label2utt_train but only for test utterances
        test_uttlist -- list of test utterances
     '''
    CV_folds = pickle.load(open(data_dir + '/CV_folds.pkl','rb'))
    fild_id_list = CV_folds['file_id']
    
    # initializing to empty list for each class    
    utt2label = {}
    label2utt_train = {}
    for i in range(no_classes):
        label2utt_train[str(i)] = []

    # itearing through each line of utterance list 
    with open(data_dir + '/file_id_labels.txt', 'r') as f:
        for i in f.readlines():
            file_id = i.strip().split()[0]
            file_class = i.strip().split()[1]
            utt2label[file_id] = file_class

    train_uttlist = []     
    for file_ind in CV_folds['train' + str(fold_no)]:
        file_id = fild_id_list[int(file_ind)]
        file_class = utt2label[file_id]
        label2utt_train[file_class].append(file_id)
        train_uttlist.append(file_id)
    
    label2utt_test = {}
    for i in range(no_classes):
        label2utt_test[str(i)] = []
    
    test_uttlist = []
    for file_ind in CV_folds['test' + str(fold_no)]:
        file_id = fild_id_list[int(file_ind)]
        file_class = utt2label[file_id]
        label2utt_test[file_class].append(file_id)
        test_uttlist.append(file_id)
    return utt2label, label2utt_train, label2utt_test, train_uttlist, test_uttlist


def data_dict_gen_from_file(data_dir, no_classes):
    '''  It does create utterance lists and dictionaries to map utterance to labels 
    Inputs: 
        data_dir -- data directory where utt2label_test.txt will exist
        no_classes -- Possible number of classes each document can be classified into
    Outputs:
        utt2label -- dictiionary with utterances as keys and labels as values
        label2utt_test -- dictiionary with labels as keys and utterances as values (opposite of utt2label)
        test_uttlist -- list of utterances equal to utt2label.keys()
    '''
    
    utt2label = {}
    label2utt_test = {}  
    test_uttlist = []  
    # initializing to empty list for each class
    for i in range(no_classes):
        label2utt_test[str(i)] = []

    # itearing through each line of utterance list     
    with open(data_dir + '/utt2label_test.txt', 'r') as f:
        for i in f.readlines(): 
            file_id = i.strip().split()[0] 
            file_class = i.strip().split()[1] 
            utt2label[file_id] = file_class 
            label2utt_test[file_class].append(file_id)        
            test_uttlist.append(file_id)
    return utt2label, label2utt_test, test_uttlist


def get_args():
    """gets command line arguments"""

    '''Run from /mnt/NFS.cltlabnas1vg0/users/raghavendra/GitHub_version_TextOnly/  (path in AWS, if it doesn't run, try copying my /home/raghavendra/.bashrc and try it)

        example usage: CUDA_VISIBLE_DEVICES=" " python evaluate_Text_git.py  /mnt/NFS.cltlabnas1vg0/users/raghavendra/GitHub_version_TextOnly/sample_model/run_1test_model_SPOKEN_Siamese.yaml /mnt/NFS.cltlabnas1vg0/users/raghavendra/GitHub_version_TextOnly/sample_model/run_1SPOKEN_Siamese_2.5_foldno_5_40-0.79.hdf5 data_CV/ temp/ 2.5 temp 5 --new_transcripts 0 -tr_dir  //mnt/NFS.cltlabnas1vg0/users/raghavendra/Siamese_expts/transcripts_mod/ 
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('arch_file', help='architecture file. Its is yaml file stored in training stage. ')
    parser.add_argument('model_path', help='model file (.hdf5 file). It has weights')
    parser.add_argument('data_dir', type=str, help='data dir')
    parser.add_argument('out_dir', help='dir to store output result')
    parser.add_argument('wt', type=float, help='wt factor for verification loss function. Used only to store the results file')
    parser.add_argument('suffix', help='suffix to be added to results file')
    parser.add_argument('fold_no', type=int, help='CV fold no to be evalauted')

    parser.add_argument('-n', '--new_transcripts', type=int, default=0, help='If True then make sure utt2label_test.txt exists in data dir. For format look at data_dir/file_id_labels.txt')
    parser.add_argument('-tr_dir', '--transcripts_dir', type=str, default='/var/users/raghavendra/CSAT_scripts/transcripts_mod/',
                    help='dir for text data')
    parser.add_argument('-p', '--post_string', default='test', help='data set to be evaluated')
    #parser.add_argument('--verbose', type=int, default=0,
    #                    help="Not used.")

    args = parser.parse_args()
    return args

def main():
    ''' 
    This function loads the supplied neural network model and evalautes a list of utterances         
    Inputs: Obtained from get_args() function
    Outputs: Outputs accuracy and f-score to out_dir
    
    Format: 
    CUDA_VISIBLE_DEVICES=" " python ${src_dir}/evaluate_Text_git.py arch_file model_path data_dir  out_dir weight suffix fold_no -n new_transcripts -tr_dir transcripts_dir -p post_string

    Inputs description: 
        arch_file -- model architecture used for training
        model_path -- model weights stored in training
        data_dir -- Path for data dir which should contain tokenizer and utt2label_test.txt
        out_dir -- dir to store the results file
        weight -- Constant factor of verification loss, not used in any where except to store the result. Technically, can    
                    be given any value at the time of evaluation.
        suffix -- suffix need to be added to result file name. Again, can be given any string.

        fold_no -- Cross validation fold number. Tokenizer will be selected based on this, model_path and arch_file
                        should also be correspong to this fold number
 
        (-n)--new_transcripts -- boolean, set to 1 if you are using new transcripts, otherwise set to 0 (default)
        (-tr_dir)--transcripts_dir -- path for transcripts
        (-p)--post_string -- data set to be evaluated. Can take "train", "test"(default)
   
    '''
    args = get_args()

    # paralemeters to be set
    MaxLen = 1000  # max number of words used in training stage. Mostly it would be 1000 always
    global batch_size 
    batch_size = 64  # can be set to different nuimber if you want

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    
    # loading trainig model and creating new model as per requirements for evaluation
    print("arch file is {}".format(args.arch_file))
    print("Model path is {}".format(args.model_path))
    model = user_load_model(args.arch_file)
    model.load_weights(args.model_path)
    test_model = model
    no_classes = test_model.output_shape[1]
    print("Original model is ")
    print(model.summary())
    print("Test model is ")
    print(test_model.summary())
    print(args.new_transcripts) 

    # creating utterance list and mapping to corresponding labels 
    if args.new_transcripts == 0:
        utt2label, label2utt_train, label2utt_test, train_uttlist, test_uttlist = data_dict_gen(args.data_dir, no_classes, args.fold_no)
    else:
        utt2label, label2utt_test, test_uttlist = data_dict_gen_from_file(args.data_dir, no_classes)
   
    #audio_data_load(train_uttlist, test_uttlist) 
    test_label = []
    test_p = []
    print("Post_string is {}".format(args.post_string))

    # loading tokenizer used at the time of training
    tokenizer = pickle.load(open(args.data_dir + '/tokenizer_None_CV_' + str(args.fold_no) + 'fold.pkl', 'rb'))

    # data generator for test data
    data_gen = allpairs_gen_joint(tokenizer, test_uttlist, utt2label, no_classes = no_classes, transcripts_dir=args.transcripts_dir, MaxLen= MaxLen)

    # calculating number of batches. It is equal to number of times we feed forward the data through DNN
    no_lines = len(test_uttlist)
    no_steps = int(np.ceil(no_lines/batch_size))
    print("no_steps are {}".format(no_steps))

    # generating data and feed forwarding to predicting labels
    for i in range(no_steps):
        pair = next(data_gen)
        x = test_model.predict(pair[0])
        temp = np.argmax(x, axis=1)
        test_p.append(temp)
        temp = np.argmax(pair[1]['main_output_1'], axis=1)
        test_label.append(temp)
    test_label = np.hstack(test_label)
    test_p = np.hstack(test_p)

    # calculate the accuracy
    test_acc = np.mean(test_label == test_p) * 100.0
    print("Number of samples processed is {}".format(len(test_p)))
    
    f1_score = metrics.f1_score(test_p, test_label, average='macro')
    #precision, recall, fscore, support = score(test_label, test_p)
    #print("Train Acc is {}".format(train_acc))
    print("Test Acc and f1-score are {}, {}".format(test_acc, f1_score))
    #print("precision, recall, fscore, support are {}, {}, {}, {}".format( precision, recall, fscore, support))
    with open(args.out_dir + '/result_' + args.suffix + '.txt', 'a') as f:
        f.write(str(args.fold_no) + '\t' + str(args.wt) + '\t' +   str(test_acc) + '\t'  +  str(f1_score) + '\n' )


if __name__ == "__main__":
    main()

