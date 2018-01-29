from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras import backend as K
import scipy.io as sio
import numpy as np
import keras
import sys
import time
import h5py, pickle
import os
#from keras_tokenizer import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_tokenizer_original import *

'''
    Example script for creating tokenizer. I stored the data in CV_folds.pkl. 
    This pickle file contains train and test lists for each fold.
    data_dir --  data dir with CV_folds.pkl
    num_words -- number of words to be considered to tokenize. If you want to consider all words set it to "None"
    For details, refer https://keras.io/preprocessing/text/#tokenizer
'''

#data_dir = '/var/users/raghavendra/CSAT_scripts/data_CV/'
data_dir = sys.argv[1]
num_words = int(sys.argv[2])
transcripts_dir = '/var/users/raghavendra/CSAT_scripts/transcripts_mod/'

filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
#filters = ""

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

CV_folds = pickle.load(open(data_dir + '/CV_folds.pkl','rb'))      

for i in range(1,6):
    conversation = {}
    all_files = []
    for ind in CV_folds['train' + str(i)]:
        file_id = CV_folds['file_id'][int(ind)]
        with open(transcripts_dir + file_id + '.csv', 'r') as f:
            conversation = f.readlines()
            conversation = [sentence.strip() for sentence in conversation]
            conversation = ' '.join(conversation)
            all_files.append(conversation)
                
    tokenizer = Tokenizer(num_words=num_words, filters=filters,
                                       lower=True,
                                       split=" ",
                                       char_level=False)
    print(len(all_files)) 
    tokenizer.fit_on_texts(all_files)
    with open(data_dir +'tokenizer_' + str(num_words) + '_CV_' + str(i) + 'fold.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        #f = pickle.load(open(expt_dir + '/History_0.1.pkl','rb'))
