import keras
from keras import backend as K
from keras.models import model_from_yaml
import numpy as np
from keras.preprocessing.sequence import pad_sequences 
import global_vars

def lik_ratio(y_true, y_pred):
    ''' computes likelihood ratio between predictions and true labels. Did not put in utils.py because I am using neg_loss_wt as global         variable   
    Inputs: Ground truth label and predicted label
    Output: returns the verification loss (liklihood ratio)
    '''
    
    return K.mean(-(1-y_true) * K.log(y_pred+1e-20) - global_vars.neg_loss_wt * (y_true) * K.log(1-y_pred+1e-20))


def cosine_distance(vects):
    ''' computes dot product between two inputs. "vects" is a list with two matrices'''
    x, y = vects
    print(x,y)
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    dot_prod = K.sum((x * y), axis=-1, keepdims=True)
    return K.sigmoid(dot_prod)


def cos_dist_output_shape(shapes):
    ''' needed for Lambda functions. Returns shape of output of Lambda function. Used only in Lambda functions '''
    shape1, shape2 = shapes
    return (shape1[0], 1)


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
    input_rep = pad_sequences(sequences, maxlen=global_vars.MaxLen)
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


def user_load_model(arch_file):
    ''' returns model architecture 
        arch_file -- .yaml file created in training stage
    '''
    yaml_file = open(arch_file, 'r')
    yaml_string = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(yaml_string)
    return model



