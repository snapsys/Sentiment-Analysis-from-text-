import keras
from keras import backend as K
from keras.models import model_from_yaml
import numpy as np
from keras.preprocessing.sequence import pad_sequences 
from masking_layers import *


def Triplet_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06 http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        Used to get Triplet loss
        y_pred -- matrix of distances
        y_true -- vector of labels (generally obtained from mini batch generator)
    '''

    cond1 = [tf.cast(tf.equal(y_true[i], y_true[j]), tf.float32) for i in range(total_no_samples) for j in range(total_no_samples)]
    cond1 = tf.reshape(cond1, [total_no_samples, -1])

    loss = [ cond1[i,j]*(1-cond1[i,k])*(-y_pred[i, j] + y_pred[i, k] + alpha) for i in range(total_no_samples) for j in range(i+1, total_no_samples) for k in range(total_no_samples)]

    sorted_loss,indices = tf.nn.top_k(loss, k=total_no_samples, sorted=True)
    return tf.reduce_sum(sorted_loss)/total_no_samples


def Siamese_lik_ratio(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06 http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        Used to get Siamese loss
        y_pred -- matrix of similarities (not distances) -- This is inconsistent with convention
        y_true -- vector of labels (generally obtained from mini batch generator)
    '''
    y_pred = [y_pred[i,j] for i in range(total_no_samples) for j in range(i+1, total_no_samples)]
    y_pred = tf.stack(y_pred, axis=0)
    y_true_pair = [tf.cast(tf.not_equal(y_true[i], y_true[j]), tf.float32) for i in range(total_no_samples) for j in range(i+1, total_no_samples)]
    y_true_pair = tf.stack(y_true_pair, axis=0)
    no_neg_pairs = tf.reduce_sum(y_true_pair)
    no_pos_pairs = tf.reduce_sum(1-y_true_pair)
    neg_loss_wt = tf.cond(no_neg_pairs > 0, lambda: tf.divide(no_pos_pairs, no_neg_pairs), lambda: tf.Variable(0.0))
    return K.mean(-(1-y_true_pair) * K.log(y_pred+1e-20) - neg_loss_wt * (y_true_pair) * K.log(1-y_pred+1e-20))


def lik_ratio(y_true, y_pred):
    ''' computes likelihood ratio between predictions and true labels. Did not put in utils.py because I am using neg_loss_wt as global variable   
        --> Different from Siamese_lik_ratio. It just takes two predictions and computes a metric similar to binary cross entropy
    Inputs: Ground truth label and predicted label
    Output: returns the verification loss (liklihood ratio)
    '''
    return K.mean(-(1-y_true) * K.log(y_pred+1e-20) - global_vars.neg_loss_wt * (y_true) * K.log(1-y_pred+1e-20))


def cosine_distance_twoIPs(vects):
    ''' computes dot product between two inputs. "vects" is a list with two matrices'''
    x, y = vects
    print(x,y)
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    dot_prod = K.sum((x * y), axis=-1, keepdims=True)
    return K.sigmoid(dot_prod)


def cosine_distance(vects): 
    ''' used to calculate cosine similarities (not distances) within a minibatch. 
        vects -- hidden layer activations on which you want to apply verification loss
        Returns sigmoid of cosine similarity matrix
    '''
    x = vects[0]   
    x = K.l2_normalize(x, axis=-1)  
    dot_prod = K.dot(x, K.transpose(x))  
    return K.sigmoid(dot_prod)  



def euclidean_distance(vects):
    ''' Used to get euclidean distance matrix with in minibatch
        vects -- hidden layer activations on which you want to apply verification loss 
        Returns euclidean distance matrix

    '''
    x = vects[0]
    #x = K.l2_normalize(x, axis=-1)
    magnitude = tf.reduce_sum(x*x, 1)
    magnitude = tf.reshape(magnitude, [-1, 1])
    dist = magnitude - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(magnitude)
    return dist


def cos_dist_output_shape(shapes):
    ''' needed for Lambda functions. Returns shape of output of Lambda function. Used only in Lambda functions '''
    shape1, shape2 = shapes
    return (shape1[0], 1)


def euclidean_Siamese(y_true, y_pred):
    ''' used to get Siamese loss
        y_pred -- matrix of distances (not similarities)
        y_true -- vector of labels (generally obtained from mini batch generator)
    '''
    y_pred = [y_pred[i,j] for i in range(total_no_samples) for j in range(i+1, total_no_samples)]
    y_pred = tf.stack(y_pred, axis=0)
    y_true_pair = [tf.cast(tf.not_equal(y_true[i], y_true[j]), tf.float32) for i in range(total_no_samples) for j in range(i+1, total_no_samples)]
    y_true_pair = tf.stack(y_true_pair, axis=0)
    no_neg_pairs = tf.reduce_sum(y_true_pair)
    no_pos_pairs = tf.reduce_sum(1-y_true_pair)
    neg_loss_wt = tf.cond(no_neg_pairs > 0, lambda: tf.divide(no_pos_pairs, no_neg_pairs), lambda: tf.Variable(0.0))
    margin = 0.3
    #return K.mean((1-y_true_pair) * K.square(y_pred) * 0 + (y_true_pair) * K.square(K.maximum(margin - y_pred, 0)))

    return K.mean((1-y_true_pair) * K.square(y_pred) + neg_loss_wt * (y_true_pair) * K.square(K.maximum(margin - y_pred, 0)))


def Triplet_acc(y_true, y_pred):
    '''Used to calculate Triplet Accuracy. 
        y_pred -- matrix of distances (not similarities)
        y_true -- vector of labels (generally obtained from mini batch generator)
    '''
    cond1 = [tf.cast(tf.equal(y_true[i], y_true[j]), tf.float32) for i in range(total_no_samples) for j in range(total_no_samples)]
    cond1 = tf.reshape(cond1, [total_no_samples, -1])

    no_triplets = tf.reduce_sum([cond1[i,j]*(1-cond1[i,k]) for i in range(total_no_samples) for j in range(i+1, total_no_samples) for k in range(total_no_samples)])

    count_correct = tf.reduce_sum([cond1[i,j]*(1-cond1[i,k])*tf.cast(tf.less(-y_pred[i, j] + y_pred[i, k], 0), tf.float32) for i in range(total_no_samples) for j in range(i+1, total_no_samples) for k in range(total_no_samples)])

    acc = tf.cond(no_triplets > 0, lambda: tf.divide(count_correct, no_triplets), lambda: tf.Variable(0.0))
    return acc


def doc_represent_embedding(tokenizer, data, MaxLen):
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


def text_file_load(file_id, transcripts_dir):
    ''' returns sequence of words of a document removing new line character. Expects input file to have only relevant text
        file_id -- name of file to be used
    '''
    with open(transcripts_dir + file_id + '.csv', 'r') as f:
        conversation = f.readlines()
        conversation = [sentence.strip() for sentence in conversation]
        conversation = ' '.join(conversation)
    return conversation


def text_file_load_Reverse(file_id, transcripts_dir):
    ''' returns sequence of words of a document removing new line character. Expects input file to have only relevant text
        file_id -- name of file to be used
    '''
    with open(transcripts_dir + file_id + '.csv', 'r') as f:
        conversation = f.readlines()
        conversation.reverse()
        conversation = [sentence.strip() for sentence in conversation]
        conversation = ' '.join(conversation)
    return conversation


def text_file_load_CustOnly(file_id, transcripts_dir):
    ''' returns sequence of words of a document removing new line character. Expects input file to have only relevant text
        file_id -- name of file to be used
    '''
    with open(transcripts_dir + file_id + '.csv', 'r') as f:
        conversation = f.readlines()
        conversation = [sentence.strip() for sentence in conversation]
        conversation = ' '.join(conversation)
    return conversation



def text_load_SpeechTranscripts(file_id, transcripts_dir):
    ''' returns sequence of words of a document removing new line character. Expects input file to have only relevant text
        file_id -- name of file to be used
    '''
    with open(transcripts_dir + file_id + '.csv', 'r') as f:
        conversation = f.readlines()
        conversation = [sentence.strip().split(',')[-1] for sentence in conversation]
        conversation = ' '.join(conversation)
    return conversation


def text_load_SpeechTranscripts_SpkrID(file_id, MaxLen, transcripts_dir):
    ''' returns sequence of words of a document removing new line character. Expects input file to have only relevant text
        file_id -- name of file to be used
        MaxLen -- maximum number of words considered for each document
    '''
    with open(transcripts_dir + file_id + '.csv', 'r') as f:
        conversation = f.readlines()
        spkrID = np.array([int(sentence.strip().split(',')[-2]) for sentence in conversation])
    if len(spkrID) >= MaxLen:
        spkrID = spkrID[:MaxLen]
    else:
        temp = np.zeros(MaxLen - len(spkrID))
        spkrID = np.hstack([temp, spkrID])

    return spkrID


def user_load_model(arch_file):
    ''' returns model architecture 
        arch_file -- .yaml file created in training stage
    '''
    yaml_file = open(arch_file, 'r')
    yaml_string = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(yaml_string)
    return model

def user_load_model_MaskedPooling_SpkrID(arch_file):
    ''' returns model architecture. It is different from user_load_model(), it is used to load arch file with custom objects 
        arch_file -- .yaml file created in training stage
    '''
    yaml_file = open(arch_file, 'r')
    yaml_string = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(yaml_string, custom_objects={'GlobalMaskedAveragePooling1D_Spkr1':GlobalMaskedAveragePooling1D_Spkr1,'GlobalMaskedAveragePooling1D_Spkr2':GlobalMaskedAveragePooling1D_Spkr2})
    return model



