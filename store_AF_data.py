import pandas       
from hdf_utils import *           

''' This script is data specific. This works on /var/users/purvak/csat/walgreen/features_target_balanced_v2.csv
    If you have new data set different to this format, please write your own script. 
    Input -- data in the /var/users/purvak/csat/walgreen/features_target_balanced_v2.csv  format
    output -- stores the required data in h5 format in AF_data_path
'''

task_comp_flag = int(sys.argv[1]) # set it to 0 if you do not want to consider task completion info in your feature set 
AF_data_path = sys.argv[2] # stores the data in this file

# Run like this: python store_AF_data.py 1 AF_data.h5

# picks the columns with following labels
if task_comp_flag == 0:
    features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate']
elif task_comp_flag == 1:
    features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate', 'q3']
elif task_comp_flag == 2:  
    features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate','q3_1','q3_2','q3_3']

cuid =['cuid'] # utterance id column
target = ['binaryNumber'] # labels column
AF_data_csv = "/var/users/purvak/csat/walgreen/features_target_balanced_v2.csv"
dataframe_features = pandas.read_csv(AF_data_csv, usecols=features)
dataframe_cuid = pandas.read_csv(AF_data_csv, usecols=cuid)
dataframe_target = pandas.read_csv(AF_data_csv, usecols=target)
# converting to dict
dataframe_cuid = dataframe_cuid.to_dict()['cuid']
audio_feat = dataframe_features.values
audio_feat = audio_feat.reshape(audio_feat.shape + (1,))    

# writing the data to h5 file
for utt in utt2ind.keys():
    utt_data = audio_feat[utt2ind[utt]]
    # utt_data is data to be stored
    # utt is uttearance id
    hdf5_write_compression(utt_data, AF_data_path, utt)

