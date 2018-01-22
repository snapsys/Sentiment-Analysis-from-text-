
import pandas       
from hdf_utils import *           

task_comp_flag = 1

if task_comp_flag == 0:
    features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate']
elif task_comp_flag == 1:
    features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate', 'q3']
elif task_comp_flag == 2:  
    features = ['TimeOnTask','MeanWrdsPerSysTurn','MeanWrdsPerUsrTurn','MeanSysTurnDur','NumOverlaps','UsrCallDom','UsrRate','q3_1','q3_2','q3_3']

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

AF_data_path = 'AF_data.h5'
for utt in utt2ind.keys():
    utt_data = audio_feat[utt2ind[utt]]
    hdf5_write_compression(utt_data, AF_data_path, utt)


