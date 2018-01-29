# Sentiment-Analysis-from-text-

### **** TODO
1.  Document main() in evaluate_CV_v2_FUSE_CNNwithAF.py.  When
    documenting a function please include the three essentials: 1.
    overall description (including "side
    effects"), 2. inputs, 3. outputs

2.  We need detailed descriptions of all data formats, not just
pointers to example files.

3.  We need all programs for preprocessing features.  How do I got
    from a raw transcript to final output?


## Tested with versions:

Python 3.6.3

Tensorflow 1.4.1

Keras 2.0.8


### 1. Reproduce results (complete):

evaluate_CV_v2_FUSE_CNNwithAF.py is script for evalution

example usage:

src_dir=/mnt/NFS.cltlabnas1vg0/users/raghavendra/Github_version

out_dir=/your/path/here

CUDA_VISIBLE_DEVICES=" " python ${src_dir}/evaluate_CV_v2_FUSE_CNNwithAF.py ${src_dir}/best_result/run_1model_SPOKEN_Siamese.yaml ${src_dir}/best_result/run_1SPOKEN_Siamese_3.0_foldno_1_48-0.72.hdf5 ${src_dir}/data_CV/ ${out_dir}/results/ 3 fold_1 1,2 ${src_dir}/AF_data.h5 -p test -f 1 -n 0

### 2. Test on new data (incomplete):   
**data directory (data_dir)** should have following files:   
**tokenizer** for each fold -- can be found here: ${src_dir}/../CSAT_scripts/data_CV/tokenizer_None_CV_*fold.pkl            
                             -- used to turn the text into appropriate input format. Usually converting the text into sequence of word indices. For details look at https://keras.io/preprocessing/text/#tokenizer 

**utt2label_test.txt** if you are running on new transcripts. It contains two columns, first column with file name and second column with corresponding label       
Example:     
    21J8CSQUKBBC9U0QQG 1    
    9K1MVL2OBUHSTUK5F6 1    
    2CD35MEMKMD1UGG2AD 0    
    F4G5186A3HDA1AUPKF 1    
    FII20I812P26QFPGLQ 0    
    A0EAE75296JVLECGS2 0    

**Temporal features Info:** 
    Temporal features are expected to be stored in h5 format. It is Hierarchical Data format, used to store and access efficiently. For more details refer https://support.hdfgroup.org/HDF5/   
    The scripts expect you to have seperate data set (data set is h5 terminalogy) for each utterance in h5 file. Details of how to store can be understood from store_AF_data.py.

**Text files Info:**    
    All the scripts expect you to supply the text files with ONLY WITH RELEVANT data. I.e., it should contain only the words which need to be processed.     
    **Example** file is below:  
        hello    
        hello thank you  
        hello thank you for calling xyz dot com my name is debbie im a up   
        hi sandy im i im try im up loaded a photo   
        um i wanna get out of the way before i lose the fifty percent off   
        um and i cant i of loaded the photos but im trying to um    
        put them in my car in order them but i need some help   
        okay    

    One example of **WRONG** format is below: (first few columns contain unwanted information, only the last column is relevant for us) 
        21J8CSQUKBBC9U0QQG,15.81,16.725,2,hello 
        21J8CSQUKBBC9U0QQG,16.23,17.235,1,hello thank you   
        21J8CSQUKBBC9U0QQG,17.64,21.735,1,hello thank you for calling xyz dot com my name is debbie im a up 
        21J8CSQUKBBC9U0QQG,22.68,28.875,2,hi sandy im i im try im up loaded a photo 
        21J8CSQUKBBC9U0QQG,29.13,34.635,2,um i wanna get out of the way before i lose the fifty percent off     
        21J8CSQUKBBC9U0QQG,35.88,41.925,2,um and i cant i of loaded the photos but im trying to um  
        21J8CSQUKBBC9U0QQG,42.09,46.395,2,put them in my car in order them but i need some help 
        21J8CSQUKBBC9U0QQG,48.18,49.635,1,okay  


### **** Everything below here is unclear to me and needs explanation and detail.



Things to consider before testing:

1. You need to have utt2label_test.txt in data_dir with filename and
label info

2. Script expect you to have text data file with .csv extension. You need to have only relevant text in the file. Example file is in 
/var/users/raghavendra/CSAT_scripts/transcripts_mod//CSAT_scripts/transcripts_mod/AOC35MRJJ4QM6K56D3.csv

3. Need to have temporal features in a .h5 file. Example script to create .h5 file is /mnt/NFS.cltlabnas1vg0/users/raghavendra/Github_version/store_AF_data.py




### 3.  Training (incomplete)

Siamese_FUSE_text_temporal.py  is training script


4. System Architecture  


![Alt text](https://github.com/snapsys/Sentiment-Analysis-from-text-/blob/master/base_model_1.jpg)





