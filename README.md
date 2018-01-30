# Sentiment-Analysis-from-text-

### **** TODO
1.  Document main() in evaluate_CV_v2_FUSE_CNNwithAF.py.  When
    documenting a function please include the three essentials: 1.
    overall description (including "side
    effects"), 2. inputs, 3. outputs


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

    Format: 
        CUDA_VISIBLE_DEVICES=" " python ${src_dir}/evaluate_CV_v2_FUSE_CNNwithAF.py arch_file model_path data_dir  out_dir weight suffix layer_indices AF_data_path -n new_transcripts -tr_dir transcripts_dir -p post_string -f fold_no 

        Inputs description: 
            arch_file -- model architecture used for training
            model_path -- model weights stored in training
            data_dir -- Path for data dir which should contain tokenizer and utt2label_test.txt
            out_dir -- dir to store the results file
            weight -- Constant factor of verification loss, not used in any where except to store the result. Technically, can    
                        be given any value at the time of evaluation.
            suffix -- suffix need to be added to result file name. Again, can be given any string.
            layer_indices -- list of layer indices of your model using which you will extract features. In the present case,
                             anything can be given because I hard coded layer indices in the code.
            AF_data_path -- path for h5 file which should contain temporal features
            (-n)--new_transcripts -- boolean, set to 1 if you are using new transcripts, otherwise set to 0 (default)
            (-tr_dir)--transcripts_dir -- path for transcripts
            (-p)--post_string -- data set to be evaluated. Can take "train", "test"(default)
            (-f)--fold_no -- Cross validation fold number. Tokenizer will be selected based on this, model_path and arch_file
                            should also be correspong to this fold number
            
            
        
        
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
    
    Example file is below:  
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



### 3.  Training (incomplete)

Siamese_FUSE_text_temporal.py  is training script


### 4. System Architecture

Top level architecture is:  


![Alt text](https://github.com/snapsys/Sentiment-Analysis-from-text-/blob/master/sum_base_models_dropout.jpg)

where **BaseModel(BM)** archtecture is shown below  


![Alt text](https://github.com/snapsys/Sentiment-Analysis-from-text-/blob/master/base_model_2.jpg)






