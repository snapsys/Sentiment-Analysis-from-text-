# Sentiment-Analysis-from-text-

Siamese_FUSE_text_temporal.py  is training script
evaluate_CV_v2_FUSE_CNNwithAF.py is script for evalution

Run from /mnt/NFS.cltlabnas1vg0/users/raghavendra/Github_version  
example usage: CUDA_VISIBLE_DEVICES=" " python evaluate_CV_v2_FUSE_CNNwithAF.py best_result/run_1model_SPOKEN_Siamese.yaml best_result/run_1SPOKEN_Siamese_3.0_foldno_1_48-0.72.hdf5 ../CSAT_scripts/data_CV/ results/ 3 fold_1 1,2 AF_data.h5 -p test -f 1 -n 0



# Things to consider before testing
# You need to have utt2label_test.txt in data_dir with filename and label info
# Script expect you to have text data file with .csv extension. You need to have only relevant text in the file. Example file is in 
/var/users/raghavendra/CSAT_scripts/transcripts_mod//CSAT_scripts/transcripts_mod/AOC35MRJJ4QM6K56D3.csv

# Need to have temporal features in a .h5 file. Example script to create .h5 file is /mnt/NFS.cltlabnas1vg0/users/raghavendra/Github_version/store_AF_data.py





