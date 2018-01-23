# Sentiment-Analysis-from-text-

## Training script:
Siamese_FUSE_text_temporal.py  is training script

## Evaluation script:
evaluate_CV_v2_FUSE_CNNwithAF.py is script for evalution



example usage:

src_dir=/mnt/NFS.cltlabnas1vg0/users/raghavendra/Github_version

out_dir=/your/path/here

CUDA_VISIBLE_DEVICES=" " python ${src_dir}/evaluate_CV_v2_FUSE_CNNwithAF.py ${src_dir}/best_result/run_1model_SPOKEN_Siamese.yaml ${src_dir}/best_result/run_1SPOKEN_Siamese_3.0_foldno_1_48-0.72.hdf5 ${src_dir}/data_CV/ ${out_dir}/results/ 3 fold_1 1,2 ${src_dir}/AF_data.h5 -p test -f 1 -n 0

data_dir contents required:
tokenizer for each fold -- can be found here: ${src_dir}/../CSAT_scripts/data_CV/tokenizer_None_CV_*fold.pkl 
utt2label_test.txt if you are running on new transcripts.



Things to consider before testing:

1. You need to have utt2label_test.txt in data_dir with filename and
label info

2. Script expect you to have text data file with .csv extension. You need to have only relevant text in the file. Example file is in 
/var/users/raghavendra/CSAT_scripts/transcripts_mod//CSAT_scripts/transcripts_mod/AOC35MRJJ4QM6K56D3.csv

3. Need to have temporal features in a .h5 file. Example script to create .h5 file is /mnt/NFS.cltlabnas1vg0/users/raghavendra/Github_version/store_AF_data.py





