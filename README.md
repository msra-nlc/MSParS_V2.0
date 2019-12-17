# MSParS-V2.0-

Under Constricution



## Task 1: Clarification Identification



### Requirements
* Python 3.5
* nltk


### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_classifier.py --data_path "data/multi-turn/step1/"  #Generate data for baseline CNN
python3 pre_process_predicate_hir.py --data_path "data/multi-turn/step1/"  #Generate data for HAN and DMN
```



### Train & Test
```
cd ../a09_DynamicMemoryNet/
python3 a8_train.py --cache_file_h5py ../preprocess_file/data/new_data/multi_turn/hir_data.h5 --cache_file_pickle ../preprocess_file/data/new_data/multi_turn/hir_vocab_label.pik # DMN

cd ../a05_HierarchicalAttentionNetwork/
python3 p1_HierarchicalAttention_train.py --cache_file_h5py ../preprocess_file/data/new_data/multi_turn/hir_data.h5 --cache_file_pickle ../preprocess_file/data/new_data/multi_turn/hir_vocab_label.pik #HAN

cd ../a02_TextCNN/
python3 p7_TextCNN_train.py --cache_file_h5py ../preprocess_file/data/new_data/multi_turn/data.h51 --cache_file_pickle #CNN



```



## Task 2: Clarification Generation



### Requirements
* Python 3.5
* nltk


### Preprocessing
```
python3 preprocess.py -train_src multi-turn-data/src-train.txt -train_tgt multi-turn-data/tgt-train.txt -valid_src multi-turn-data/src-test.txt -valid_tgt multi-turn-data/tgt-test.txt -save_data multi-turn-data/demo -dynamic_dict -share_vocab
```



### Train
```
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py -data multi-turn-data/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 1000 -copy_attn > log_multi_transformer_copy.txt &

```

### Test
```
python translate.py -model available_models/demo-single-model-transformer-without-copy_step_10000.pt -src single-turn-data/src-test.txt -output multi-without-copy.txt -replace_unk -verbose -gpu 0 -beam_size 1
python merge.py
perl tools/multi-bleu.perl /data/v-jinxu/code/OpenNMT-py-master_ord/OpenNMT-py-master/single-turn-data/tgt-test1.txt < final_output


```