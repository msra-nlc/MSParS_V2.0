# MSParS-V2.0

Code for paper "Asking Clarification Questions in Knowledge-Based Question Answering"

Under Construction



## Task 1: Clarification Identification



### Requirements
* python 3.5.6
* torchtext 0.4.0
* torch 1.2.0
* h5py 2.8.0

### Baseline 1: CNN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_classifier.py --data_path "data/multi-turn/step1/"  #Generate data for baseline CNN
```

#### Train & Test
```
cd ../a02_TextCNN/
python3 p7_TextCNN_train.py --cache_file_h5py ../preprocess_file/data/new_data/multi_turn/data.h51 --cache_file_pickle #CNN
```

### Baseline 2: HAN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py --data_path "data/multi-turn/step1/"  #Generate data for HAN and DMN
```

#### Train & Test
```
cd ../a05_HierarchicalAttentionNetwork/
python3 p1_HierarchicalAttention_train.py --cache_file_h5py ../preprocess_file/data/new_data/multi_turn/hir_data.h5 --cache_file_pickle ../preprocess_file/data/new_data/multi_turn/hir_vocab_label.pik #HAN
```

### Baseline 3: DMN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py --data_path "data/multi-turn/step1/"  #Generate data for HAN and DMN
```

### Baseline 4: Transformer
#### Preprocessing
```
cd multi-turn-data
generate_input_file.py
cd ..
python3 preprocess.py -train_src multi-turn-data/src-train.txt -train_tgt multi-turn-data/tgt-train.txt -valid_src multi-turn-data/src-test.txt -valid_tgt multi-turn-data/tgt-test.txt -save_data multi-turn-data/demo -dynamic_dict

```

#### Train & Test
```
python3 train.py -data multi-turn-data/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 8 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 200
```

## Task 2: Clarification Generation



### Requirements
* python 3.5.6
* torchtext 0.4.0
* torch 1.2.0


### Preprocessing
```
python3 preprocess.py -train_src multi-turn-data/src-train.txt -train_tgt multi-turn-data/tgt-train.txt -valid_src multi-turn-data/src-test.txt -valid_tgt multi-turn-data/tgt-test.txt -save_data multi-turn-data/demo -dynamic_dict -share_vocab
```



### Train
```
python3 train.py -data multi-turn-data/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 1000 -copy_attn

```

### Test
```
python translate.py -model available_models/demo-single-model-transformer-without-copy_step_10000.pt -src single-turn-data/src-test.txt -output multi-copy.txt -replace_unk -verbose -gpu 0 -beam_size 1
python merge.py
perl tools/multi-bleu.perl multi-turn-data/tgt-test1.txt < final_output
```



## Task 3: Clarification-based Question Answering



### Requirements
* python 3.5.6
* torchtext 0.4.0
* torch 1.2.0
* h5py 2.8.0

### Baseline 1: CNN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_classifier.py --data_path "data/multi-turn/step3_entity/"  #Generate data for baseline CNN
```

#### Train & Test
```
cd ../a02_TextCNN/
python3 p7_TextCNN_train.py --cache_file_h5py ../preprocess_file/data/multi-turn/step3_entity/data.h51 --cache_file_pickle #CNN
```

### Baseline 2: HAN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py --data_path "data/multi-turn/step3_entity/"  #Generate data for HAN and DMN
```

#### Train & Test
```
cd ../a05_HierarchicalAttentionNetwork/
python3 p1_HierarchicalAttention_train.py --cache_file_h5py ../preprocess_file/data/multi-turn/step3_entity/hir_data.h5 --cache_file_pickle ../preprocess_file/data/multi-turn/step3_entity/hir_vocab_label.pik #HAN
```

### Baseline 3: DMN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py --data_path "data/multi-turn/step3_entity/"  #Generate data for HAN and DMN
```

### Baseline 4: Transformer
#### Preprocessing
```
cd multi-turn-data
generate_input_file.py
cd ..
python3 preprocess.py -train_src multi-turn-data/src-train.txt -train_tgt multi-turn-data/tgt-train.txt -valid_src multi-turn-data/src-test.txt -valid_tgt multi-turn-data/tgt-test.txt -save_data multi-turn-data/demo -dynamic_dict

```

#### Train & Test
```
python3 train.py -data multi-turn-data/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 8 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 200
```
