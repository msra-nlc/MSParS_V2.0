# MSParS-V2.0

Code and dataset for paper "Asking Clarification Questions in Knowledge-Based Question Answering". We show the running commands by taking multi-turn case as an example. 

## Folder Description
1. data. It contains the constructed dataset, which is split into two parts: single-turn data and multi-turn data. Each part contains the data of three tasks: task1--clarification identification, task2--clarification generation, task3--question answering.  
2. code. The code folder includes two kinds of models: classification models and genertive models. Task1 and task3 share the same classification models while task2 uses generative models.

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
python3 pre_process.py --data_path ../../../data/multi-turn/task1/ --task task1 #Generate data for baseline CNN
```

#### Train & Test
```
cd ../a02_TextCNN/
python3 p7_TextCNN_train.py --cache_file_h5py ../../../data/multi-turn/task1/data.h51 --cache_file_pickle ../../../data/multi-turn/task1/vocab_label.pik #CNN
```

### Baseline 2: HAN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_hir.py --data_pat ../../../data/multi-turn/task1/ --task task1  #Generate data for HAN and DMN
```

#### Train & Test
```
cd ../a05_HierarchicalAttentionNetwork/
python3 p1_HierarchicalAttention_train.py --cache_file_h5py ../../../data/multi-turn/task1/hir_data.h5 --cache_file_pickle ../../../data/multi-turn/task1/hir_vocab_label.pik #HAN
```

### Baseline 3: DMN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py "../../../data/multi-turn/task1/" --task "task1"  #Generate data for HAN and DMN
```

#### Train & Test
```
cd ../a09_DynamicMemoryNet/
python3 a8_train.py --cache_file_h5py ../../../data/multi-turn/task1/hir_data.h5 --cache_file_pickle ../../../data/multi-turn/task1/hir_vocab_label.pik #HAN
```


### Baseline 4: Transformer
#### Preprocessing
```
cd code/classification/preprocess_file
python generate_input_file.py --data_path "../../../data/multi-turn/task1/" --task "task1"
cd ..
python3 preprocess.py -train_src ../../../data/multi-turn/task1/src-train.txt -train_tgt ../../../data/multi-turn/task1/tgt-train.txt -valid_src ../../../data/multi-turn/task1/src-test.txt -valid_tgt ../../../data/multi-turn/task1/tgt-test.txt -save_data ../../../data/multi-turn/task1/demo -dynamic_dict

```

#### Train & Test
```
python3 train.py -data ../../../data/multi-turn/task1/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 8 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 200
```

## Task 2: Clarification Generation



### Requirements
* python 3.5.6
* torchtext 0.4.0
* torch 1.2.0


### Preprocessing
```
python3 preprocess.py -train_src ../../../data/multi-turn/task2/src-train.txt -train_tgt ../../../data/multi-turn/task2/tgt-train.txt -valid_src ../../../data/multi-turn/task2/src-test.txt -valid_tgt ../../../data/multi-turn/task2/tgt-test.txt -save_data ../../../data/multi-turn/task2/demo -dynamic_dict -share_vocab
```



### Train
```
python3 train.py -data ../../../data/multi-turn/task2/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 1000 -copy_attn

```

### Test
```
python translate.py -model available_models/demo-single-model-transformer-without-copy_step_10000.pt -src ../../../data/multi-turn/task2/src-test.txt -output multi-copy.txt -replace_unk -verbose -gpu 0 -beam_size 1
python merge.py
perl tools/multi-bleu.perl ../../../data/multi-turn/task2/tgt-test1.txt < final_output
```



## Task 3.1: Clarification-based Question Answering -- Entity Prediction



### Requirements
* python 3.5.6
* torchtext 0.4.0
* torch 1.2.0
* h5py 2.8.0
* tflearn 0.3.2

### Baseline 1: CNN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_classifier.py --data_path "../../../data/multi-turn/task3/" --task "task3.1" #Generate data for baseline CNN
```

#### Train & Test
```
cd ../a02_TextCNN/
python3 p7_TextCNN_train.py --cache_file_h5py ../../../data/multi-turn/task3/data.h51 --cache_file_pickle ../../../data/multi-turn/task3/vocab_label.pik #CNN
```

### Baseline 2: HAN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py --data_pat "../../../data/multi-turn/task3/" --task "task3.1"  #Generate data for HAN and DMN
```

#### Train & Test
```
cd ../a05_HierarchicalAttentionNetwork/
python3 p1_HierarchicalAttention_train.py --cache_file_h5py ../../../data/multi-turn/task3/hir_data.h5 --cache_file_pickle ../../../data/multi-turn/task3/hir_vocab_label.pik #HAN
```

### Baseline 3: DMN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py "../../../data/multi-turn/task3/" --task "task3.1"  #Generate data for HAN and DMN
```

#### Train & Test
```
cd ../a09_DynamicMemoryNet/
python3 a8_train.py --cache_file_h5py ../../../data/multi-turn/task3/hir_data.h5 --cache_file_pickle ../../../data/multi-turn/task3/hir_vocab_label.pik #HAN
```


### Baseline 4: Transformer
#### Preprocessing
```
cd code/classification/preprocess_file
python generate_input_file.py --data_path "../../../data/multi-turn/task3/" --task "task3.1"
cd ..
python3 preprocess.py -train_src ../../../data/multi-turn/task3/src-train.txt -train_tgt ../../../data/multi-turn/task3/tgt-train.txt -valid_src ../../../data/multi-turn/task3/src-test.txt -valid_tgt ../../../data/multi-turn/task3/tgt-test.txt -save_data ../../../data/multi-turn/task3/demo -dynamic_dict

```

#### Train & Test
```
python3 train.py -data ../../../data/multi-turn/task3/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 8 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 200
```


## Task 3.2: Clarification-based Question Answering -- Predicate Prediction



### Requirements
* python 3.5.6
* torchtext 0.4.0
* torch 1.2.0
* h5py 2.8.0

### Baseline 1: CNN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_classifier.py --data_path "../../../data/multi-turn/task3/" --task "task3.2" #Generate data for baseline CNN
```

#### Train & Test
```
cd ../a02_TextCNN/
python3 p7_TextCNN_train.py --cache_file_h5py ../../../data/multi-turn/task3/data.h51 --cache_file_pickle ../../../data/multi-turn/task3/vocab_label.pik #CNN
```

### Baseline 2: HAN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py --data_pat "../../../data/multi-turn/task3/" --task "task3.2"  #Generate data for HAN and DMN
```

#### Train & Test
```
cd ../a05_HierarchicalAttentionNetwork/
python3 p1_HierarchicalAttention_train.py --cache_file_h5py ../../../data/multi-turn/task3/hir_data.h5 --cache_file_pickle ../../../data/multi-turn/task3/hir_vocab_label.pik #HAN
```

### Baseline 3: DMN
#### Preprocessing
```
cd code/classification/preprocess_file
python3 pre_process_predicate_hir.py "../../../data/multi-turn/task3/" --task "task3.2"  #Generate data for HAN and DMN
```

#### Train & Test
```
cd ../a09_DynamicMemoryNet/
python3 a8_train.py --cache_file_h5py ../../../data/multi-turn/task3/hir_data.h5 --cache_file_pickle ../../../data/multi-turn/task3/hir_vocab_label.pik #HAN
```


### Baseline 4: Transformer
#### Preprocessing
```
cd code/classification/preprocess_file
python generate_input_file.py --data_path "../../../data/multi-turn/task3/" --task "task3.2"
cd ..
python3 preprocess.py -train_src ../../../data/multi-turn/task3/src-train.txt -train_tgt ../../../data/multi-turn/task3/tgt-train.txt -valid_src ../../../data/multi-turn/task3/src-test.txt -valid_tgt ../../../data/multi-turn/task3/tgt-test.txt -save_data ../../../data/multi-turn/task3/demo -dynamic_dict

```

#### Train & Test
```
python3 train.py -data ../../../data/multi-turn/task3/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 8 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 200
```


## News

* Update and review single-turn data

