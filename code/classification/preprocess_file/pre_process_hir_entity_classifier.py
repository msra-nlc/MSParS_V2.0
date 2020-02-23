# import some packages
import pandas as pd
from collections import Counter
from tflearn.data_utils import pad_sequences
import random
import numpy as np
import h5py
import pickle
import json
import codecs
print("import package successful...")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path")
parser.add_argument("--task", choices=["task1","task3.1","task3.2"]) # task1, task3
args = parser.parse_args()
if args.task == "task3.1":
    train_file = "train_sml.txt"
    dev_file = "dev_sml.txt"
    test_file = "test_sml.txt"
elif args.task == "task3.2":
    train_file = "train_sml_classifier.txt"
    dev_file = "dev_sml_classifier.txt"
    test_file = "test_sml_classifier.txt"
else:
    train_file = "train_classifier.txt"
    dev_file = "dev_classifier.txt"
    test_file = "test_classifier.txt"

def read_word(path):
    sr = codecs.open(path, "r", "utf-8")
    lines = sr.readlines()
    word_fre = {}
    for line in lines:
        line = json.loads(line)
        
        items = " <SP> ".join([line["context"],line["entity1"],line["entity2"]]) 
        
        #label = items[1].strip()
        #input_ = items[0].strip().replace("<S>", "SEP")
        #input_sentence = input_.split(" <SP> ")
        #label = int(label)
        words = items.split(" ")
        for word in words:
            if word in word_fre:
                word_fre[word] += 1
            else:
                word_fre[word] = 1
    print(len(word_fre))
    return [x[0] for x in sorted(word_fre.items(), key = lambda x: x[1])[::-1][:50000]]

       
def read_type(path):
    types = {}

    sr = codecs.open(path, "r", "utf-8")
    lines = sr.readlines()
    print(len(lines))
    for line in lines:
        line = json.loads(line)
        items = line["entity1"]
        entity1 = items.split(" <S> ")
        label = entity1[1].strip().split(" ")
        if label in types:
            types[label] += 1
        else:
            types[label] = 1

        entity2 = line["entity2"].split(" <S> ")
        label = entity2[1].strip().split(" ")
        if label in types:
            types[label] += 1
        else:
            types[label] = 1
        
        
    return types

def read_file(path):
    inputs = []
    targets = []
    sr = codecs.open(path, "r", "utf-8")
    lines = sr.readlines()
    for line in lines:
        line = json.loads(line)
        
        items = " <SP> ".join([line["context"],line["entity1"],line["entity2"]]) 
        label = line["user_response_answer"].strip()
        input_ = items.strip().replace("<S>", "SEP")
        input_sentence = input_.split(" <SP> ")
        label = int(label)
        inputs.append(input_sentence)
        targets.append(label)
    return inputs, targets


base_path=args.data_path + "/"


trainx, trainy=read_file(base_path + train_file)
validx, validy=read_file(base_path + dev_file)
testx, testy=read_file(base_path + test_file)


print(len(trainx))
print(len(validx))
print(len(testx))


# read source file as csv
#word_embedding_object = open('data/glove.6B.50d.txt')
#lines_wv = word_embedding_object.readlines()
#word_embedding_object.close()
char_list = []
words = read_word(base_path + train_file)
char_list.extend(['PAD', 'UNK', 'CLS', 'SEP', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5'])
print(len(words))
print(words[0])
char_list.extend(words)
PAD_ID = 0
UNK_ID = 1

# write to vocab.txt under data/ieee_zhihu_cup
vocab_path = base_path + 'hir_vocab.txt'
vocab_char_object = codecs.open(vocab_path, 'w', 'utf-8')

word2index = {}
for i, char in enumerate(char_list):
    if i < 10: print(i, char)
    word2index[char] = i
    vocab_char_object.write(char + "\n")
vocab_char_object.close()
print("vocabulary of word generated....")



label_list=["0", "1"]
label2index={}
label_target_object=open(base_path+'label_set.txt','w')
for i, label_freq in enumerate(label_list):
    label=label_freq
    label2index[label]=i
    label_target_object.write(label+"\n")
label_target_object.close()
print("generate label dict successful...")


def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result




def get(str_, dic):
    if str_ in dic:
        return dic[str_]
    else:
        return 1

def get_X_Y(train_data_x,train_data_y,label_size):
    """
    get X and Y given input and labels
    input:
    train_data_x:
    train_data_y:
    label_size: number of total unique labels(e.g. 1999 in this task)
    output:
    X,Y
    """
    X=[]
    Y=[]

    train_data_x_tiny_test=train_data_x
    train_data_y_tiny_test=train_data_y

    for index, title_char in enumerate(train_data_x_tiny_test):

        #title_char=row.split(" ")
        title_char_id_list=[[get(x,word2index) for x in row.split(" ") if x.strip()] for row in title_char]
        X.append(title_char_id_list)


    for index, row in enumerate(train_data_y_tiny_test):

        topic_id_list=str(row)
        label_list_dense=[label2index[l] for l in topic_id_list if l.strip()]
        label_list_sparse=transform_multilabel_as_multihot(label_list_dense,label_size)
        Y.append(label_list_sparse)
    return X,Y


def save_data(cache_file_h5py,cache_file_pickle,word2index,label2index,train_X,train_X_len,train_Y,vaild_X,valid_X_len,valid_Y,test_X,test_X_len,test_Y):
    # train/valid/test data using h5py
    f = h5py.File(cache_file_h5py, 'w')
    f['train_X'] = train_X
    f['train_X_len'] = train_X_len
    f['train_Y'] = train_Y
    f['vaild_X'] = vaild_X
    f['vaild_X_len'] = valid_X_len
    f['valid_Y'] = valid_Y
    f['test_X'] = test_X
    f['test_X_len'] = test_X_len
    f['test_Y'] = test_Y
    f.close()
    # save word2index, label2index
    with open(cache_file_pickle, 'ab') as target_file:
        pickle.dump((word2index,label2index), target_file)



label_size=len(label2index)
cache_path_h5py=base_path+'hir_data.h5'
cache_path_pickle=base_path+'hir_vocab_label.pik'
max_sentence_length=100

# step 1: get (X,y)
X,train_Y=get_X_Y(trainx,trainy,label_size)
train_X_len = [[len(x) for x in item] for item in X]
train_X = [pad_sequences(item, maxlen=max_sentence_length, value=0.) for item in X]
#train_X = [item + [for i in range(3-)] for item in X]pad_sequences(X, maxlen=3, value=[0. for i in range(100)])  # padding to max length
#X=np.array(X)
train_Y=np.array(train_Y)

# pad and truncate to a max_sequence_length
#train_X = pad_sequences(X, maxlen=max_sentence_length, value=0.)  # padding to max length

X,valid_Y=get_X_Y(validx,validy,label_size)
valid_X_len = [[len(x) for x in item] for item in X]
valid_X = [pad_sequences(item, maxlen=max_sentence_length, value=0.) for item in X]
#valid_X = pad_sequences(X, maxlen=3, value=[0. for i in range(max_sentence_length)])  # padding to max length
valid_Y=np.array(valid_Y)

# pad and truncate to a max_sequence_length
#valid_X = pad_sequences(X, maxlen=max_sentence_length, value=0.)  # padding to max length

X,test_Y=get_X_Y(testx,testy,label_size)
test_X_len = [[len(x) for x in item] for item in X]
test_X = [pad_sequences(item, maxlen=max_sentence_length, value=0.) for item in X]
#test_X = pad_sequences(X, maxlen=3, value=[0. for i in range(max_sentence_length)])  # padding to max length
test_Y=np.array(test_Y)
# pad and truncate to a max_sequence_length
#test_X = pad_sequences(X, maxlen=max_sentence_length, value=0.)  # padding to max length


print("train_X:",len(train_X),";train_Y:",len(train_Y),";vaild_X.shape:",len(valid_X),";valid_Y:",len(valid_Y),";test_X:",len(test_X),";test_Y:",len(test_Y))
print(train_X[0])
print(train_Y[0])


# step 3: save to file system
save_data(cache_path_h5py,cache_path_pickle,word2index,label2index,train_X,train_X_len,train_Y,valid_X,valid_X_len,valid_Y,test_X,test_X_len,test_Y)
print("save cache files to file system successfully!")
