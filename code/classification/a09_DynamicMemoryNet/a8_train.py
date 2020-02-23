# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys
import tensorflow as tf
import numpy as np
from a8_dynamic_memory_network import DynamicMemoryNetwork
import h5py
from tflearn.data_utils import to_categorical, pad_sequences
import os,math
import pickle

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cache_file_h5py","","path of training/validation/test data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_string("cache_file_pickle","","path of vocabulary and label files") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_integer("num_classes",2,"number of label") #3 ADDITIONAL TOKEN: _GO,_END,_PAD
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.87一次衰减多�?
tf.app.flags.DEFINE_string("ckpt_dir","../checkpoint_dynamic_memory_network/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",100,"max sentence length") #100
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",20,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #�?0轮做一次验�?
tf.app.flags.DEFINE_integer("validate_step", 2000, "how many step to validate.") #1500做一次检�?
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
#tf.app.flags.DEFINE_string("traning_data_path","../train-zhihu4-only-title-all.txt","path of traning data.") #O.K.train-zhihu4-only-title-all.txt-->training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
#tf.app.flags.DEFINE_string("word2vec_model_path","../zhihu-word2vec-title-desc.bin-100","word2vec's vocabulary and vectors") #zhihu-word2vec.bin-100-->zhihu-word2vec-multilabel-minicount15.bin-100
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.") #set this false. becase we are using it is a sequence of token here.
tf.app.flags.DEFINE_integer("hidden_size",300,"hidden size")
tf.app.flags.DEFINE_integer("story_length",2,"story length")
# you can do experiment by change below two hyperparameter, performance may be changed.
tf.app.flags.DEFINE_boolean("use_gated_gru",False,"whether to use gated gru as  memory update mechanism. if false,use weighted sum of candidate sentences according to gate")
tf.app.flags.DEFINE_integer("num_pass",2,"number of pass to run") #e.g. num_pass=1,2,3,4.
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")

tf.app.flags.DEFINE_boolean("decode_with_sequences",False,"if your task is sequence generating, you need to set this true.default is false, for predict a label")


def write_to_file(path, list_):
  sw = open(path, 'w')
  for word in list_:
    sw.write(str(word)+"\n")
  sw.close()
  
#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #1.load data(X:list of lint,y:int).
    #if os.path.exists(FLAGS.cache_path):  # load training data from cache file.
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    #else:
    print(FLAGS.cache_file_h5py)
    word2index, label2index,trainX,trainXlen,trainY,validX,validXlen,validY,testX,testXlen,testY = load_data(FLAGS.cache_file_h5py,
                                                                                      FLAGS.cache_file_pickle)
    vocab_size = len(word2index)
    print("cnn_model.vocab_size:", vocab_size)
    num_classes = len(label2index)
    print("num_classes:", num_classes)
    num_examples = len(trainX)
    print("num_examples of training:", num_examples, ";sentence_len:", FLAGS.sequence_length)
    # train, test= load_data_multilabel(FLAGS.traning_data_path,vocabulary_word2index, vocabulary_label2index,FLAGS.sentence_len)
    # trainX, trainY = train;testX, testY = test
    # print some message for debug purpose
    print("trainX[0:10]:", trainX[0:10])
    print("trainY[0]:", trainY[0:10])
    #train_y_short = get_target_label_short(trainY[0])
    print("train_y_short:", trainY[0])
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        model = DynamicMemoryNetwork(num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                                     FLAGS.story_length,vocab_size, FLAGS.embed_size, FLAGS.hidden_size, FLAGS.is_training,num_pass=FLAGS.num_pass,
                                     use_gated_gru=FLAGS.use_gated_gru,decode_with_sequences=FLAGS.decode_with_sequences,multi_label_flag=FLAGS.multi_label_flag,l2_lambda=FLAGS.l2_lambda)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            #if FLAGS.use_embedding: #load pre-trained word embedding
            #    assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model,word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch=sess.run(model.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        print("number_of_training_data:",number_of_training_data)
        previous_eval_loss=10000
        best_eval_loss=10000
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])#;print("trainY[start:end]:",trainY[start:end])
                feed_dict = {model.query: trainX[start:end][:,0],model.story: trainX[start:end][:,1:],model.dropout_keep_prob: 1.0}
                #print(len(trainX[start:end]))
                if not FLAGS.multi_label_flag:
                    feed_dict[model.answer_single] = trainY[start:end]
                else:
                    feed_dict[model.answer_multilabel]=trainY[start:end]
                curr_loss,curr_acc,_=sess.run([model.loss_val,model.accuracy,model.train_op],feed_dict) #curr_acc--->TextCNN.accuracy
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                if counter %50==0:
                    print("dynamic_memory_network[use_gated_gru=False,num_pass=2]==>Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f"
                          %(epoch,counter,math.exp(loss/float(counter)) if (loss/float(counter))<20 else 10000.000,acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################
                '''if FLAGS.batch_size!=0 and (start%(FLAGS.validate_step*FLAGS.batch_size)==0): #(epoch % FLAGS.validate_every) or  if epoch % FLAGS.validate_every == 0:
                    eval_loss, eval_acc = do_eval(sess, model, testX, testY, batch_size)
                    print(eval_acc)
                    print("dynamic_memory_network[use_gated_gru=False,num_pass=2].validation.part. previous_eval_loss:", math.exp(previous_eval_loss) if previous_eval_loss<20 else 10000.000,";current_eval_loss:", math.exp(eval_loss) if eval_loss<20 else 10000.000)
                    if eval_loss > previous_eval_loss: #if loss is not decreasing
                        # reduce the learning rate by a factor of 0.5
                        print("dynamic_memory_network[use_gated_gru=False,num_pass=2]==>validation.part.going to reduce the learning rate.")
                        learning_rate1 = sess.run(model.learning_rate)
                        lrr=sess.run([model.learning_rate_decay_half_op])
                        learning_rate2 = sess.run(model.learning_rate)
                        print("dynamic_memory_network[use_gated_gru=False,num_pass=2]==>validation.part.learning_rate1:", learning_rate1, " ;learning_rate2:",learning_rate2)
                    else:# loss is decreasing
                        if eval_loss<best_eval_loss:
                            print("dynamic_memory_network[use_gated_gru=False,num_pass=2]==>going to save the model.eval_loss:",math.exp(eval_loss) if eval_loss<20 else 10000.000,";best_eval_loss:",math.exp(best_eval_loss) if best_eval_loss<20 else 10000.000)
                            # save model to checkpoint
                            #save_path = FLAGS.ckpt_dir + "model.ckpt"
                            #saver.save(sess, save_path, global_step=epoch)
                            best_eval_loss=eval_loss
                    previous_eval_loss = eval_loss'''
                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

            # 5.test on test set
            predicate_y, test_loss, test_acc = do_eval(sess, model, testX, testY, batch_size)
            write_to_file(FLAGS.cache_file_h5py + str(epoch), predicate_y)
            print("Test Loss:%.3f\tTest Accuracy: %.3f" % (test_loss, test_acc))
            _,valid_loss, valid_acc = do_eval(sess, model, validX, validY, batch_size)
            print("valid Loss:%.3f\valid Accuracy: %.3f" % (valid_loss, valid_acc))
    

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# do evalation on validation dataset, report loss and accuracy
def do_eval(sess,model,evalX,evalY,batch_size,eval_decoder_input=None):
    #ii=0
    number_examples=len(evalX)
    predicate_list = []
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {model.query: evalX[start:end][:,0],model.story: evalX[start:end][:,1:], model.dropout_keep_prob: 1}
        if not FLAGS.multi_label_flag:
            feed_dict[model.answer_single] = evalY[start:end]
        else:
            feed_dict[model.answer_multilabel] = evalY[start:end]
        curr_eval_loss, logits,curr_eval_acc,pred= sess.run([model.loss_val,model.logits,model.accuracy,model.predictions],feed_dict)#curr_eval_acc--->textCNN.accuracy
        predict_y = get_label_using_logits(logits)
        #print(logits)
        predicate_list += predict_y
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return predicate_list, eval_loss/float(eval_counter),eval_acc/float(eval_counter)

#get top5 predicted labels
def get_label_using_logits(logits,top_number=5):
    # index_list=np.argsort(logits)[-top_number:]
    #vindex_list=index_list[::-1]
    #y_predict_labels = [i for i in range(len(logits)) if logits[i] >= 0.50]  # TODO 0.5PW e.g.[2,12,13,10]
    y_predict_labels = list(np.argmax(logits,-1))

    return y_predict_labels

# calcuate accuracy
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

def load_data(cache_file_h5py,cache_file_pickle):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:",list(f_data.keys()))
    train_X=f_data['train_X'] # np.array(
    train_X_len = f_data['train_X_len'] 
    print("train_X.shape:",train_X.shape)
    train_Y=f_data['train_Y'] # np.array(
    print("train_Y.shape:",train_Y.shape,";")
    vaild_X=f_data['vaild_X'] # np.array(
    vaild_X_len = f_data['vaild_X_len'] 
    valid_Y=f_data['valid_Y'] # np.array(
    test_X=f_data['test_X'] # np.array(
    test_X_len = f_data['test_X_len'] 
    test_Y=f_data['test_Y'] # np.array(
    #print(train_X)
    #f_data.close()

    word2index, label2index=None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...",label2index)
    return word2index, label2index,train_X,train_X_len,train_Y,vaild_X,vaild_X_len,valid_Y,test_X,test_X_len,test_Y
if __name__ == "__main__":
    tf.app.run()
