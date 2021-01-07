import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os
import pandas
import nltk

from keras.models import load_model
import csv
import distance
from jellyfish._jellyfish import damerau_levenshtein_distance
from sklearn import metrics
from datetime import datetime, timedelta
from collections import Counter


sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

from dnc import DNC
from recurrent_controller import StatelessRecurrentController





def bleu_score(target_batch, predict_batch, print_prob=0.995):
    s=[]
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        for t in target_batch[b]:
            if t >1:
                trim_target.append(t)
        for t in predict_batch[b]:
            if t >1:
                trim_predict.append(t)
        if np.random.rand()>print_prob:
            print('{} vs {}'.format(trim_target, trim_predict))
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([trim_target], trim_predict, weights=[0.5,0.5])
        s.append(BLEUscore)
    return np.mean(s)

def set_score_pre(target_batch, predict_batch):
    s = []
    s2 = []
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        for t in target_batch[b]:
            if t > 1:
                trim_target.append(t)
        for t in predict_batch[b]:
            if t > 1:
                trim_predict.append(t)
        if np.random.rand()>0.999:
            print('{} vs {}'.format(trim_target, trim_predict))
        acc = len(set(trim_target).intersection(set(trim_predict)))/len(set(trim_target))
        acc2=0
        if len(set(trim_predict))>0:
            acc2 = len(set(trim_target).intersection(set(trim_predict))) / len(trim_predict)
        s.append(acc)
        s2.append(acc2)
    return np.mean(s), np.mean(s2)

def set_score_hist(target_batch, predict_batch):
    acc_label={}
    guess_label={}
    count_label={}

    for b in range(target_batch.shape[0]):
        for  t, t2 in zip(target_batch[b], predict_batch[b]):
            # print('{} ----- {}'.format(t, t2))
            trim_target = []
            for tt in t:
                if tt > 1:
                    trim_target.append(tt)
            for l in trim_target:
                if l not in count_label:
                    count_label[l]=0
                count_label[l]+=1

            trim_predict = []
            for tt in t2:
                if tt > 1:
                    trim_predict.append(tt)
            if np.random.rand()>0.99:
                print('{} vs {}'.format(trim_target, trim_predict))

            for l in trim_predict:
                if l not in guess_label:
                    guess_label[l]=0
                guess_label[l]+=1

            correct = list(set(trim_target).intersection(set(trim_predict)))
            for c in correct:
                if c not in acc_label:
                    acc_label[c]=0
                acc_label[c]+=1
    recall=[]
    precision=[]
    fscore=[]
    for k,v in sorted(count_label.items()):
        if k in acc_label:
            rec = acc_label[k] / count_label[k]
            prec= acc_label[k] / guess_label[k]
            recall.append(rec)
            precision.append(prec)
            fscore.append(2*rec*prec/(rec+prec))

        else:
            recall.append(0)
            precision.append(0)
            fscore.append(0)
    return recall, precision, fscore

def batch_mae(reals, preds, pprint=0.999):
    avgs=0
    c=0
    for i,real in enumerate(reals):
        if np.random.rand() > pprint:
            print('{} vs {}'.format(reals[i], preds[i]))
        for r, p in zip(reals[i],preds[i]):
            avgs += np.abs(r-p)
        c+=1

    return avgs/c


import editdistance as ed

def batch_norm_edit_score(reals, preds, pprint=0.999):
    avgs=0
    c=0
    for i,real in enumerate(reals):
        avgs += norm_edit_score(reals[i],preds[i],pprint)
        c+=1
    return avgs/c

def norm_edit_score(real, pred, pprob=0.999):

    trimpred=[]
    for p in pred:
        if p>1:
            trimpred.append(p)
    trimreal=[]
    for r in real:
        if r>1:
            trimreal.append(r)
    if np.random.rand() > pprob:
        print('{} vs {}'.format(trimreal, trimpred))
    if trimpred is []:
        return 1
    #print(trimreal)
    return ed.eval(trimpred,trimreal)/max(len(trimpred),len(trimreal))

def norm_edit_score_raw(real, pred, pprob=0.999):

    if np.random.rand() > pprob:
        print('{} vs {}'.format(real, pred))
    return ed.eval(pred,real)/max(len(pred),len(real))

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    # print('-----')
    # print(index)
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec

def prepare_sample(dig_list, proc_list, word_space_size_input, word_space_size_output, index=-1):
    if index<0:
        index = int(np.random.choice(len(dig_list),1))

    # print('\n{}'.format(index))
    ins=dig_list[index]
    ose=proc_list[index]
    seq_len = len(ins) + 1 + len(ose)
    input_vec = np.zeros(seq_len)
    for iii, token in enumerate(ins):
        input_vec[iii] = token
    input_vec[len(ins)] = 1
    output_vec = np.zeros(seq_len)
    decoder_point = len(ins) + 1
    for iii, token in enumerate(ose):
        output_vec[decoder_point + iii] = token
    input_vec = np.array([[onehot(code, word_space_size_input) for code in input_vec]])
    output_vec = np.array([[onehot(code, word_space_size_output) for code in output_vec]])
    return input_vec, output_vec, seq_len, decoder_point, index

def prepare_sample_batch(dig_list,proc_list,word_space_size_input,word_space_size_output, bs, lm_train=False):
    if isinstance(bs, int):
        indexs = np.random.choice(len(dig_list),bs,replace=False)
    else:
        #print('from {} to {}'.format(bs[0],bs[1]))
        indexs=list(range(bs[0],bs[1]))
    minlen=0
    moutlne=0
    for index in indexs:
        minlen=max(len(dig_list[index]),minlen)
        moutlne = max(len(proc_list[index]+[0]), moutlne)
    # moutlne*=2
    input_vecs=[]
    output_vecs=[]
    seq_len = minlen + 1 + moutlne
    decoder_point = minlen + 1
    out_list=[]
    masks=[]
    for index in indexs:
        # print('\n{}'.format(index))
        ins=dig_list[index]
        ose=proc_list[index]+[0]
        out_list.append(ose)
        input_vec = np.zeros(seq_len)
        output_vec = np.zeros(seq_len)
        mask=np.zeros(seq_len, dtype=np.bool)
        for iii, token in enumerate(ins):
            input_vec[minlen-len(ins)+iii] = token
            if lm_train:
                output_vec[minlen - len(ins) + iii+1] = token
                mask[minlen - len(ins) + iii+1] = True
        input_vec[minlen] = 1




        for iii, token in enumerate(ose):
            output_vec[decoder_point + iii] = token
            mask[decoder_point + iii]=True

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')
        input_vec = [onehot(code, word_space_size_input) for code in input_vec]
        output_vec = [onehot(code, word_space_size_output) for code in output_vec]
        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
        masks.append(mask)

    # raise False
    return np.asarray(input_vecs), np.asarray(output_vecs), seq_len, decoder_point, np.asarray(masks), out_list

def prepare_sample_batch_feature(X,y, bs):
    if isinstance(bs, int):
        indexs = np.random.choice(len(X),bs,replace=False)
    else:
        #print('from {} to {}'.format(bs[0],bs[1]))
        indexs=list(range(bs[0],bs[1]))


    minlen=X.shape[1]
    moutlen=y.shape[1]
    # moutlne*=2
    input_vecs=[]
    output_vecs=[]
    seq_len = minlen + 1 + moutlen
    decoder_point = minlen + 1
    out_list=[]
    masks=[]

    for index in indexs:
        # print('\n{}'.format(index))
        ins=X[index]
        if y.shape[2]>1:
            ose=np.zeros((moutlen, y.shape[2]+2))
            ose[:moutlen,2:]=y[index]
            ro=[]
            for l in range(moutlen):
                ro.append(np.argmax(ose[l],axis=-1))
        else:
            ose = np.zeros((moutlen, 1))
            ose[:moutlen] = y[index]
            ro = []
            for l in range(moutlen):
                ro.append(y[index][l])
        # print(ro)
        # print(y[index])
        # print(ose[:moutlen])
        # print(np.argmax(ose[l],axis=-1))
        # raise  False
        out_list.append(ro)
        input_vec = np.zeros((seq_len, X.shape[2]))
        if y.shape[2]>1:
            output_vec = np.zeros((seq_len, y.shape[2]+2))
        else:
            output_vec = np.zeros((seq_len,1))
        mask=np.zeros(seq_len, dtype=np.bool)
        for iii, token in enumerate(ins):
            input_vec[minlen-len(ins)+iii] = token
        input_vec[minlen] = np.ones(X.shape[2])




        for iii, token in enumerate(ose):
            output_vec[decoder_point + iii] = token
            mask[decoder_point + iii]=True

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')
        #
        # raise  False

        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
        masks.append(mask)

    # raise False
    return np.asarray(input_vecs), np.asarray(output_vecs), seq_len, decoder_point, np.asarray(masks), out_list


def prepare_sample_batch_feature_mix(X,y1,y2,bs):
    if isinstance(bs, int):
        indexs = np.random.choice(len(X),bs,replace=False)
    else:
        #print('from {} to {}'.format(bs[0],bs[1]))
        indexs=list(range(bs[0],bs[1]))


    minlen=X.shape[1]
    moutlen=y1.shape[1]
    # moutlne*=2
    input_vecs=[]
    output_vecs=[]
    seq_len = minlen + 1 + moutlen
    decoder_point = minlen + 1
    out_list1 = []
    out_list2 = []
    masks=[]

    for index in indexs:
        # print('\n{}'.format(index))
        ins=X[index]
        ose1=np.zeros((moutlen, y1.shape[2]+2))
        ose1[:moutlen,2:]=y1[index]
        ro1=[]
        for l in range(moutlen):
            ro1.append(np.argmax(ose1[l],axis=-1))
        ose2 = np.zeros((moutlen, 1))
        ose2[:moutlen] = y2[index]
        ro2 = []
        for l in range(moutlen):
            ro2.append(y2[index][l])
        # print(ro)
        # print(y[index])
        # print(ose[:moutlen])
        # print(np.argmax(ose[l],axis=-1))
        # raise  False
        out_list1.append(ro1)
        out_list2.append(ro2)
        input_vec = np.zeros((seq_len, X.shape[2]))
        output_vec1 = np.zeros((seq_len, y1.shape[2]+2))
        output_vec2 = np.zeros((seq_len,1))
        mask=np.zeros(seq_len, dtype=np.bool)
        for iii, token in enumerate(ins):
            input_vec[minlen-len(ins)+iii] = token
        input_vec[minlen] = np.ones(X.shape[2])




        for iii, token in enumerate(ose1):
            output_vec1[decoder_point + iii] = token
            mask[decoder_point + iii]=True
        for iii, token in enumerate(ose2):
            output_vec2[decoder_point + iii] = token

        output_vec = np.concatenate([output_vec1,output_vec2],axis=-1)

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')
        #
        # raise  False

        input_vecs.append(input_vec)
        output_vecs.append(output_vec)
        masks.append(mask)

    # raise False
    return np.asarray(input_vecs), np.asarray(output_vecs), seq_len, decoder_point, np.asarray(masks), out_list1, out_list2

def load_dict(dir='./data/BusinessProcess/Moodle'):
    return pickle.load(open(dir+'/event_vocab.pkl','rb'))

def load_single_sequence(fname):
    seqs=[]
    rl=''
    for l in open(fname):
        if l.strip()[-1]==']':
            if rl!='':
                l=rl
            s=l.strip()[1:-1].strip().split()
            seqs.append([int(x)+1 for x in s])
            rl=''
        else:
            rl+=l+' '
    return seqs

def load_sequence(dir='./data/BusinessProcess/Moodle/'):
    train_in=dir+'/train_prefixes.txt'
    train_out=dir+'/train_suffixes.txt'
    test_in = dir + '/test_prefixes.txt'
    test_out = dir + '/test_suffixes.txt'

    str_in=load_single_sequence(train_in)
    strain_out = load_single_sequence(train_out)
    stest_in = load_single_sequence(test_in)
    stest_out = load_single_sequence(test_out)

    return str_in, strain_out, stest_in, stest_out

def moodle_task():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_moodle')
    batch_size = 10

    _,_,_,char2label=load_dict()

    str_in, strain_out, stest_in, stest_out = load_sequence()

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))


    input_size = len(char2label)+1
    output_size = len(char2label)+1
    sequence_max_length = 100

    words_count = 64
    word_size = 100
    read_heads = 1

    learning_rate = 1e-4
    momentum = 0.9

    iterations = 150000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer()
            # summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,hidden_controller_dim=100,
                use_emb=False,
                use_mem=False,
                decoder_mode=True,
                dual_controller=False,
                write_protect=False,
                dual_emb=True
            )

            output,prob,loss,apply_gradients=ncomputer.build_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            train_writer = tf.summary.FileWriter('./data/log_moddle/', session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_vec, output_vec, seq_len, decoder_point, masks, _ = \
                        prepare_sample_batch(str_in, strain_out, input_size, output_size, bs=batch_size, lm_train=False)




                    summerize = (i % 500 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)
                    if i<=1000000:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                        })

                    last_100_losses.append(loss_value)

                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores = []
                        for ii in range(10):
                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list =\
                                prepare_sample_batch(str_in, strain_out, input_size, output_size, bs=batch_size, lm_train=False)


                            out = session.run([prob], feed_dict={ncomputer.input_data: input_vec,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.mask:masks})
                            out = np.reshape(np.asarray(out), [-1, seq_len,output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list=[]
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]==0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            trscores.append(batch_norm_edit_score(rout_list, bout_list,0.95))



                        print('-----')

                        tescores = []
                        tescores2 = []
                        tescores3 = []
                        big_out_list = []
                        losses = []
                        ntb=len(stest_in)//batch_size+1
                        for ii in range(ntb):
                            if ii*batch_size==len(stest_in):
                                break
                            bs=[ii*batch_size, min((ii+1)*batch_size,len(stest_in))]
                            rs = bs[1] - bs[0]
                            if bs[1]>=len(stest_in):
                                bs=[len(stest_in)-batch_size, len(stest_in)]

                            input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                                prepare_sample_batch(stest_in, stest_out, input_size, output_size, bs, lm_train=False)
                            out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_data: input_data,
                                                                             ncomputer.target_output: target_output,
                                                                             ncomputer.sequence_length: seq_len,
                                                                             ncomputer.decoder_point: decoder_point,
                                                                             ncomputer.mask:masks
                                                                            })

                            losses.append(loss_v)
                            out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list = []

                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]==0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            tescores.append(batch_norm_edit_score(rout_list[:rs], bout_list[:rs], 0.995))
                            #pre, rec = set_score_pre(np.asarray([stest_out[ii]]), np.asarray([out_list]))
                            #tescores2.append(pre)
                            #tescores3.append(rec)
                            # print(pro_list_test)
                            # print(big_out_list)
                            # rec, pre, fsc = set_score_hist(np.asarray([pro_list_test]),np.asarray([big_out_list]))
                        tloss = np.mean(losses)
                        tscore = np.mean(tescores)
                        print('tr bleu {} vs te bleu {}'.format(np.mean(trscores), np.mean(tescores)))
                        #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
                        print('test loss {}'.format(tloss))
                        summary.value.add(tag='train_bleu', simple_value=np.mean(trscores))
                        summary.value.add(tag='test_bleu', simple_value=np.mean(tescores))
                        summary.value.add(tag='test_loss', simple_value=tloss)
                        #summary.value.add(tag='test_recall', simple_value=np.mean(tescores2))
                        #summary.value.add(tag='test_precision', simple_value=np.mean(tescores3))
                        train_writer.add_summary(summary, i)
                        train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if minloss>tloss:
                        minloss=tloss
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())


                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... "),


def write_predict(wfile, list_pred):
    with open(wfile,'w') as f:
        for p in list_pred:
            f.write('[')
            for n in p[:-1]:
                f.write(str(n-1)+' ')
            f.write(str(p[-1]-1))
            f.write(']')
            f.write('\n')

def moodle_test():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_moodle')
    batch_size = 10

    main_dir='./data/BusinessProcess/Moodle/'

    _,_,_,char2label=load_dict()

    str_in, strain_out, stest_in, stest_out = load_sequence(main_dir)

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))


    input_size = len(char2label)+1
    output_size = len(char2label)+1
    sequence_max_length = 100

    words_count = 64
    word_size = 100
    read_heads = 1

    learning_rate = 1e-4
    momentum = 0.9

    iterations = 150000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer()
            # summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=False,
                write_protect=False,
                dual_emb=True
            )

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())


            print('-----')

            tescores = []
            losses = []
            ntb=len(stest_in)//batch_size+1
            all_preds=[]
            for ii in range(ntb):
                if ii*batch_size==len(stest_in):
                    break
                bs=[ii*batch_size, min((ii+1)*batch_size,len(stest_in))]
                rs = bs[1] - bs[0]
                if bs[1]>=len(stest_in):
                    bs=[len(stest_in)-batch_size, len(stest_in)]

                input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                    prepare_sample_batch(stest_in, stest_out, input_size, output_size, bs)
                out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_data: input_data,
                                                                 ncomputer.target_output: target_output,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.mask:masks
                                                                })

                losses.append(loss_v)
                out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                out = np.argmax(out, axis=-1)
                bout_list = []

                for b in range(out.shape[0]):
                    out_list = []
                    for io in range(decoder_point, out.shape[1]):
                        if out[b][io]==0:
                            break
                        out_list.append(out[b][io])
                    bout_list.append(out_list)

                for bb in bout_list[:rs]:
                    all_preds.append(bb)

                tescores.append(batch_norm_edit_score(rout_list[:rs], bout_list[:rs], 0.995))
            prefile=main_dir+'/{}.test_predict.txt'.format(ncomputer.print_config())
            write_predict(prefile, all_preds)
            predict_seq=load_single_sequence(prefile)
            tescores2=[]
            for rseq,pseq in zip(stest_out, predict_seq):
                s=norm_edit_score(rseq,pseq)
                tescores2.append(s)

            tloss = np.mean(losses)
            print('test ed {}'.format(np.mean(tescores)))
            print('test ed {}'.format(np.mean(tescores2)))

            #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
            print('test loss {}'.format(tloss))





def financial_log_task():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_financial_log')
    batch_size = 10

    _,_,_,char2label=load_dict('./data/BusinessProcess/Financial_Log/')

    str_in, strain_out, stest_in, stest_out = load_sequence('./data/BusinessProcess/Financial_Log/')

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))


    input_size = len(char2label)+1
    output_size = len(char2label)+1
    sequence_max_length = 100

    words_count = 64
    word_size = 64
    read_heads = 1

    learning_rate = 1e-4
    momentum = 0.9

    iterations = 10000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer()
            # summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim = 100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                use_teacher=True
            )



            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()
            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            train_writer = tf.summary.FileWriter('./data/log_financial_log/', session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_vec, output_vec, seq_len, decoder_point, masks, _ = \
                        prepare_sample_batch(str_in, strain_out, input_size, output_size, bs=batch_size, lm_train=False)




                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)
                    if i<=1000000:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks,
                            ncomputer.teacher_force:ncomputer.get_bool_rand_curriculum(seq_len, i/100, type='sig')
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks,
                            ncomputer.teacher_force: ncomputer.get_bool_rand(seq_len,0)
                        })

                    last_100_losses.append(loss_value)

                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores = []
                        for ii in range(10):
                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list =\
                                prepare_sample_batch(str_in, strain_out, input_size, output_size, bs=batch_size, lm_train=False)


                            out = session.run([prob], feed_dict={ncomputer.input_data: input_vec,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.mask:masks,
                                                                 ncomputer.target_output: output_vec,
                                                                 ncomputer.teacher_force: ncomputer.get_bool_rand(seq_len,0)})
                            out = np.reshape(np.asarray(out), [-1, seq_len,output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list=[]
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]==0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            trscores.append(batch_norm_edit_score(rout_list, bout_list,0.95))



                        print('-----')

                        tescores = []
                        tescores2 = []
                        tescores3 = []
                        big_out_list = []
                        losses = []
                        ntb=len(stest_in)//batch_size+1
                        for ii in range(ntb):
                            if ii*batch_size==len(stest_in):
                                break
                            bs=[ii*batch_size, min((ii+1)*batch_size,len(stest_in))]
                            rs = bs[1] - bs[0]
                            if bs[1]>=len(stest_in):
                                bs=[len(stest_in)-batch_size, len(stest_in)]

                            input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                                prepare_sample_batch(stest_in, stest_out, input_size, output_size, bs, lm_train=False)
                            out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_data: input_data,
                                                                             ncomputer.target_output: target_output,
                                                                             ncomputer.sequence_length: seq_len,
                                                                             ncomputer.decoder_point: decoder_point,
                                                                             ncomputer.mask:masks,
                                                                             ncomputer.teacher_force: ncomputer.get_bool_rand(seq_len,0)
                                                                            })

                            losses.append(loss_v)
                            out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list = []

                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]==0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            tescores.append(batch_norm_edit_score(rout_list[:rs], bout_list[:rs], 0.995))
                            #pre, rec = set_score_pre(np.asarray([stest_out[ii]]), np.asarray([out_list]))
                            #tescores2.append(pre)
                            #tescores3.append(rec)
                            # print(pro_list_test)
                            # print(big_out_list)
                            # rec, pre, fsc = set_score_hist(np.asarray([pro_list_test]),np.asarray([big_out_list]))
                        tloss = np.mean(losses)
                        tscore = np.mean(tescores)
                        print('tr bleu {} vs te bleu {}'.format(np.mean(trscores), np.mean(tescores)))
                        #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
                        print('test loss {}'.format(tloss))
                        summary.value.add(tag='train_bleu', simple_value=np.mean(trscores))
                        summary.value.add(tag='test_bleu', simple_value=np.mean(tescores))
                        summary.value.add(tag='test_loss', simple_value=tloss)
                        #summary.value.add(tag='test_recall', simple_value=np.mean(tescores2))
                        #summary.value.add(tag='test_precision', simple_value=np.mean(tescores3))
                        train_writer.add_summary(summary, i)
                        train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if minloss>tscore:
                        minloss=tscore
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())


                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... "),


def financial_log_test():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_Financial_Log')
    batch_size = 10

    main_dir='./data/BusinessProcess/Financial_Log/'

    _, _, _, char2label = load_dict('./data/BusinessProcess/Financial_Log/')

    str_in, strain_out, stest_in, stest_out = load_sequence(main_dir)

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))


    input_size = len(char2label)+1
    output_size = len(char2label)+1
    sequence_max_length = 100

    words_count = 64
    word_size = 64
    read_heads = 1

    learning_rate = 1e-4
    momentum = 0.9

    iterations = 150000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer()
            # summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                dual_emb=True
            )

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()
            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())


            print('-----')

            tescores = []
            losses = []
            ntb=len(stest_in)//batch_size+1
            all_preds=[]
            for ii in range(ntb):
                if ii*batch_size==len(stest_in):
                    break
                bs=[ii*batch_size, min((ii+1)*batch_size,len(stest_in))]
                rs = bs[1] - bs[0]
                if bs[1]>=len(stest_in):
                    bs=[len(stest_in)-batch_size, len(stest_in)]

                input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                    prepare_sample_batch(stest_in, stest_out, input_size, output_size, bs)
                out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_data: input_data,
                                                                 ncomputer.target_output: target_output,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.mask:masks
                                                                })

                losses.append(loss_v)
                out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                out = np.argmax(out, axis=-1)
                bout_list = []

                for b in range(out.shape[0]):
                    out_list = []
                    for io in range(decoder_point, out.shape[1]):
                        if out[b][io]==0:
                            break
                        out_list.append(out[b][io])
                    bout_list.append(out_list)

                for bb in bout_list[:rs]:
                    all_preds.append(bb)

                tescores.append(batch_norm_edit_score(rout_list[:rs], bout_list[:rs], 0.995))
            prefile=main_dir+'/{}.test_predict.txt'.format(ncomputer.print_config())
            write_predict(prefile, all_preds)
            predict_seq=load_single_sequence(prefile)
            tescores2=[]
            for rseq,pseq in zip(stest_out, predict_seq):
                s=norm_edit_score(rseq,pseq)
                tescores2.append(s)

            tloss = np.mean(losses)
            print('test ed {}'.format(np.mean(tescores)))
            print('test ed {}'.format(np.mean(tescores2)))

            #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
            print('test loss {}'.format(tloss))


def ibm_task():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_ibm')
    batch_size = 10

    _,_,_,char2label=load_dict('./data/BusinessProcess/IBM_Anonymous/')

    str_in, strain_out, stest_in, stest_out = load_sequence('./data/BusinessProcess/IBM_Anonymous/')

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))


    input_size = len(char2label)+1
    output_size = len(char2label)+1
    sequence_max_length = 100

    words_count = 64
    word_size = 100
    read_heads = 1

    learning_rate = 1e-4
    momentum = 0.9

    iterations = 50000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer()
            # summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim = 100,
                use_emb=False,
                use_mem=False,
                decoder_mode=True,
                dual_controller=False,
                write_protect=False,
            )

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            train_writer = tf.summary.FileWriter('./data/log_ibm/', session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_vec, output_vec, seq_len, decoder_point, masks, _ = \
                        prepare_sample_batch(str_in, strain_out, input_size, output_size, bs=batch_size, lm_train=True)





                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)
                    if i<=1000000:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })

                    last_100_losses.append(loss_value)

                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores = []
                        for ii in range(10):
                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list =\
                                prepare_sample_batch(str_in, strain_out, input_size, output_size, bs=batch_size, lm_train=True)


                            out = session.run([prob], feed_dict={ncomputer.input_data: input_vec,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.mask:masks})
                            out = np.reshape(np.asarray(out), [-1, seq_len,output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list=[]
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]==0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            trscores.append(batch_norm_edit_score(rout_list, bout_list,0.95))



                        print('-----')

                        tescores = []
                        tescores2 = []
                        tescores3 = []
                        big_out_list = []
                        losses = []
                        ntb=len(stest_in)//batch_size+1
                        for ii in range(ntb):
                            if ii*batch_size==len(stest_in):
                                break
                            bs=[ii*batch_size, min((ii+1)*batch_size,len(stest_in))]
                            rs = bs[1] - bs[0]
                            if bs[1]>=len(stest_in):
                                bs=[len(stest_in)-batch_size, len(stest_in)]

                            input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                                prepare_sample_batch(stest_in, stest_out, input_size, output_size, bs, lm_train=True)
                            out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_data: input_data,
                                                                             ncomputer.target_output: target_output,
                                                                             ncomputer.sequence_length: seq_len,
                                                                             ncomputer.decoder_point: decoder_point,
                                                                             ncomputer.mask:masks
                                                                            })

                            losses.append(loss_v)
                            out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list = []

                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]==0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            tescores.append(batch_norm_edit_score(rout_list[:rs], bout_list[:rs], 0.995))
                            #pre, rec = set_score_pre(np.asarray([stest_out[ii]]), np.asarray([out_list]))
                            #tescores2.append(pre)
                            #tescores3.append(rec)
                            # print(pro_list_test)
                            # print(big_out_list)
                            # rec, pre, fsc = set_score_hist(np.asarray([pro_list_test]),np.asarray([big_out_list]))
                        tloss = np.mean(losses)
                        tscore = np.mean(tescores)
                        print('tr bleu {} vs te bleu {}'.format(np.mean(trscores), np.mean(tescores)))
                        #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
                        print('test loss {}'.format(tloss))
                        summary.value.add(tag='train_bleu', simple_value=np.mean(trscores))
                        summary.value.add(tag='test_bleu', simple_value=np.mean(tescores))
                        summary.value.add(tag='test_loss', simple_value=tloss)
                        #summary.value.add(tag='test_recall', simple_value=np.mean(tescores2))
                        #summary.value.add(tag='test_precision', simple_value=np.mean(tescores3))
                        train_writer.add_summary(summary, i)
                        train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if minloss>tscore:
                        minloss=tscore
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())


                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... ")

def ibm_test():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_ibm')
    batch_size = 10

    main_dir='./data/BusinessProcess/IBM_Anonymous/'

    _, _, _, char2label = load_dict('./data/BusinessProcess/IBM_Anonymous/')

    str_in, strain_out, stest_in, stest_out = load_sequence(main_dir)

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))


    input_size = len(char2label)+1
    output_size = len(char2label)+1
    sequence_max_length = 100

    words_count = 64
    word_size = 100
    read_heads = 1

    learning_rate = 1e-4
    momentum = 0.9

    iterations = 150000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer()
            # summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=False,
                write_protect=False,
                dual_emb=True
            )

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()
            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())


            print('-----')

            tescores = []
            losses = []
            ntb=len(stest_in)//batch_size+1
            all_preds=[]
            for ii in range(ntb):
                if ii*batch_size==len(stest_in):
                    break
                bs=[ii*batch_size, min((ii+1)*batch_size,len(stest_in))]
                rs = bs[1] - bs[0]
                if bs[1]>=len(stest_in):
                    bs=[len(stest_in)-batch_size, len(stest_in)]

                input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                    prepare_sample_batch(stest_in, stest_out, input_size, output_size, bs)
                out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_data: input_data,
                                                                 ncomputer.target_output: target_output,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.mask:masks
                                                                })

                losses.append(loss_v)
                out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                out = np.argmax(out, axis=-1)
                bout_list = []

                for b in range(out.shape[0]):
                    out_list = []
                    for io in range(decoder_point, out.shape[1]):
                        if out[b][io]==0:
                            break
                        out_list.append(out[b][io])
                    bout_list.append(out_list)

                for bb in bout_list[:rs]:
                    all_preds.append(bb)

                tescores.append(batch_norm_edit_score(rout_list[:rs], bout_list[:rs], 0.995))
            prefile=main_dir+'/{}.test_predict.txt'.format(ncomputer.print_config())
            write_predict(prefile, all_preds)
            predict_seq=load_single_sequence(prefile)
            tescores2=[]
            for rseq,pseq in zip(stest_out, predict_seq):
                s=norm_edit_score(rseq,pseq)
                tescores2.append(s)

            tloss = np.mean(losses)
            print('test ed {}'.format(np.mean(tescores)))
            print('test ed {}'.format(np.mean(tescores2)))

            #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
            print('test loss {}'.format(tloss))


def load_np_data(dir):
    X=pickle.load(open(dir+'/ps_X.pkl','rb'),encoding='latin1')
    y_a = pickle.load(open(dir + '/ps_ya.pkl','rb'),encoding='latin1')
    y_t = pickle.load(open(dir + '/ps_yt.pkl','rb'),encoding='latin1')

    return X, y_a, y_t



def help_desk_task():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_help_desk')
    batch_size = 15

    X, y_a, y_t = load_np_data('./data/help_desk/')

    if len(y_a.shape)==2:
        y_a = np.reshape(y_a,[y_a.shape[0],1,y_a.shape[1]])


    all_index = list(range(len(X)))

    # np.random.shuffle(all_index)

    train_index = all_index[:int(len(X) * 0.8)]
    test_index = all_index[int(len(X) * 0.8):]


    X_train = X[train_index]
    X_test = X[test_index]

    y_a_train = y_a[train_index]
    y_a_test = y_a[test_index]

    y_t_train = y_t[train_index]
    y_t_test = y_t[test_index]

    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))


    input_size = dim_in
    output_size = dim_out +2
    sequence_max_length = 100

    words_count = 20
    word_size = 100
    read_heads = 4



    iterations = 10000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim = 150,
                use_emb=False,
                use_mem=True,
                decoder_mode=False,
                dual_controller=False,
                write_protect=False,
                attend_dim=0
            )

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())

            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            train_writer = tf.summary.FileWriter('./data/log_help_desk/', session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_vec, output_vec, seq_len, decoder_point, masks, rout_list = \
                        prepare_sample_batch_feature(X_train, y_a_train, bs=batch_size)




                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)
                    if i<=1000000:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })

                    last_100_losses.append(loss_value)

                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores = []
                        for ii in range(10):
                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list =\
                                prepare_sample_batch_feature(X_train, y_a_train, bs=batch_size)


                            out = session.run([prob], feed_dict={ncomputer.input_data: input_vec,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.mask:masks})
                            out = np.reshape(np.asarray(out), [-1, seq_len,output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list=[]
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]==0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)
                            # print(rout_list)
                            # print(bout_list)
                            # raise False
                            trscores.append(batch_norm_edit_score(rout_list, bout_list,0.95))



                        print('-----')

                        tescores = []
                        tescores2 = []
                        tescores3 = []
                        big_out_list = []
                        losses = []
                        ntb=len(X_test)//batch_size+1
                        for ii in range(ntb):
                            if ii*batch_size==len(X_test):
                                break
                            bs=[ii*batch_size, min((ii+1)*batch_size,len(X_test))]
                            rs = bs[1] - bs[0]
                            if bs[1]>=len(X_test):
                                bs=[len(X_test)-batch_size, len(X_test)]

                            input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                                prepare_sample_batch_feature(X_test, y_a_test, bs)
                            out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_data: input_data,
                                                                             ncomputer.target_output: target_output,
                                                                             ncomputer.sequence_length: seq_len,
                                                                             ncomputer.decoder_point: decoder_point,
                                                                             ncomputer.mask:masks
                                                                            })

                            losses.append(loss_v)
                            out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list = []

                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]==0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            tescores.append(batch_norm_edit_score(rout_list[:rs], bout_list[:rs], 0.995))
                            pre, rec = set_score_pre(np.asarray(rout_list[:rs]), np.asarray(bout_list[:rs]))
                            tescores2.append(pre)
                            tescores3.append(rec)
                            # print(pro_list_test)
                            # print(big_out_list)
                            # rec, pre, fsc = set_score_hist(np.asarray([pro_list_test]),np.asarray([big_out_list]))
                        tloss = np.mean(losses)
                        tscore = np.mean(tescores)
                        print('tr edit {} vs te edit {}'.format(np.mean(trscores), np.mean(tescores)))
                        print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
                        print('test loss {}'.format(tloss))
                        summary.value.add(tag='test_prec', simple_value=np.mean(tescores2))
                        summary.value.add(tag='test_edit', simple_value=np.mean(tescores))
                        summary.value.add(tag='test_loss', simple_value=tloss)
                        #summary.value.add(tag='test_recall', simple_value=np.mean(tescores2))
                        #summary.value.add(tag='test_precision', simple_value=np.mean(tescores3))
                        train_writer.add_summary(summary, i)
                        train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if minloss>tscore:
                        minloss=tscore
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())


                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... ")

def help_desk_test():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_help_desk')
    batch_size = 15

    X, y_a, y_t = load_np_data('./data/help_desk/')

    if len(y_a.shape) < 3:
        y_a = np.reshape(y_a, [y_a.shape[0], 1, y_a.shape[1]])

    num_sample = len(X)
    X_train = X[:int(0.8 * num_sample)]
    X_test = X[int(0.8 * num_sample):]

    y_a_train = y_a[:int(0.8 * num_sample)]
    y_a_test = y_a[int(0.8 * num_sample):]

    y_t_train = y_t[:int(0.8 * num_sample)]
    y_t_test = y_t[int(0.8 * num_sample):]

    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    input_size = dim_in
    output_size = dim_out + 2
    sequence_max_length = 100

    words_count = 64
    word_size = 100
    read_heads = 1

    iterations = 5000
    start_step = 0

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=False,
                decoder_mode=True,
                dual_controller=True,
                write_protect=False,
            )

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())
            llprint("Done!\n")



            print('-----')

            tescores = []
            tescores2 = []
            tescores3 = []

            losses = []
            ntb = len(X_test) // batch_size + 1
            all_pred=[]
            all_real=[]
            for ii in range(ntb):
                if ii * batch_size == len(X_test):
                    break
                bs = [ii * batch_size, min((ii + 1) * batch_size, len(X_test))]
                rs = bs[1] - bs[0]
                if bs[1] >= len(X_test):
                    bs = [len(X_test) - batch_size, len(X_test)]

                input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                    prepare_sample_batch_feature(X_test, y_a_test, bs)
                out, loss_v = session.run([prob, loss], feed_dict={ncomputer.input_data: input_data,
                                                                   ncomputer.target_output: target_output,
                                                                   ncomputer.sequence_length: seq_len,
                                                                   ncomputer.decoder_point: decoder_point,
                                                                   ncomputer.mask: masks
                                                                   })

                losses.append(loss_v)
                out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                out = np.argmax(out, axis=-1)
                bout_list = []

                for b in range(out.shape[0]):
                    out_list = []
                    for io in range(decoder_point, out.shape[1]):
                        if out[b][io] == 0:
                            break
                        out_list.append(out[b][io])
                    bout_list.append(out_list)

                tescores.append(batch_norm_edit_score(rout_list[:rs], bout_list[:rs], 0.995))
                pre, rec = set_score_pre(np.asarray(rout_list[:rs]), np.asarray(bout_list[:rs]))
                tescores2.append(pre)
                tescores3.append(rec)
                all_pred+=bout_list[:rs]
                all_real+=rout_list[:rs]
                # print(pro_list_test)
                # print(big_out_list)
                # rec, pre, fsc = set_score_hist(np.asarray([pro_list_test]),np.asarray([big_out_list]))
            tloss = np.mean(losses)
            tscore = np.mean(tescores)
            print('te edit {}'.format(np.mean(tescores)))
            print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
            print('test loss {}'.format(tloss))

            correct=[]
            acc=0
            for p,r in zip(all_pred, all_real):
                if p==r:
                    correct.append(1)
                    acc+=1
                else:
                    correct.append(0)

            print('final acc {}'.format(acc/len(all_pred)))
            pd = pandas.DataFrame({'real':all_real,'predict':all_pred,'correct':correct})
            pd.to_csv(ckpts_dir+'/{}.help_desk_next_act.csv'.format(ncomputer.print_config()))


def exact_help_desk_test():
    chars = pickle.load(open('./data/help_desk/desk_tmp/chars.pkl', 'rb'), encoding='latin1')
    char_indices = pickle.load(open('./data/help_desk/desk_tmp/char_indices.pkl', 'rb'), encoding='latin1')
    target_indices_char = pickle.load(open('./data/help_desk/desk_tmp/target_indices_char.pkl', 'rb'), encoding='latin1')
    target_char_indices = pickle.load(open('./data/help_desk/desk_tmp/target_char_indices.pkl', 'rb'), encoding='latin1')
    target_chars = pickle.load(open('./data/help_desk/desk_tmp/target_chars.pkl', 'rb'), encoding='latin1')
    maxlen = pickle.load(open('./data/help_desk/desk_tmp/maxlen.pkl', 'rb'), encoding='latin1')
    divisor = pickle.load(open('./data/help_desk/desk_tmp/divisor.pkl', 'rb'), encoding='latin1')
    divisor2 = pickle.load(open('./data/help_desk/desk_tmp/divisor2.pkl', 'rb'), encoding='latin1')
    lines = pickle.load(open('./data/help_desk/desk_tmp/lines.pkl', 'rb'), encoding='latin1')
    lines_t = pickle.load(open('./data/help_desk/desk_tmp/lines_t.pkl', 'rb'), encoding='latin1')
    lines_t2 = pickle.load(open('./data/help_desk/desk_tmp/lines_t2.pkl', 'rb'), encoding='latin1')
    lines_t3 = pickle.load(open('./data/help_desk/desk_tmp/lines_t3.pkl', 'rb'), encoding='bytes')

    # lines_t4 = fold1_t4 + fold2_t4

    # set parameters
    predict_size = 1


    # define helper functions
    def encode(sentence, times, times3, maxlen=maxlen):
        num_features = len(chars) + 5
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence)
        times2 = np.cumsum(times)
        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight
            multiset_abstraction = Counter(sentence[:t + 1])
            for c in chars:
                if c == char:
                    X[0, t + leftpad, char_indices[c]] = 1
            X[0, t + leftpad, len(chars)] = t + 1
            X[0, t + leftpad, len(chars) + 1] = times[t] / divisor
            X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2
            X[0, t + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
            X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7
        return X

    def getSymbol(predictions):
        maxPrediction = 0
        symbol = ''
        i = 0;
        print(predictions)
        for prediction in predictions:
            if (prediction >= maxPrediction):
                maxPrediction = prediction
                symbol = target_indices_char[i]
            i += 1
        return symbol

    one_ahead_gt = []
    one_ahead_pred = []

    two_ahead_gt = []
    two_ahead_pred = []

    three_ahead_gt = []
    three_ahead_pred = []

    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_help_desk')
    batch_size = 1

    X, y_a, y_t = load_np_data('./data/help_desk/')

    if len(y_a.shape) < 3:
        y_a = np.reshape(y_a, [y_a.shape[0], 1, y_a.shape[1]])

    num_sample = len(X)
    X_train = X[:int(0.8 * num_sample)]
    X_test = X[int(0.8 * num_sample):]

    y_a_train = y_a[:int(0.8 * num_sample)]
    y_a_test = y_a[int(0.8 * num_sample):]

    y_t_train = y_t[:int(0.8 * num_sample)]
    y_t_test = y_t[int(0.8 * num_sample):]

    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    input_size = dim_in
    output_size = dim_out + 2
    sequence_max_length = 100
    test_data = []
    test_label = []
    words_count = 20
    word_size = 100
    read_heads = 4

    iterations = 5000
    start_step = 0

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:
            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=150,
                use_emb=False,
                use_mem=True,
                decoder_mode=False,
                dual_controller=False,
                write_protect=False,
                attend_dim=0
            )

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())
            llprint("Done!\n")

            print('-----')

            eventlog = "helpdesk.csv"
            # make predictions
            with open('./data/help_desk/next_activity_and_time_%s' % eventlog, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(
                    ["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times",
                     "Predicted times", "RMSE", "MAE", "Median AE"])
                for prefix_size in range(2, maxlen):
                    print(prefix_size)
                    for line, times, times3 in zip(lines, lines_t, lines_t3):
                        times.append(0)
                        cropped_line = ''.join(line[:prefix_size])
                        cropped_times = times[:prefix_size]
                        cropped_times3 = times3[:prefix_size]
                        if '!' in cropped_line:
                            continue  # make no prediction for this case, since this case has ended already
                        ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                        ground_truth_t = times[prefix_size:prefix_size + predict_size]
                        predicted = ''
                        predicted_t = []
                        for i in range(predict_size):
                            if len(ground_truth) <= i:
                                continue
                            enc = encode(cropped_line, cropped_times, cropped_times3)
                            test_data.append(enc)
                            input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                                prepare_sample_batch_feature(enc, np.asarray([y_a_test[0]]), 1)
                            out  = session.run(prob, feed_dict={ncomputer.input_data: input_data,
                                                                               ncomputer.sequence_length: seq_len,
                                                                               ncomputer.decoder_point: decoder_point,
                                                                               ncomputer.mask: masks
                                                                               })
                            out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                            out = np.argmax(out, axis=-1)
                            bout = []
                            for io in range(decoder_point, out.shape[1]):
                                bout.append(max(out[0][io]-2,0))
                            y_char = bout[0]
                            # print(y_char)
                            y_t = 0
                            prediction = target_indices_char[y_char]
                            cropped_line += prediction
                            if y_t < 0:
                                y_t = 0
                            cropped_times.append(y_t)
                            y_t = y_t * divisor
                            cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                            predicted_t.append(y_t)
                            if i == 0:
                                if len(ground_truth_t) > 0:
                                    one_ahead_pred.append(y_t)
                                    one_ahead_gt.append(ground_truth_t[0])
                            if i == 1:
                                if len(ground_truth_t) > 1:
                                    two_ahead_pred.append(y_t)
                                    two_ahead_gt.append(ground_truth_t[1])
                            if i == 2:
                                if len(ground_truth_t) > 2:
                                    three_ahead_pred.append(y_t)
                                    three_ahead_gt.append(ground_truth_t[2])
                            if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                                print('! predicted, end case')
                                break
                            predicted += prediction
                        output = []
                        if len(ground_truth) > 0:
                            output.append(prefix_size)
                            output.append((ground_truth).encode("utf-8"))
                            output.append((predicted).encode("utf-8"))
                            output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                            dls = 1 - (
                            damerau_levenshtein_distance((predicted), (ground_truth)) / max(len(predicted), len(ground_truth)))
                            if dls < 0:
                                dls = 0  # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                            output.append(dls)
                            output.append(1 - distance.jaccard(predicted, ground_truth))
                            output.append('; '.join(str(x) for x in ground_truth_t))
                            output.append('; '.join(str(x) for x in predicted_t))
                            if len(predicted_t) > len(
                                    ground_truth_t):  # if predicted more events than length of case, only use needed number of events for time evaluation
                                predicted_t = predicted_t[:len(ground_truth_t)]
                            if len(ground_truth_t) > len(
                                    predicted_t):  # if predicted less events than length of case, put 0 as placeholder prediction
                                predicted_t.extend(range(len(ground_truth_t) - len(predicted_t)))
                            if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                                output.append('')
                                output.append(metrics.mean_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                                output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                            else:
                                output.append('')
                                output.append('')
                                output.append('')
                            spamwriter.writerow(output)


def help_desk_task_time():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_help_desk_time')
    batch_size = 15

    X, y_a, y_t = load_np_data('./data/help_desk/')

    if len(y_a.shape)<3:
        y_a= np.reshape(y_a,[y_a.shape[0],1,y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])

    all_index = list(range(len(X)))

    # np.random.shuffle(all_index)

    train_index = all_index[:int(len(X) * 0.8)]
    test_index = all_index[int(len(X) * 0.8):]


    X_train = X[train_index]
    X_test = X[test_index]

    y_a_train = y_a[train_index]
    y_a_test = y_a[test_index]

    y_t_train = y_t[train_index]
    y_t_test = y_t[test_index]



    dim_in = X_train.shape[2]
    dim_out = 1

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))


    input_size = dim_in
    output_size = 1
    sequence_max_length = 100

    words_count = 20
    word_size = 20
    read_heads = 1



    iterations = 10000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim = 100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=False,
                attend_dim=0
            )

            moutput, loss, apply_gradients = ncomputer.build_mae_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())

            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            train_writer = tf.summary.FileWriter('./data/log_help_desk/', session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_vec, output_vec, seq_len, decoder_point, masks, rout_list = \
                        prepare_sample_batch_feature(X_train, y_t_train, bs=batch_size)




                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)
                    if i<=1000000:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })

                    last_100_losses.append(loss_value)

                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores = []
                        for ii in range(10):
                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list =\
                                prepare_sample_batch_feature(X_train, y_t_train, bs=batch_size)


                            out = session.run([moutput], feed_dict={ncomputer.input_data: input_vec,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.mask:masks})

                            out = np.reshape(np.asarray(out), [-1, seq_len])
                            rout_list = np.reshape(np.asarray(rout_list),[-1,y_t_train.shape[1]])
                            bout_list=[]
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)
                            # print(rout_list)
                            # print(bout_list)
                            # raise False
                            trscores.append(batch_mae(rout_list, bout_list,0.95))



                        print('-----')

                        tescores = []
                        tescores2 = []
                        tescores3 = []
                        big_out_list = []
                        losses = []
                        ntb=len(X_test)//batch_size+1
                        for ii in range(ntb):
                            if ii*batch_size==len(X_test):
                                break
                            bs=[ii*batch_size, min((ii+1)*batch_size,len(X_test))]
                            rs = bs[1] - bs[0]
                            if bs[1]>=len(X_test):
                                bs=[len(X_test)-batch_size, len(X_test)]

                            input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                                prepare_sample_batch_feature(X_test, y_t_test, bs)
                            out, loss_v = session.run([moutput, loss], feed_dict={ncomputer.input_data: input_data,
                                                                             ncomputer.target_output: target_output,
                                                                             ncomputer.sequence_length: seq_len,
                                                                             ncomputer.decoder_point: decoder_point,
                                                                             ncomputer.mask:masks
                                                                            })

                            losses.append(loss_v)
                            out = np.reshape(np.asarray(out), [-1, seq_len])
                            rout_list = np.reshape(np.asarray(rout_list), [-1, y_t_train.shape[1]])
                            bout_list = []

                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            tescores.append(batch_mae(rout_list[:rs], bout_list[:rs], 0.995))


                        tloss = np.mean(losses)
                        tscore = np.mean(tescores)
                        print('tr mae {} vs te mae {}'.format(np.mean(trscores), np.mean(tescores)))

                        summary.value.add(tag='test_mae', simple_value=np.mean(tescores))
                        summary.value.add(tag='test_loss', simple_value=tloss)
                        #summary.value.add(tag='test_recall', simple_value=np.mean(tescores2))
                        #summary.value.add(tag='test_precision', simple_value=np.mean(tescores3))
                        train_writer.add_summary(summary, i)
                        train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if minloss>tscore:
                        minloss=tscore
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())


                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... ")

def exact_help_desk_test_time():
    chars = pickle.load(open('./data/help_desk/desk_tmp/chars.pkl', 'rb'), encoding='latin1')
    char_indices = pickle.load(open('./data/help_desk/desk_tmp/char_indices.pkl', 'rb'), encoding='latin1')
    target_indices_char = pickle.load(open('./data/help_desk/desk_tmp/target_indices_char.pkl', 'rb'), encoding='latin1')
    target_char_indices = pickle.load(open('./data/help_desk/desk_tmp/target_char_indices.pkl', 'rb'), encoding='latin1')
    target_chars = pickle.load(open('./data/help_desk/desk_tmp/target_chars.pkl', 'rb'), encoding='latin1')
    maxlen = pickle.load(open('./data/help_desk/desk_tmp/maxlen.pkl', 'rb'), encoding='latin1')
    divisor = pickle.load(open('./data/help_desk/desk_tmp/divisor.pkl', 'rb'), encoding='latin1')
    divisor2 = pickle.load(open('./data/help_desk/desk_tmp/divisor2.pkl', 'rb'), encoding='latin1')
    lines = pickle.load(open('./data/help_desk/desk_tmp/lines.pkl', 'rb'), encoding='latin1')
    lines_t = pickle.load(open('./data/help_desk/desk_tmp/lines_t.pkl', 'rb'), encoding='latin1')
    lines_t2 = pickle.load(open('./data/help_desk/desk_tmp/lines_t2.pkl', 'rb'), encoding='latin1')
    lines_t3 = pickle.load(open('./data/help_desk/desk_tmp/lines_t3.pkl', 'rb'), encoding='bytes')

    # lines_t4 = fold1_t4 + fold2_t4

    # set parameters
    predict_size = 1


    # define helper functions
    def encode(sentence, times, times3, maxlen=maxlen):
        num_features = len(chars) + 5
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence)
        times2 = np.cumsum(times)
        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight
            multiset_abstraction = Counter(sentence[:t + 1])
            for c in chars:
                if c == char:
                    X[0, t + leftpad, char_indices[c]] = 1
            X[0, t + leftpad, len(chars)] = t + 1
            X[0, t + leftpad, len(chars) + 1] = times[t] / divisor
            X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2
            X[0, t + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
            X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7
        return X

    def getSymbol(predictions):
        maxPrediction = 0
        symbol = ''
        i = 0;
        print(predictions)
        for prediction in predictions:
            if (prediction >= maxPrediction):
                maxPrediction = prediction
                symbol = target_indices_char[i]
            i += 1
        return symbol

    one_ahead_gt = []
    one_ahead_pred = []

    two_ahead_gt = []
    two_ahead_pred = []

    three_ahead_gt = []
    three_ahead_pred = []

    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_help_desk_time')
    batch_size = 1

    X, y_a, y_t = load_np_data('./data/help_desk/')

    if len(y_a.shape) < 3:
        y_a = np.reshape(y_a, [y_a.shape[0], 1, y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])

    num_sample = len(X)
    X_train = X[:int(0.8 * num_sample)]
    X_test = X[int(0.8 * num_sample):]

    y_a_train = y_a[:int(0.8 * num_sample)]
    y_a_test = y_a[int(0.8 * num_sample):]

    y_t_train = y_t[:int(0.8 * num_sample)]
    y_t_test = y_t[int(0.8 * num_sample):]

    dim_in = X_train.shape[2]
    dim_out = 1

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    input_size = dim_in
    output_size = 1
    sequence_max_length = 100
    test_data = []
    test_label = []
    words_count = 20
    word_size = 20
    read_heads = 1

    iterations = 5000
    start_step = 0

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:
            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=False,
                attend_dim=0
            )

            moutput, loss, apply_gradients = ncomputer.build_mae_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())
            llprint("Done!\n")

            print('-----')

            eventlog = "helpdesk.csv"
            # make predictions
            with open('./data/help_desk/next_time_%s' % eventlog, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(
                    ["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times",
                     "Predicted times", "RMSE", "MAE", "Median AE"])
                for prefix_size in range(2, maxlen):
                    print(prefix_size)
                    for line, times, times3 in zip(lines, lines_t, lines_t3):
                        times.append(0)
                        cropped_line = ''.join(line[:prefix_size])
                        cropped_times = times[:prefix_size]
                        cropped_times3 = times3[:prefix_size]
                        if '!' in cropped_line:
                            continue  # make no prediction for this case, since this case has ended already
                        ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                        ground_truth_t = times[prefix_size:prefix_size + predict_size]
                        predicted = ''
                        predicted_t = []
                        for i in range(predict_size):
                            if len(ground_truth) <= i:
                                continue
                            enc = encode(cropped_line, cropped_times, cropped_times3)
                            test_data.append(enc)

                            input_data, target_output, seq_len, decoder_point, masks, rout_list = \
                                prepare_sample_batch_feature(enc, np.asarray([y_t_test[0]]), 1)
                            out  = session.run([moutput], feed_dict={ncomputer.input_data: input_data,
                                                                               ncomputer.sequence_length: seq_len,
                                                                               ncomputer.decoder_point: decoder_point,
                                                                               ncomputer.mask: masks
                                                                               })
                            out = np.reshape(np.asarray(out), [-1, seq_len])
                            bout = []
                            for io in range(decoder_point, out.shape[1]):
                                bout.append(out[0][io])
                            y_char = 0
                            # print(y_char)
                            y_t = bout[0]
                            prediction = target_indices_char[y_char]
                            cropped_line += prediction
                            if y_t < 0:
                                y_t = 0
                            cropped_times.append(y_t)
                            y_t = y_t * divisor
                            cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                            predicted_t.append(y_t)
                            if i == 0:
                                if len(ground_truth_t) > 0:
                                    one_ahead_pred.append(y_t)
                                    one_ahead_gt.append(ground_truth_t[0])
                            if i == 1:
                                if len(ground_truth_t) > 1:
                                    two_ahead_pred.append(y_t)
                                    two_ahead_gt.append(ground_truth_t[1])
                            if i == 2:
                                if len(ground_truth_t) > 2:
                                    three_ahead_pred.append(y_t)
                                    three_ahead_gt.append(ground_truth_t[2])
                            if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                                print('! predicted, end case')
                                break
                            predicted += prediction
                        output = []
                        if len(ground_truth) > 0:
                            output.append(prefix_size)
                            output.append((ground_truth).encode("utf-8"))
                            output.append((predicted).encode("utf-8"))
                            output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                            dls = 1 - (
                            damerau_levenshtein_distance((predicted), (ground_truth)) / max(len(predicted), len(ground_truth)))
                            if dls < 0:
                                dls = 0  # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                            output.append(dls)
                            output.append(1 - distance.jaccard(predicted, ground_truth))
                            output.append('; '.join(str(x) for x in ground_truth_t))
                            output.append('; '.join(str(x) for x in predicted_t))
                            if len(predicted_t) > len(
                                    ground_truth_t):  # if predicted more events than length of case, only use needed number of events for time evaluation
                                predicted_t = predicted_t[:len(ground_truth_t)]
                            if len(ground_truth_t) > len(
                                    predicted_t):  # if predicted less events than length of case, put 0 as placeholder prediction
                                predicted_t.extend(range(len(ground_truth_t) - len(predicted_t)))
                            if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                                output.append('')
                                output.append(metrics.mean_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                                output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                            else:
                                output.append('')
                                output.append('')
                                output.append('')
                            spamwriter.writerow(output)

def help_desk_task_mix():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_help_desk_mix')
    batch_size = 15

    X, y_a, y_t = load_np_data('./data/help_desk/')

    if len(y_a.shape)<3:
        y_a= np.reshape(y_a,[y_a.shape[0],1,y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])

    all_index = list(range(len(X)))

    # np.random.shuffle(all_index)

    train_index = all_index[:int(len(X) * 0.8)]
    test_index = all_index[int(len(X) * 0.8):]


    X_train = X[train_index]
    X_test = X[test_index]

    y_a_train = y_a[train_index]
    y_a_test = y_a[test_index]

    y_t_train = y_t[train_index]
    y_t_test = y_t[test_index]



    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2]+ y_t_train.shape[1]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))


    input_size = dim_in
    output_size = dim_out+2
    sequence_max_length = 100

    words_count = 10
    word_size = 20
    read_heads = 1



    iterations = 10000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim = 100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                attend_dim=0
            )

            output1, output2, loss, apply_gradients = ncomputer.build_mix_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())

            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            train_writer = tf.summary.FileWriter('./data/log_help_desk/mix/', session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                        prepare_sample_batch_feature_mix(X_train,y_a_train,y_t_train, bs=batch_size)




                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)
                    if i<=1000000:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })

                    last_100_losses.append(loss_value)

                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores1 = []
                        trscores2 = []
                        for ii in range(10):
                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 =\
                                prepare_sample_batch_feature_mix(X_train,y_a_train,y_t_train, bs=batch_size)


                            out1,out2 = session.run([output1,output2], feed_dict={ncomputer.input_data: input_vec,
                                                                                  ncomputer.target_output: output_vec,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.mask:masks})

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size-1])
                            out1 = np.argmax(out1, axis=-1)
                            bout_list1 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out1.shape[0]):
                                out_list1 = []
                                for io in range(decoder_point, out1.shape[1]):
                                    out_list1.append(out1[b][io])
                                bout_list1.append(out_list1)
                            # print(rout_list)
                            # print(bout_list)
                            # raise False
                            pre, rec = set_score_pre(np.asarray(rout_list1), np.asarray(bout_list1))
                            trscores1.append(pre)

                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            rout_list2 = np.reshape(np.asarray(rout_list2),[-1,y_t_train.shape[1]])
                            bout_list2=[]
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out2.shape[0]):
                                out_list2 = []
                                for io in range(decoder_point, out2.shape[1]):
                                    out_list2.append(out2[b][io])
                                bout_list2.append(out_list2)
                            # print(rout_list)
                            # print(bout_list)
                            # raise False
                            trscores2.append(batch_mae(rout_list2, bout_list2,0.95))



                        print('-----')

                        tescores1 = []
                        tescores2 = []

                        losses = []
                        ntb=len(X_test)//batch_size+1
                        for ii in range(ntb):
                            if ii*batch_size==len(X_test):
                                break
                            bs=[ii*batch_size, min((ii+1)*batch_size,len(X_test))]
                            rs = bs[1] - bs[0]
                            if bs[1]>=len(X_test):
                                bs=[len(X_test)-batch_size, len(X_test)]

                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                                prepare_sample_batch_feature_mix(X_test, y_a_test, y_t_test, bs=bs)

                            out1, out2, loss_v = session.run([output1, output2, loss], feed_dict={ncomputer.input_data: input_vec,
                                                                                    ncomputer.decoder_point: decoder_point,
                                                                                    ncomputer.target_output: output_vec,
                                                                                    ncomputer.sequence_length: seq_len,
                                                                                    ncomputer.mask: masks})

                            losses.append(loss_v)

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size - 1])
                            out1 = np.argmax(out1, axis=-1)
                            bout_list1 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out1.shape[0]):
                                out_list1 = []
                                for io in range(decoder_point, out1.shape[1]):
                                    out_list1.append(out1[b][io])
                                bout_list1.append(out_list1)
                            pre, rec = set_score_pre(np.asarray(rout_list1[:rs]), np.asarray(bout_list1[:rs]))
                            tescores1.append(pre)


                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            rout_list2 = np.reshape(np.asarray(rout_list2), [-1, y_t_train.shape[1]])
                            bout_list2 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out2.shape[0]):
                                out_list2 = []
                                for io in range(decoder_point, out2.shape[1]):
                                    out_list2.append(out2[b][io])
                                bout_list2.append(out_list2)

                            tescores2.append(batch_mae(rout_list2[:rs], bout_list2[:rs], 0.995))


                        tloss = np.mean(losses)
                        print('test lost {}'.format(tloss))
                        print('tr pre {} vs te pre {}'.format(np.mean(trscores1), np.mean(tescores1)))
                        print('tr mae {} vs te mae {}'.format(np.mean(trscores2), np.mean(tescores2)))
                        summary.value.add(tag='test_pre', simple_value=np.mean(tescores1))
                        summary.value.add(tag='test_mae', simple_value=np.mean(tescores2))
                        summary.value.add(tag='test_loss', simple_value=tloss)

                        train_writer.add_summary(summary, i)
                        train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if minloss>tloss:
                        minloss=tloss
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())


                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... ")

def exact_help_desk_test_mix():
    chars = pickle.load(open('./data/help_desk/desk_tmp/chars.pkl', 'rb'), encoding='latin1')
    char_indices = pickle.load(open('./data/help_desk/desk_tmp/char_indices.pkl', 'rb'), encoding='latin1')
    target_indices_char = pickle.load(open('./data/help_desk/desk_tmp/target_indices_char.pkl', 'rb'), encoding='latin1')
    target_char_indices = pickle.load(open('./data/help_desk/desk_tmp/target_char_indices.pkl', 'rb'), encoding='latin1')
    target_chars = pickle.load(open('./data/help_desk/desk_tmp/target_chars.pkl', 'rb'), encoding='latin1')
    maxlen = pickle.load(open('./data/help_desk/desk_tmp/maxlen.pkl', 'rb'), encoding='latin1')
    divisor = pickle.load(open('./data/help_desk/desk_tmp/divisor.pkl', 'rb'), encoding='latin1')
    divisor2 = pickle.load(open('./data/help_desk/desk_tmp/divisor2.pkl', 'rb'), encoding='latin1')
    lines = pickle.load(open('./data/help_desk/desk_tmp/lines.pkl', 'rb'), encoding='latin1')
    lines_t = pickle.load(open('./data/help_desk/desk_tmp/lines_t.pkl', 'rb'), encoding='latin1')
    lines_t2 = pickle.load(open('./data/help_desk/desk_tmp/lines_t2.pkl', 'rb'), encoding='latin1')
    lines_t3 = pickle.load(open('./data/help_desk/desk_tmp/lines_t3.pkl', 'rb'), encoding='bytes')

    # lines_t4 = fold1_t4 + fold2_t4

    # set parameters
    predict_size = 1


    # define helper functions
    def encode(sentence, times, times3, maxlen=maxlen):
        num_features = len(chars) + 5
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence)
        times2 = np.cumsum(times)
        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight
            multiset_abstraction = Counter(sentence[:t + 1])
            for c in chars:
                if c == char:
                    X[0, t + leftpad, char_indices[c]] = 1
            X[0, t + leftpad, len(chars)] = t + 1
            X[0, t + leftpad, len(chars) + 1] = times[t] / divisor
            X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2
            X[0, t + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
            X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7
        return X

    def getSymbol(predictions):
        maxPrediction = 0
        symbol = ''
        i = 0;
        print(predictions)
        for prediction in predictions:
            if (prediction >= maxPrediction):
                maxPrediction = prediction
                symbol = target_indices_char[i]
            i += 1
        return symbol

    one_ahead_gt = []
    one_ahead_pred = []

    two_ahead_gt = []
    two_ahead_pred = []

    three_ahead_gt = []
    three_ahead_pred = []

    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_help_desk_mix')
    batch_size = 1

    X, y_a, y_t = load_np_data('./data/help_desk/')

    if len(y_a.shape) < 3:
        y_a = np.reshape(y_a, [y_a.shape[0], 1, y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])

    num_sample = len(X)
    X_train = X[:int(0.8 * num_sample)]
    X_test = X[int(0.8 * num_sample):]

    y_a_train = y_a[:int(0.8 * num_sample)]
    y_a_test = y_a[int(0.8 * num_sample):]

    y_t_train = y_t[:int(0.8 * num_sample)]
    y_t_test = y_t[int(0.8 * num_sample):]

    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2] + y_t_train.shape[1]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    input_size = dim_in
    output_size = dim_out + 2
    sequence_max_length = 100

    words_count = 5
    word_size = 20
    read_heads = 1
    test_data=[]
    iterations = 10000
    start_step = 0

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                attend_dim=0
            )

            output1, output2, loss, apply_gradients = ncomputer.build_mix_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())
            llprint("Done!\n")

            print('-----')

            eventlog = "helpdesk.csv"
            # make predictions
            with open('./data/help_desk/next_activity_and_time_%s' % eventlog, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(
                    ["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times",
                     "Predicted times", "RMSE", "MAE", "Median AE"])
                for prefix_size in range(2, maxlen):
                    print(prefix_size)
                    for line, times, times3 in zip(lines, lines_t, lines_t3):
                        times.append(0)
                        cropped_line = ''.join(line[:prefix_size])
                        cropped_times = times[:prefix_size]
                        cropped_times3 = times3[:prefix_size]
                        if '!' in cropped_line:
                            continue  # make no prediction for this case, since this case has ended already
                        ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                        ground_truth_t = times[prefix_size:prefix_size + predict_size]
                        predicted = ''
                        predicted_t = []
                        for i in range(predict_size):
                            if len(ground_truth) <= i:
                                continue
                            enc = encode(cropped_line, cropped_times, cropped_times3)
                            test_data.append(enc)

                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                                prepare_sample_batch_feature_mix(enc,np.asarray([y_a_test[0]]), np.asarray([y_t_test[0]]), 1)

                            out1, out2, loss_v = session.run([output1, output2, loss],
                                                             feed_dict={ncomputer.input_data: input_vec,
                                                                        ncomputer.decoder_point: decoder_point,
                                                                        ncomputer.target_output: output_vec,
                                                                        ncomputer.sequence_length: seq_len,
                                                                        ncomputer.mask: masks})

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size-1])
                            out1 = np.argmax(out1, axis=-1)
                            bout1 = []
                            for io in range(decoder_point, out1.shape[1]):
                                bout1.append(max(out1[0][io] - 2, 1))
                            y_char = bout1[0]

                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            bout2 = []
                            for io in range(decoder_point, out2.shape[1]):
                                bout2.append(out2[0][io])

                            # print(y_char)
                            y_t = bout2[0]
                            prediction = target_indices_char[y_char]
                            cropped_line += prediction
                            if y_t < 0:
                                y_t = 0
                            cropped_times.append(y_t)
                            y_t = y_t * divisor
                            cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                            predicted_t.append(y_t)
                            if i == 0:
                                if len(ground_truth_t) > 0:
                                    one_ahead_pred.append(y_t)
                                    one_ahead_gt.append(ground_truth_t[0])
                            if i == 1:
                                if len(ground_truth_t) > 1:
                                    two_ahead_pred.append(y_t)
                                    two_ahead_gt.append(ground_truth_t[1])
                            if i == 2:
                                if len(ground_truth_t) > 2:
                                    three_ahead_pred.append(y_t)
                                    three_ahead_gt.append(ground_truth_t[2])
                            if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                                print('! predicted, end case')
                                break
                            predicted += prediction
                        output = []
                        if len(ground_truth) > 0:
                            output.append(prefix_size)
                            output.append((ground_truth).encode("utf-8"))
                            output.append((predicted).encode("utf-8"))
                            output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                            dls = 1 - (
                            damerau_levenshtein_distance((predicted), (ground_truth)) / max(len(predicted), len(ground_truth)))
                            if dls < 0:
                                dls = 0  # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                            output.append(dls)
                            output.append(1 - distance.jaccard(predicted, ground_truth))
                            output.append('; '.join(str(x) for x in ground_truth_t))
                            output.append('; '.join(str(x) for x in predicted_t))
                            if len(predicted_t) > len(
                                    ground_truth_t):  # if predicted more events than length of case, only use needed number of events for time evaluation
                                predicted_t = predicted_t[:len(ground_truth_t)]
                            if len(ground_truth_t) > len(
                                    predicted_t):  # if predicted less events than length of case, put 0 as placeholder prediction
                                predicted_t.extend(range(len(ground_truth_t) - len(predicted_t)))
                            if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                                output.append('')
                                output.append(metrics.mean_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                                output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                            else:
                                output.append('')
                                output.append('')
                                output.append('')
                            spamwriter.writerow(output)

def exact_help_desk_test_mix2():
    chars = pickle.load(open('./data/help_desk/desk_tmp2/chars.pkl', 'rb'), encoding='latin1')
    char_indices = pickle.load(open('./data/help_desk/desk_tmp2/char_indices.pkl', 'rb'), encoding='latin1')
    target_indices_char = pickle.load(open('./data/help_desk/desk_tmp2/target_indices_char.pkl', 'rb'), encoding='latin1')
    target_char_indices = pickle.load(open('./data/help_desk/desk_tmp2/target_char_indices.pkl', 'rb'), encoding='latin1')
    target_chars = pickle.load(open('./data/help_desk/desk_tmp2/target_chars.pkl', 'rb'), encoding='latin1')
    maxlen = pickle.load(open('./data/help_desk/desk_tmp2/maxlen.pkl', 'rb'), encoding='latin1')
    divisor = pickle.load(open('./data/help_desk/desk_tmp2/divisor.pkl', 'rb'), encoding='latin1')
    divisor2 = pickle.load(open('./data/help_desk/desk_tmp2/divisor2.pkl', 'rb'), encoding='latin1')
    divisor3 = pickle.load(open('./data/help_desk/desk_tmp2/divisor3.pkl', 'rb'), encoding='latin1')
    lines = pickle.load(open('./data/help_desk/desk_tmp2/lines.pkl', 'rb'), encoding='latin1')
    lines_t = pickle.load(open('./data/help_desk/desk_tmp2/lines_t.pkl', 'rb'), encoding='latin1')
    lines_t2 = pickle.load(open('./data/help_desk/desk_tmp2/lines_t2.pkl', 'rb'), encoding='latin1')
    lines_t3 = pickle.load(open('./data/help_desk/desk_tmp2/lines_t3.pkl', 'rb'), encoding='bytes')

    # lines_t4 = fold1_t4 + fold2_t4

    # set parameters
    predict_size = maxlen
    print('xxxx')
    print(maxlen)
    print('xxxx')
    # define helper functions
    def encode(sentence, times, times3, maxlen=maxlen):
        num_features = len(chars) + 5
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence)
        times2 = np.cumsum(times)
        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight
            multiset_abstraction = Counter(sentence[:t + 1])
            for c in chars:
                if c == char:
                    X[0, t + leftpad, char_indices[c]] = 1
            X[0, t + leftpad, len(chars)] = t + 1
            X[0, t + leftpad, len(chars) + 1] = times[t] / divisor
            X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2
            X[0, t + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
            X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7
        return X

    def getSymbol(predictions):
        maxPrediction = 0
        symbol = ''
        i = 0;
        for prediction in predictions:
            if (prediction >= maxPrediction):
                maxPrediction = prediction
                symbol = target_indices_char[i]
            i += 1
        return symbol

    one_ahead_gt = []
    one_ahead_pred = []

    two_ahead_gt = []
    two_ahead_pred = []

    three_ahead_gt = []
    three_ahead_pred = []

    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_help_desk_mix')
    batch_size = 1

    X, y_a, y_t = load_np_data('./data/help_desk/')

    if len(y_a.shape) < 3:
        y_a = np.reshape(y_a, [y_a.shape[0], 1, y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])

    num_sample = len(X)
    X_train = X[:int(0.8 * num_sample)]
    X_test = X[int(0.8 * num_sample):]

    y_a_train = y_a[:int(0.8 * num_sample)]
    y_a_test = y_a[int(0.8 * num_sample):]

    y_t_train = y_t[:int(0.8 * num_sample)]
    y_t_test = y_t[int(0.8 * num_sample):]

    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2] + y_t_train.shape[1]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    input_size = dim_in
    output_size = dim_out + 2
    sequence_max_length = 100

    words_count = 5
    word_size = 20
    read_heads = 1
    test_data=[]
    iterations = 10000
    start_step = 0

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                attend_dim=0
            )

            output1, output2, loss, apply_gradients = ncomputer.build_mix_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config())
            llprint("Done!\n")

            print('-----')

            eventlog = "helpdesk.csv"
            # make predictions
            with open('./data/help_desk/suffix_and_remaining_time_%s' % eventlog, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard",
                                     "Ground truth times", "Predicted times", "RMSE", "MAE", "Median AE"])
                for prefix_size in range(2, maxlen):
                    print(prefix_size)
                    for line, times, times2, times3 in zip(lines, lines_t, lines_t2, lines_t3):
                        times.append(0)
                        cropped_line = ''.join(line[:prefix_size])
                        cropped_times = times[:prefix_size]
                        cropped_times3 = times3[:prefix_size]
                        if len(times2) < prefix_size:
                            continue  # make no prediction for this case, since this case has ended already
                        ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                        ground_truth_t = times2[prefix_size - 1]
                        case_end_time = times2[len(times2) - 1]
                        ground_truth_t = case_end_time - ground_truth_t
                        predicted = ''
                        total_predicted_time = 0
                        for i in range(predict_size):
                            enc = encode(cropped_line, cropped_times, cropped_times3)
                            test_data.append(enc)

                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                                prepare_sample_batch_feature_mix(enc,np.asarray([y_a_test[0]]), np.asarray([y_t_test[0]]), 1)

                            out1, out2, loss_v = session.run([output1, output2, loss],
                                                             feed_dict={ncomputer.input_data: input_vec,
                                                                        ncomputer.decoder_point: decoder_point,
                                                                        ncomputer.target_output: output_vec,
                                                                        ncomputer.sequence_length: seq_len,
                                                                        ncomputer.mask: masks})

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size-1])
                            out1 = np.argmax(out1, axis=-1)
                            bout1 = []
                            for io in range(decoder_point, out1.shape[1]):
                                bout1.append(max(out1[0][io] - 2, 0))
                            y_char = bout1[0]

                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            bout2 = []
                            for io in range(decoder_point, out2.shape[1]):
                                bout2.append(out2[0][io])

                            # print(y_char)
                            y_t = bout2[0]
                            prediction = target_indices_char[y_char]  # undo one-hot encoding

                            cropped_line += prediction
                            if y_t < 0:
                                y_t = 0
                            cropped_times.append(y_t)
                            if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                                one_ahead_pred.append(total_predicted_time)
                                one_ahead_gt.append(ground_truth_t)
                                print('! predicted, end case')
                                break
                            y_t = y_t * divisor3
                            cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                            total_predicted_time = total_predicted_time + y_t
                            predicted += prediction
                        output = []
                        if len(ground_truth) > 0:
                            output.append(prefix_size)
                            output.append((ground_truth).encode("utf-8"))
                            output.append((predicted).encode("utf-8"))
                            output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                            dls = 1 - (damerau_levenshtein_distance((predicted), (ground_truth)) / max(len(predicted),
                                                                                                       len(
                                                                                                           ground_truth)))
                            if dls < 0:
                                dls = 0  # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                            output.append(dls)
                            output.append(1 - distance.jaccard(predicted, ground_truth))
                            output.append(ground_truth_t)
                            output.append(total_predicted_time)
                            output.append('')
                            output.append(metrics.mean_absolute_error([ground_truth_t], [total_predicted_time]))
                            output.append(metrics.median_absolute_error([ground_truth_t], [total_predicted_time]))
                            spamwriter.writerow(output)


def get_nb_params_shape(shape):

    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

def count_number_trainable_params():
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params



def bpi_task_mix():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_bpi_nor_mix')


    X, y_a, y_t = load_np_data('./data/bpi/')

    if len(y_a.shape)<3:
        y_a= np.reshape(y_a,[y_a.shape[0],1,y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])
    batch_size = X.shape[1]
    print('bs {}'.format(batch_size))
    all_index = list(range(len(X)))

    # np.random.shuffle(all_index)

    train_index = all_index[:int(len(X) * 0.8)]
    test_index = all_index[int(len(X) * 0.8):]

    # print(X.shape[1])
    # raise False

    X_train = X[train_index]
    X_test = X[test_index]

    y_a_train = y_a[train_index]
    y_a_test = y_a[test_index]

    y_t_train = y_t[train_index]
    y_t_test = y_t[test_index]



    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2]+ y_t_train.shape[1]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    raise False

    input_size = dim_in
    output_size = dim_out+2
    sequence_max_length = 100

    words_count = 5
    word_size = 20
    read_heads = 1



    iterations = 20000
    start_step = 0


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim = 100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                attend_dim=0
            )

            output1, output2, loss, apply_gradients = ncomputer.build_mix_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            print(count_number_trainable_params())
            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            train_writer = tf.summary.FileWriter('./data/log_bpi_nor/', session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                        prepare_sample_batch_feature_mix(X_train,y_a_train,y_t_train, bs=batch_size)




                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)
                    if i<=1000000:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks
                        })

                    last_100_losses.append(loss_value)

                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores1 = []
                        trscores2 = []
                        for ii in range(10):
                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 =\
                                prepare_sample_batch_feature_mix(X_train,y_a_train,y_t_train, bs=batch_size)


                            out1,out2 = session.run([output1,output2], feed_dict={ncomputer.input_data: input_vec,
                                                                                  ncomputer.target_output: output_vec,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.mask:masks})

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size-1])
                            out1 = np.argmax(out1, axis=-1)
                            bout_list1 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out1.shape[0]):
                                out_list1 = []
                                for io in range(decoder_point, out1.shape[1]):
                                    out_list1.append(out1[b][io])
                                bout_list1.append(out_list1)
                            # print(rout_list)
                            # print(bout_list)
                            # raise False
                            pre, rec = set_score_pre(np.asarray(rout_list1), np.asarray(bout_list1))
                            trscores1.append(pre)

                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            rout_list2 = np.reshape(np.asarray(rout_list2),[-1,y_t_train.shape[1]])
                            bout_list2=[]
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out2.shape[0]):
                                out_list2 = []
                                for io in range(decoder_point, out2.shape[1]):
                                    out_list2.append(out2[b][io])
                                bout_list2.append(out_list2)
                            # print(rout_list)
                            # print(bout_list)
                            # raise False
                            trscores2.append(batch_mae(rout_list2, bout_list2,0.95))



                        print('-----')

                        tescores1 = []
                        tescores2 = []

                        losses = []
                        ntb=len(X_test)//batch_size+1
                        for ii in range(ntb):
                            if ii*batch_size==len(X_test):
                                break
                            bs=[ii*batch_size, min((ii+1)*batch_size,len(X_test))]
                            rs = bs[1] - bs[0]
                            if bs[1]>=len(X_test):
                                bs=[len(X_test)-batch_size, len(X_test)]

                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                                prepare_sample_batch_feature_mix(X_test, y_a_test, y_t_test, bs=bs)

                            out1, out2, loss_v = session.run([output1, output2, loss], feed_dict={ncomputer.input_data: input_vec,
                                                                                    ncomputer.decoder_point: decoder_point,
                                                                                    ncomputer.target_output: output_vec,
                                                                                    ncomputer.sequence_length: seq_len,
                                                                                    ncomputer.mask: masks})

                            losses.append(loss_v)

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size - 1])
                            out1 = np.argmax(out1, axis=-1)
                            bout_list1 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out1.shape[0]):
                                out_list1 = []
                                for io in range(decoder_point, out1.shape[1]):
                                    out_list1.append(out1[b][io])
                                bout_list1.append(out_list1)
                            pre, rec = set_score_pre(np.asarray(rout_list1[:rs]), np.asarray(bout_list1[:rs]))
                            tescores1.append(pre)


                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            rout_list2 = np.reshape(np.asarray(rout_list2), [-1, y_t_train.shape[1]])
                            bout_list2 = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))
                            for b in range(out2.shape[0]):
                                out_list2 = []
                                for io in range(decoder_point, out2.shape[1]):
                                    out_list2.append(out2[b][io])
                                bout_list2.append(out_list2)

                            tescores2.append(batch_mae(rout_list2[:rs], bout_list2[:rs], 0.995))


                        tloss = np.mean(losses)
                        print('test lost {} vs min loss {}'.format(tloss,minloss))
                        print('tr pre {} vs te pre {}'.format(np.mean(trscores1), np.mean(tescores1)))
                        print('tr mae {} vs te mae {}'.format(np.mean(trscores2), np.mean(tescores2)))
                        summary.value.add(tag='test_pre', simple_value=np.mean(tescores1))
                        summary.value.add(tag='test_mae', simple_value=np.mean(tescores2))
                        summary.value.add(tag='test_loss', simple_value=tloss)

                        train_writer.add_summary(summary, i)
                        train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if minloss>tloss:
                        minloss=tloss
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())
                        print("\nSaving Checkpoint ...\n"),



                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... ")

def exact_bpi_test_mix():
    chars = pickle.load(open('./data/bpi/bpi_tmp/chars.pkl', 'rb'), encoding='latin1')
    char_indices = pickle.load(open('./data/bpi/bpi_tmp/char_indices.pkl', 'rb'), encoding='latin1')
    target_indices_char = pickle.load(open('./data/bpi/bpi_tmp/target_indices_char.pkl', 'rb'), encoding='latin1')
    target_char_indices = pickle.load(open('./data/bpi/bpi_tmp/target_char_indices.pkl', 'rb'), encoding='latin1')
    target_chars = pickle.load(open('./data/bpi/bpi_tmp/target_chars.pkl', 'rb'), encoding='latin1')
    maxlen = pickle.load(open('./data/bpi/bpi_tmp/maxlen.pkl', 'rb'), encoding='latin1')
    divisor = pickle.load(open('./data/bpi/bpi_tmp/divisor.pkl', 'rb'), encoding='latin1')
    divisor2 = pickle.load(open('./data/bpi/bpi_tmp/divisor2.pkl', 'rb'), encoding='latin1')
    lines = pickle.load(open('./data/bpi/bpi_tmp/lines.pkl', 'rb'), encoding='latin1')
    lines_t = pickle.load(open('./data/bpi/bpi_tmp/lines_t.pkl', 'rb'), encoding='latin1')
    lines_t2 = pickle.load(open('./data/bpi/bpi_tmp/lines_t2.pkl', 'rb'), encoding='latin1')
    lines_t3 = pickle.load(open('./data/bpi/bpi_tmp/lines_t3.pkl', 'rb'), encoding='bytes')

    # lines_t4 = fold1_t4 + fold2_t4

    # set parameters
    predict_size = 1


    # define helper functions
    def encode(sentence, times, times3, maxlen=maxlen):
        num_features = len(chars) + 5
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence)
        times2 = np.cumsum(times)
        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight
            multiset_abstraction = Counter(sentence[:t + 1])
            for c in chars:
                if c == char:
                    X[0, t + leftpad, char_indices[c]] = 1
            X[0, t + leftpad, len(chars)] = t + 1
            X[0, t + leftpad, len(chars) + 1] = times[t] / divisor
            X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2
            X[0, t + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
            X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7
        return X

    def getSymbol(predictions):
        maxPrediction = 0
        symbol = ''
        i = 0;
        print(predictions)
        for prediction in predictions:
            if (prediction >= maxPrediction):
                maxPrediction = prediction
                symbol = target_indices_char[i]
            i += 1
        return symbol

    one_ahead_gt = []
    one_ahead_pred = []

    two_ahead_gt = []
    two_ahead_pred = []

    three_ahead_gt = []
    three_ahead_pred = []

    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_bpi_mix')
    batch_size = 1

    X, y_a, y_t = load_np_data('./data/bpi/')

    if len(y_a.shape) < 3:
        y_a = np.reshape(y_a, [y_a.shape[0], 1, y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])

    num_sample = len(X)
    X_train = X[:int(0.8 * num_sample)]
    X_test = X[int(0.8 * num_sample):]

    y_a_train = y_a[:int(0.8 * num_sample)]
    y_a_test = y_a[int(0.8 * num_sample):]

    y_t_train = y_t[:int(0.8 * num_sample)]
    y_t_test = y_t[int(0.8 * num_sample):]

    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2] + y_t_train.shape[1]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    input_size = dim_in
    output_size = dim_out + 2
    sequence_max_length = 100

    words_count = 20
    word_size = 20
    read_heads = 1
    test_data=[]
    iterations = 10000
    start_step = 0

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                attend_dim=0
            )

            output1, output2, loss, apply_gradients = ncomputer.build_mix_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config()+'_ba')
            llprint("Done!\n")

            print('-----')

            eventlog = "helpdesk.csv"
            # make predictions
            with open('./data/bpi/next_activity_and_time_%s' % eventlog, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(
                    ["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times",
                     "Predicted times", "RMSE", "MAE", "Median AE"])
                for prefix_size in range(2, maxlen):
                    print(prefix_size)
                    for line, times, times3 in zip(lines, lines_t, lines_t3):
                        times.append(0)
                        cropped_line = ''.join(line[:prefix_size])
                        cropped_times = times[:prefix_size]
                        cropped_times3 = times3[:prefix_size]
                        if '!' in cropped_line:
                            continue  # make no prediction for this case, since this case has ended already
                        ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                        ground_truth_t = times[prefix_size:prefix_size + predict_size]
                        predicted = ''
                        predicted_t = []
                        for i in range(predict_size):
                            if len(ground_truth) <= i:
                                continue
                            enc = encode(cropped_line, cropped_times, cropped_times3)
                            test_data.append(enc)

                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                                prepare_sample_batch_feature_mix(enc,np.asarray([y_a_test[0]]), np.asarray([y_t_test[0]]), 1)

                            out1, out2, loss_v = session.run([output1, output2, loss],
                                                             feed_dict={ncomputer.input_data: input_vec,
                                                                        ncomputer.decoder_point: decoder_point,
                                                                        ncomputer.target_output: output_vec,
                                                                        ncomputer.sequence_length: seq_len,
                                                                        ncomputer.mask: masks})

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size-1])
                            out1 = np.argmax(out1, axis=-1)
                            bout1 = []
                            for io in range(decoder_point, out1.shape[1]):
                                bout1.append(max(out1[0][io] - 2, 1))
                            y_char = bout1[0]

                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            bout2 = []
                            for io in range(decoder_point, out2.shape[1]):
                                bout2.append(out2[0][io])

                            # print(y_char)
                            y_t = bout2[0]
                            prediction = target_indices_char[y_char]
                            cropped_line += prediction
                            if y_t < 0:
                                y_t = 0
                            cropped_times.append(y_t)
                            y_t = y_t * divisor
                            cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                            predicted_t.append(y_t)
                            if i == 0:
                                if len(ground_truth_t) > 0:
                                    one_ahead_pred.append(y_t)
                                    one_ahead_gt.append(ground_truth_t[0])
                            if i == 1:
                                if len(ground_truth_t) > 1:
                                    two_ahead_pred.append(y_t)
                                    two_ahead_gt.append(ground_truth_t[1])
                            if i == 2:
                                if len(ground_truth_t) > 2:
                                    three_ahead_pred.append(y_t)
                                    three_ahead_gt.append(ground_truth_t[2])
                            if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                                print('! predicted, end case')
                                break
                            predicted += prediction
                        output = []
                        if len(ground_truth) > 0:
                            output.append(prefix_size)
                            output.append((ground_truth).encode("utf-8"))
                            output.append((predicted).encode("utf-8"))
                            output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                            dls = 1 - (
                            damerau_levenshtein_distance((predicted), (ground_truth)) / max(len(predicted), len(ground_truth)))
                            if dls < 0:
                                dls = 0  # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                            output.append(dls)
                            output.append(1 - distance.jaccard(predicted, ground_truth))
                            output.append('; '.join(str(x) for x in ground_truth_t))
                            output.append('; '.join(str(x) for x in predicted_t))
                            if len(predicted_t) > len(
                                    ground_truth_t):  # if predicted more events than length of case, only use needed number of events for time evaluation
                                predicted_t = predicted_t[:len(ground_truth_t)]
                            if len(ground_truth_t) > len(
                                    predicted_t):  # if predicted less events than length of case, put 0 as placeholder prediction
                                predicted_t.extend(range(len(ground_truth_t) - len(predicted_t)))
                            if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                                output.append('')
                                output.append(metrics.mean_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                                output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                            else:
                                output.append('')
                                output.append('')
                                output.append('')
                            spamwriter.writerow(output)

def exact_bpi_test_mix2():
    chars = pickle.load(open('./data/bpi_nor/bpi_tmp2/chars.pkl', 'rb'), encoding='latin1')
    char_indices = pickle.load(open('./data/bpi_nor/bpi_tmp2/char_indices.pkl', 'rb'), encoding='latin1')
    target_indices_char = pickle.load(open('./data/bpi_nor/bpi_tmp2/target_indices_char.pkl', 'rb'), encoding='latin1')
    target_char_indices = pickle.load(open('./data/bpi_nor/bpi_tmp2/target_char_indices.pkl', 'rb'), encoding='latin1')
    target_chars = pickle.load(open('./data/bpi_nor/bpi_tmp2/target_chars.pkl', 'rb'), encoding='latin1')
    maxlen = pickle.load(open('./data/bpi_nor/bpi_tmp2/maxlen.pkl', 'rb'), encoding='latin1')
    divisor = pickle.load(open('./data/bpi_nor/bpi_tmp2/divisor.pkl', 'rb'), encoding='latin1')
    divisor2 = pickle.load(open('./data/bpi_nor/bpi_tmp2/divisor2.pkl', 'rb'), encoding='latin1')
    divisor3 = pickle.load(open('./data/bpi_nor/bpi_tmp2/divisor3.pkl', 'rb'), encoding='latin1')
    lines = pickle.load(open('./data/bpi_nor/bpi_tmp2/lines.pkl', 'rb'), encoding='latin1')
    lines_t = pickle.load(open('./data/bpi_nor/bpi_tmp2/lines_t.pkl', 'rb'), encoding='latin1')
    lines_t2 = pickle.load(open('./data/bpi_nor/bpi_tmp2/lines_t2.pkl', 'rb'), encoding='latin1')
    lines_t3 = pickle.load(open('./data/bpi_nor/bpi_tmp2/lines_t3.pkl', 'rb'), encoding='bytes')

    # lines_t4 = fold1_t4 + fold2_t4

    # set parameters
    predict_size = maxlen
    print('xxxx')
    print(maxlen)
    print('xxxx')
    # define helper functions
    def encode(sentence, times, times3, maxlen=maxlen):
        num_features = len(chars) + 5
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence)
        times2 = np.cumsum(times)
        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight
            multiset_abstraction = Counter(sentence[:t + 1])
            for c in chars:
                if c == char:
                    X[0, t + leftpad, char_indices[c]] = 1
            X[0, t + leftpad, len(chars)] = t + 1
            X[0, t + leftpad, len(chars) + 1] = times[t] / divisor
            X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2
            X[0, t + leftpad, len(chars) + 3] = timesincemidnight.seconds / 86400
            X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7
        return X

    def getSymbol(predictions):
        maxPrediction = 0
        symbol = ''
        i = 0;
        for prediction in predictions:
            if (prediction >= maxPrediction):
                maxPrediction = prediction
                symbol = target_indices_char[i]
            i += 1
        return symbol

    one_ahead_gt = []
    one_ahead_pred = []

    two_ahead_gt = []
    two_ahead_pred = []

    three_ahead_gt = []
    three_ahead_pred = []

    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_bpi_nor_mix')
    batch_size = 1

    X, y_a, y_t = load_np_data('./data/bpi_nor/')

    if len(y_a.shape) < 3:
        y_a = np.reshape(y_a, [y_a.shape[0], 1, y_a.shape[1]])

    if len(y_t.shape) <2:
        y_t = np.reshape(y_t, [y_t.shape[0], 1, 1])

    num_sample = len(X)
    X_train = X[:int(0.8 * num_sample)]
    X_test = X[int(0.8 * num_sample):]

    y_a_train = y_a[:int(0.8 * num_sample)]
    y_a_test = y_a[int(0.8 * num_sample):]

    y_t_train = y_t[:int(0.8 * num_sample)]
    y_t_test = y_t[int(0.8 * num_sample):]

    dim_in = X_train.shape[2]
    dim_out = y_a_train.shape[2] + y_t_train.shape[1]

    print('num train {}'.format(len(X_train)))
    print('num test {}'.format(len(X_test)))
    print('dim in  {}'.format(dim_in))
    print('dim out {}'.format(dim_out))

    input_size = dim_in
    output_size = dim_out + 2
    sequence_max_length = 100

    words_count = 5
    word_size = 20
    read_heads = 1
    test_data = []
    iterations = 10000
    start_step = 0

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=100,
                use_emb=False,
                use_mem=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True,
                attend_dim=0
            )

            output1, output2, loss, apply_gradients = ncomputer.build_mix_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            ncomputer.restore(session, ckpts_dir, ncomputer.print_config()+'_ba')
            llprint("Done!\n")

            print('-----')

            eventlog = "bpi.csv"
            # make predictions
            with open('./data/bpi_nor/suffix_and_remaining_time_%s' % eventlog, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard",
                                     "Ground truth times", "Predicted times", "RMSE", "MAE", "Median AE"])

                for prefix_size in range(2, maxlen):
                    print(prefix_size)
                    cc = 0
                    for line, times, times2, times3 in zip(lines, lines_t, lines_t2, lines_t3):
                        times.append(0)
                        cropped_line = ''.join(line[:prefix_size])
                        cropped_times = times[:prefix_size]
                        cropped_times3 = times3[:prefix_size]
                        if len(times2) < prefix_size:
                            continue  # make no prediction for this case, since this case has ended already
                        ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                        ground_truth_t = times2[prefix_size - 1]
                        case_end_time = times2[len(times2) - 1]
                        ground_truth_t = case_end_time - ground_truth_t
                        predicted = ''
                        total_predicted_time = 0
                        for i in range(predict_size):
                            enc = encode(cropped_line, cropped_times, cropped_times3)
                            test_data.append(enc)

                            input_vec, output_vec, seq_len, decoder_point, masks, rout_list1, rout_list2 = \
                                prepare_sample_batch_feature_mix(enc,np.asarray([y_a_test[0]]), np.asarray([y_t_test[0]]), 1)

                            out1, out2, loss_v = session.run([output1, output2, loss],
                                                             feed_dict={ncomputer.input_data: input_vec,
                                                                        ncomputer.decoder_point: decoder_point,
                                                                        ncomputer.target_output: output_vec,
                                                                        ncomputer.sequence_length: seq_len,
                                                                        ncomputer.mask: masks})

                            out1 = np.reshape(np.asarray(out1), [-1, seq_len, output_size-1])
                            out1 = np.argmax(out1, axis=-1)
                            bout1 = []
                            for io in range(decoder_point, out1.shape[1]):
                                bout1.append(max(out1[0][io] - 2, 0))
                            y_char = bout1[0]

                            out2 = np.reshape(np.asarray(out2), [-1, seq_len])
                            bout2 = []
                            for io in range(decoder_point, out2.shape[1]):
                                bout2.append(out2[0][io])

                            # print(y_char)
                            y_t = bout2[0]
                            prediction = target_indices_char[y_char]  # undo one-hot encoding

                            cropped_line += prediction
                            if y_t < 0:
                                y_t = 0
                            cropped_times.append(y_t)
                            if prediction == '!':  # end of case was just predicted, therefore, stop predicting further into the future
                                one_ahead_pred.append(total_predicted_time)
                                one_ahead_gt.append(ground_truth_t)
                                print('! predicted, end case')
                                break
                            y_t = y_t * divisor3
                            cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                            total_predicted_time = total_predicted_time + y_t
                            predicted += prediction
                        output = []
                        if len(ground_truth) > 0:
                            output.append(prefix_size)
                            output.append((ground_truth).encode("utf-8"))
                            output.append((predicted).encode("utf-8"))
                            output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                            dls = 1 - (damerau_levenshtein_distance((predicted), (ground_truth)) / max(len(predicted),
                                                                                                       len(
                                                                                                           ground_truth)))
                            if dls < 0:
                                dls = 0  # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                            output.append(dls)
                            output.append(1 - distance.jaccard(predicted, ground_truth))
                            output.append(ground_truth_t)
                            output.append(total_predicted_time)
                            output.append('')
                            output.append(metrics.mean_absolute_error([ground_truth_t], [total_predicted_time]))
                            output.append(metrics.median_absolute_error([ground_truth_t], [total_predicted_time]))
                            spamwriter.writerow(output)
                            if cc>10:
                                # break
                                pass
                            cc+=1

if __name__ == '__main__':
    #moodle_task()
    #moodle_test()
    # financial_log_task()
    # financial_log_test()
    # ibm_task()
    # ibm_test()
    # help_desk_task()
    # help_desk_test()
    # exact_help_desk_test()
    # help_desk_task_time()
    # exact_help_desk_test_time()
    # help_desk_task_mix()
    # exact_help_desk_test_mix()
    # exact_help_desk_test_mix2()
    bpi_task_mix()
    # exact_bpi_test_mix()
    # exact_bpi_test_mix2()