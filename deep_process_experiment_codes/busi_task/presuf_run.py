import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import time
import sys
import os
import nltk
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
        if np.random.rand()>0.99:
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

def write_predict(wfile, list_pred):
    with open(wfile,'w') as f:
        for p in list_pred:
            f.write('[')
            for n in p[:-1]:
                f.write(str(n-1)+' ')
            f.write(str(p[-1]-1))
            f.write(']')
            f.write('\n')

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

def moodle_train():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_moodle')

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
    sequence_max_length = 100 #redundant, not used, use dynamic length
    batch_size = 10
    words_count = 64
    word_size = 100
    read_heads = 1

    iterations = 150000
    max_train_iterations = 100000
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
            minscore=1000
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

                    if i<=max_train_iterations:
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
                        print('tr edit {} vs te edit {}'.format(np.mean(trscores), np.mean(tescores)))
                        #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
                        print('test loss {}'.format(tloss))
                        summary.value.add(tag='train_edit', simple_value=np.mean(trscores))
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

                    if minscore>tscore:
                        minscore=tscore
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, ncomputer.print_config())


                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... "),


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





def financial_log_train():
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


    iterations = 10000
    max_train_iterations = 100000
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
            )


            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()
            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minscore=1000
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
                    if i<=max_train_iterations:
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks,
                        })
                    else:
                        loss_value = session.run(loss, feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask:masks,
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
                        print('tr edit {} vs te edit {}'.format(np.mean(trscores), np.mean(tescores)))
                        #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
                        print('test loss {}'.format(tloss))
                        summary.value.add(tag='train_edit', simple_value=np.mean(trscores))
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

                    if minscore>tscore:
                        minscore=tscore
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


def ibm_train():
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


    iterations = 50000
    max_train_iterations = 100000
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
            minscore=1000
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

                    if i<=max_train_iterations:
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
                        print('tr edit {} vs te edit {}'.format(np.mean(trscores), np.mean(tescores)))
                        #print('test rec {} prec {}'.format(np.mean(tescores2), np.mean(tescores3)))
                        print('test loss {}'.format(tloss))
                        summary.value.add(tag='train_edit', simple_value=np.mean(trscores))
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

                    if minscore>tscore:
                        minscore=tscore
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

if __name__ == '__main__':
    moodle_train()
    # moodle_test()
    # financial_log_train()
    # financial_log_test()
    # ibm_train
    # ibm_test()