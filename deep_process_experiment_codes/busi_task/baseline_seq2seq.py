import numpy as np
import tensorflow as tf
import seq_helper
import matplotlib.pyplot as plt
import pickle
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import nltk
import os
import busi_task.presuf_train as pt


def build_seq2seq2(vocab_size_in =10, vocab_size_out =10,use_emb=True,
                   input_embedding_size = 20, output_embedding_size = 20,
                   encoder_hidden_units=20, max_decoder_time_att=0):


    '''HYPER PARAM'''

    input_embedding_size = input_embedding_size

    encoder_hidden_units = encoder_hidden_units
    decoder_hidden_units = encoder_hidden_units*2 # bidirectional

    '''PREPARE INPUT OUTPUT PLACEHOLDER'''
    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


    '''EMBEDDING LAYER'''
    embeddings_in = tf.Variable(tf.random_uniform([vocab_size_in, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    embeddings_out = tf.Variable(tf.random_uniform([vocab_size_out, output_embedding_size], -1.0, 1.0), dtype=tf.float32)
    if use_emb:
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_in, encoder_inputs)
        print(encoder_inputs_embedded.shape)
    else:
        encoder_inputs_embedded = tf.one_hot(encoder_inputs, depth=vocab_size_in)
        print(encoder_inputs_embedded.shape)

    '''THE CORE LSTM for ENCODER'''
    encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
    decoder_max_time, _ = tf.unstack(tf.shape(decoder_targets))


    encoder_cell = LSTMCell(encoder_hidden_units)
    ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state,encoder_bw_final_state)) = \
        (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, #bidirectional LSTM
                                        cell_bw=encoder_cell,
                                        inputs=encoder_inputs_embedded,
                                        sequence_length=encoder_inputs_length,# variant length support, maybe still batch but gradient is different
                                        dtype=tf.float32, time_major=True)
    )
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2) # merge 2 directions

    hidden_size = encoder_outputs.shape[2]

    # merge 2 directions
    encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

    # softmax
    # Attention mechanism


    #bs x seql x max_decode_l
    if max_decoder_time_att>0:
        W_omega = tf.get_variable('eo2a_w', [hidden_size, max_decoder_time_att],
                                  initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        b_omega = tf.get_variable('eo2a_b', [max_decoder_time_att],
                                  initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        # u_omega = tf.Variable(tf.random_normal([max_decoder_time], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(encoder_outputs, [-1, int(hidden_size)]), W_omega) + tf.reshape(b_omega, [1, -1]))
        # vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(v), [-1, max_decoder_time_att])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        alphas = tf.reshape(alphas, [-1, encoder_max_time, max_decoder_time_att, 1])
        alphas = tf.transpose(alphas,[1,0,2,3])
        temp = tf.stack([encoder_outputs] * max_decoder_time_att, axis=3)
        temp = tf.transpose(temp,[0,1,3,2])
        # Output of Bi-RNN is reduced with attention vector
        att = tf.reshape(tf.reduce_sum(temp * alphas, 0), [batch_size, max_decoder_time_att, int(hidden_size)]) # bs x decode_l x hid

    encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

    encoder_final_state = LSTMStateTuple(# the state value and the hidden value
        c=encoder_final_state_c,
        h=encoder_final_state_h
    )

    '''THE CORE LSTM for DECODER'''
    decoder_cell = LSTMCell(decoder_hidden_units)


    # softmax
    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size_out], -1, 1), dtype=tf.float32)
    b = tf.Variable(tf.zeros([vocab_size_out]), dtype=tf.float32)

    '''NEED TO EMBED EOS TOKEN and PAD TOKEN TOO'''
    eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS') #must be 1
    pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD') #must be 0

    if use_emb:
        eos_step_embedded = tf.nn.embedding_lookup(embeddings_out, eos_time_slice)
        pad_step_embedded = tf.nn.embedding_lookup(embeddings_out, pad_time_slice)
    else:
        eos_step_embedded=tf.one_hot(eos_time_slice,depth=vocab_size_out)
        pad_step_embedded=tf.one_hot(pad_time_slice,depth=vocab_size_out)

    '''STEP BY STEP OF DECODER'''
    def loop_fn_initial():
        initial_elements_finished = (0 >= decoder_max_time)  # all False at the initial step
        initial_input = eos_step_embedded
        if max_decoder_time_att>0:
            initial_input = tf.concat([initial_input,
                              tf.reshape(tf.slice(att, [0,0,0],[batch_size,1,int(hidden_size)]),
                                         [batch_size,int(hidden_size)])], axis=1)
        initial_cell_state = encoder_final_state
        initial_cell_output = None
        initial_loop_state = None  # we don't need to pass any additional information
        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

    def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

        def get_next_input():#next input for decoder step is pervious predicted one
            output_logits = tf.add(tf.matmul(previous_output, W), b)
            prediction = tf.argmax(output_logits, axis=1)
            if use_emb:
                next_input = tf.nn.embedding_lookup(embeddings_out, prediction)
            else:
                next_input = tf.one_hot(prediction, depth=vocab_size_out)
            return next_input # embedding vector of the output prediction, should not use raw one hot

        elements_finished = (time >= decoder_max_time)  # this operation produces boolean tensor of [batch_size]
        # defining if corresponding sequence has ended

        finished = tf.reduce_all(elements_finished)  # -> boolean scalar
        input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)#padding redundant output
        if max_decoder_time_att>0:
            input = tf.concat([input,
                              tf.reshape(tf.slice(att, [0,time,0],[batch_size,1,int(hidden_size)]),
                                         [batch_size,int(hidden_size)])], axis=1)
        state = previous_state
        output = previous_output
        loop_state = None

        return (elements_finished,
                input, #next input of encoder -->this function only find this
                state, #keep state and perivous output for other function update
                output,
                loop_state)

    def loop_fn(time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:  # time == 0
            assert previous_output is None and previous_state is None
            return loop_fn_initial()
        else:
            return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)#apply LSTM steps
    decoder_outputs = decoder_outputs_ta.stack()
    decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
    decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
    #softmax layer
    decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
    decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size_out))
    prob = tf.nn.softmax(decoder_logits, dim=-1)
    decoder_prediction = tf.argmax(prob, -1)
    '''OPTIMIZER'''

    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size_out, dtype=tf.float32),
        logits=decoder_logits,
    )
    loss = tf.reduce_mean(stepwise_cross_entropy)


    train_op = tf.train.AdamOptimizer().minimize(loss)

    return encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, prob


def build_seq2seq_mask(vocab_size_in =10, vocab_size_out =10,use_emb=True,
                   input_embedding_size = 20, output_embedding_size = 20,
                       encoder_hidden_units=20, max_decoder_time_att=0):


    '''HYPER PARAM'''

    input_embedding_size = input_embedding_size

    encoder_hidden_units = encoder_hidden_units
    decoder_hidden_units = encoder_hidden_units*2 # bidirectional

    '''PREPARE INPUT OUTPUT PLACEHOLDER'''
    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


    '''EMBEDDING LAYER'''
    embeddings_in = tf.Variable(tf.random_uniform([vocab_size_in, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    embeddings_out = tf.Variable(tf.random_uniform([vocab_size_out, output_embedding_size], -1.0, 1.0), dtype=tf.float32)
    if use_emb:
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_in, encoder_inputs)
        print(encoder_inputs_embedded.shape)
    else:
        encoder_inputs_embedded = tf.one_hot(encoder_inputs, depth=vocab_size_in)
        print(encoder_inputs_embedded.shape)

    '''THE CORE LSTM for ENCODER'''
    encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
    decoder_max_time, _ = tf.unstack(tf.shape(decoder_targets))


    encoder_cell = LSTMCell(encoder_hidden_units)
    ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state,encoder_bw_final_state)) = \
        (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, #bidirectional LSTM
                                        cell_bw=encoder_cell,
                                        inputs=encoder_inputs_embedded,
                                        sequence_length=encoder_inputs_length,# variant length support, maybe still batch but gradient is different
                                        dtype=tf.float32, time_major=True)
    )
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2) # merge 2 directions

    hidden_size = encoder_outputs.shape[2]

    # merge 2 directions
    encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

    # softmax
    # Attention mechanism


    #bs x seql x max_decode_l
    if max_decoder_time_att>0:
        W_omega = tf.get_variable('eo2a_w', [hidden_size, max_decoder_time_att],
                                  initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        b_omega = tf.get_variable('eo2a_b', [max_decoder_time_att],
                                  initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        # u_omega = tf.Variable(tf.random_normal([max_decoder_time], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(encoder_outputs, [-1, int(hidden_size)]), W_omega) + tf.reshape(b_omega, [1, -1]))
        # vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(v), [-1, max_decoder_time_att])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        alphas = tf.reshape(alphas, [-1, encoder_max_time, max_decoder_time_att, 1])
        alphas = tf.transpose(alphas,[1,0,2,3])
        temp = tf.stack([encoder_outputs] * max_decoder_time_att, axis=3)
        temp = tf.transpose(temp,[0,1,3,2])
        # Output of Bi-RNN is reduced with attention vector
        att = tf.reshape(tf.reduce_sum(temp * alphas, 0), [batch_size, max_decoder_time_att, int(hidden_size)]) # bs x decode_l x hid

    encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

    encoder_final_state = LSTMStateTuple(# the state value and the hidden value
        c=encoder_final_state_c,
        h=encoder_final_state_h
    )

    '''THE CORE LSTM for DECODER'''
    decoder_cell = LSTMCell(decoder_hidden_units)


    # softmax
    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size_out], -1, 1), dtype=tf.float32)
    b = tf.Variable(tf.zeros([vocab_size_out]), dtype=tf.float32)

    '''NEED TO EMBED EOS TOKEN and PAD TOKEN TOO'''
    eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS') #must be 1
    pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD') #must be 0

    if use_emb:
        eos_step_embedded = tf.nn.embedding_lookup(embeddings_out, eos_time_slice)
        pad_step_embedded = tf.nn.embedding_lookup(embeddings_out, pad_time_slice)
    else:
        eos_step_embedded=tf.one_hot(eos_time_slice,depth=vocab_size_out)
        pad_step_embedded=tf.one_hot(pad_time_slice,depth=vocab_size_out)

    '''STEP BY STEP OF DECODER'''
    def loop_fn_initial():
        initial_elements_finished = (0 >= decoder_max_time)  # all False at the initial step
        initial_input = eos_step_embedded
        if max_decoder_time_att>0:
            initial_input = tf.concat([initial_input,
                              tf.reshape(tf.slice(att, [0,0,0],[batch_size,1,int(hidden_size)]),
                                         [batch_size,int(hidden_size)])], axis=1)
        initial_cell_state = encoder_final_state
        initial_cell_output = None
        initial_loop_state = None  # we don't need to pass any additional information
        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

    def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

        def get_next_input():#next input for decoder step is pervious predicted one
            output_logits = tf.add(tf.matmul(previous_output, W), b)
            prediction = tf.argmax(output_logits, axis=1)
            if use_emb:
                next_input = tf.nn.embedding_lookup(embeddings_out, prediction)
            else:
                next_input = tf.one_hot(prediction, depth=vocab_size_out)
            return next_input # embedding vector of the output prediction, should not use raw one hot

        elements_finished = (time >= decoder_max_time)  # this operation produces boolean tensor of [batch_size]
        # defining if corresponding sequence has ended

        finished = tf.reduce_all(elements_finished)  # -> boolean scalar
        input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)#padding redundant output
        if max_decoder_time_att>0:
            input = tf.concat([input,
                              tf.reshape(tf.slice(att, [0,time,0],[batch_size,1,int(hidden_size)]),
                                         [batch_size,int(hidden_size)])], axis=1)
        state = previous_state
        output = previous_output
        loop_state = None

        return (elements_finished,
                input, #next input of encoder -->this function only find this
                state, #keep state and perivous output for other function update
                output,
                loop_state)

    def loop_fn(time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:  # time == 0
            assert previous_output is None and previous_state is None
            return loop_fn_initial()
        else:
            return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)#apply LSTM steps
    decoder_outputs = decoder_outputs_ta.stack()
    decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
    decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
    #softmax layer
    decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
    decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size_out))
    prob = tf.nn.softmax(decoder_logits, dim=-1)
    decoder_prediction = tf.argmax(prob, -1)
    '''OPTIMIZER'''
    '''
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size_out, dtype=tf.float32),
        logits=decoder_logits,
    )
    loss = tf.reduce_mean(stepwise_cross_entropy)
    '''

    mask = tf.placeholder(tf.bool, [None, None], name='mask')
    score = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size_out, dtype=tf.float32),
        logits=decoder_logits, dim=-1)
    score_flatten = tf.reshape(score, [-1])
    mask_flatten = tf.reshape(mask, [-1])
    mask_score = tf.boolean_mask(score_flatten, mask_flatten)

    loss = tf.reduce_mean(mask_score)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    return encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, prob, mask



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
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([trim_target], trim_predict,  weights=[0.5,0.5])
        s.append(BLEUscore)
    return np.mean(s)

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


def set_score(target_batch, predict_batch):
    s = []
    s2 = []
    for b in range(target_batch.shape[0]):
        for t, t2 in zip(target_batch[b], predict_batch[b]):
            # print('{} ----- {}'.format(t, t2))
            trim_target = []
            for tt in t:
                if tt > 1:
                    trim_target.append(tt)


            trim_predict = []
            for tt in t2:
                if tt > 1:
                    trim_predict.append(tt)
            if np.random.rand() > 0.99:
                print('{} vs {}'.format(trim_target, trim_predict))
            acc = len(set(trim_target).intersection(set(trim_predict))) / len(set(trim_target))
            acc2 = 0
            if len(set(trim_predict)) > 0:
                acc2 = len(set(trim_target).intersection(set(trim_predict))) / len(trim_predict)
            s.append(acc)
            s2.append(acc2)
    return np.mean(s), np.mean(s2)


def moddle_task():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_seq2seq_moddle')
    batch_size = 10

    _, _, _, char2label = pt.load_dict(dir='./data/BusinessProcess/Moddle')

    str_in, strain_out, stest_in, stest_out = pt.load_sequence(dir='./data/BusinessProcess/Moddle')

    maxlout=0

    for out in strain_out+stest_out:
        maxlout=max(maxlout,len(out))

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))

    input_size = len(char2label) + 1
    output_size = len(char2label) + 1
    max_decoder_time_att = maxlout+1
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, prob, mask = \
        build_seq2seq_mask(vocab_size_in=input_size, vocab_size_out=output_size,use_emb=False,
                       input_embedding_size = 64, output_embedding_size=64,
                       encoder_hidden_units=100, max_decoder_time_att=max_decoder_time_att)
    sess.run(tf.global_variables_initializer())

    batches = seq_helper.mimic_sequence(str_in, strain_out, batch_size)

    print('head of the batch:')
    for seq in next(batches)[:10]:
        print('{} vs {}'.format(seq[0], seq[1]))

    def next_feed(encoder_inputs, encoder_inputs_length, decoder_targets):
        batch = next(batches)
        inputs = []
        outputs = []

        for b in batch:
            inputs.append(b[0])
            outputs.append(b[1])
        encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
        decoder_targets_, _, mask_ = seq_helper.batch_mask(
            [sequence for sequence in outputs]
        )
        # print(decoder_targets_)
        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_,
            mask:mask_
        }, outputs

    loss_track = []
    min_loss=10000
    max_batches = 100000
    batches_in_epoch = 1000

    train_writer = tf.summary.FileWriter('./data/log_moddle_seq2seq/', sess.graph)
    try:
        for batch in range(max_batches):
            fd, rout =next_feed(encoder_inputs, encoder_inputs_length, decoder_targets)
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            # print('xxxx {}'.format(l))
            if batch == 0 or batch % batches_in_epoch == 0:
                summary = tf.Summary()
                print('episode {} --> minibatch loss: {}'.format(batch, np.mean(loss_track)))
                summary.value.add(tag='batch_train_loss', simple_value=np.mean(loss_track))
                loss_track=[]
                trbleu_scores = []
                for _ in range(100):
                    tfd,_ = next_feed(encoder_inputs, encoder_inputs_length, decoder_targets)
                    predict_ = sess.run(decoder_prediction, tfd)
                    predict_ = np.transpose(predict_, [1, 0])
                    bout_list=[]
                    for b in range(predict_.shape[0]):
                        out_list = []
                        for io in range(predict_.shape[1]):
                            if predict_[b][io] == 0:
                                break
                            out_list.append(predict_[b][io])
                        bout_list.append(out_list)

                    trbleu_scores.append(pt.batch_norm_edit_score(rout, bout_list, 0.95))



                print('----')
                ret_b = seq_helper.mimic_sequence_all(batch_size, stest_in, stest_out)
                tebleu_scores=[]
                tloss=[]
                for bat in ret_b:
                    inputs = []
                    outputs = []
                    # print(len(bat))
                    for b in bat:
                        inputs.append(b[0])
                        outputs.append(b[1])

                    encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
                    decoder_targets_, _, mask_ = seq_helper.batch_mask(outputs)
                    tfd =  {encoder_inputs: encoder_inputs_,
                        encoder_inputs_length: encoder_input_lengths_,
                        decoder_targets: decoder_targets_,
                            mask:mask_}
                    predict_, l2, prob_ = sess.run([decoder_prediction, loss, prob], tfd)
                    predict_ = np.transpose(predict_, [1, 0])
                    # prob_ = np.transpose(prob_, [1, 0, 2])
                    # pind = np.argsort(prob_, axis=-1)
                    bout_list = []
                    for b in range(predict_.shape[0]):
                        out_list = []
                        for io in range(predict_.shape[1]):
                            if predict_[b][io] == 0:
                                break
                            out_list.append(predict_[b][io])
                        bout_list.append(out_list)

                    tebleu_scores.append(pt.batch_norm_edit_score(outputs, bout_list, 0.95))
                    tloss.append(l2)

                print('trbleu score {} vs tebleu {}'.format(np.mean(trbleu_scores),
                                                                       np.mean(tebleu_scores)))
                print('test loss {}'.format(np.mean(tloss)))
                summary.value.add(tag='train_bleu', simple_value=np.mean(trbleu_scores))
                summary.value.add(tag='test_bleu', simple_value=np.mean(tebleu_scores))
                summary.value.add(tag='test_loss', simple_value=np.mean(tloss))

                train_writer.add_summary(summary, batch)
                train_writer.flush()

                if min_loss>np.mean(tloss):
                    min_loss = np.mean(tloss)
                    checkpoint_dir = os.path.join(ckpts_dir, 'att{}'.format(max_decoder_time_att))

                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    print("Saving Checkpoint ... ")
                    tf.train.Saver(tf.trainable_variables()).save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))


    except KeyboardInterrupt:
        print('training interrupted')


def test_moddle():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_seq2seq_moddle')
    batch_size = 10

    _, _, _, char2label = pt.load_dict()

    str_in, strain_out, stest_in, stest_out = pt.load_sequence()

    maxlout = 0

    for out in strain_out + stest_out:
        maxlout = max(maxlout, len(out))

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))

    input_size = len(char2label) + 1
    output_size = len(char2label) + 1
    max_decoder_time_att = maxlout + 1
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, prob = \
        build_seq2seq2(vocab_size_in=input_size, vocab_size_out=output_size, use_emb=False,
                       input_embedding_size=64, output_embedding_size=64,
                       encoder_hidden_units=100, max_decoder_time_att=max_decoder_time_att)
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(tf.trainable_variables()).restore(sess, os.path.join(ckpts_dir, 'att{}'.format(max_decoder_time_att)))
    batches = seq_helper.mimic_sequence(str_in, strain_out, batch_size)

    print('head of the batch:')
    for seq in next(batches)[:10]:
        print('{} vs {}'.format(seq[0], seq[1]))

    def next_feed(encoder_inputs, encoder_inputs_length, decoder_targets):
        batch = next(batches)
        inputs = []
        outputs = []
        for b in batch:
            inputs.append(b[0])
            outputs.append(b[1])
        encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
        decoder_targets_, _ = seq_helper.batch(
            [sequence for sequence in outputs]
        )
        # print(decoder_targets_)
        return {
                   encoder_inputs: encoder_inputs_,
                   encoder_inputs_length: encoder_input_lengths_,
                   decoder_targets: decoder_targets_,
               }, outputs

    loss_track = []
    min_loss = 10000
    max_batches = 100000
    batches_in_epoch = 1000


    ret_b = seq_helper.mimic_sequence_all(batch_size, stest_in, stest_out)
    tebleu_scores = []
    tloss = []
    for bat in ret_b:
        inputs = []
        outputs = []
        # print(len(bat))
        for b in bat:
            inputs.append(b[0])
            outputs.append(b[1])

        encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
        decoder_targets_, _ = seq_helper.batch(outputs)
        tfd = {encoder_inputs: encoder_inputs_,
               encoder_inputs_length: encoder_input_lengths_,
               decoder_targets: decoder_targets_}
        predict_, l2, prob_ = sess.run([decoder_prediction, loss, prob], tfd)
        predict_ = np.transpose(predict_, [1, 0])
        # prob_ = np.transpose(prob_, [1, 0, 2])
        # pind = np.argsort(prob_, axis=-1)
        bout_list = []
        for b in range(predict_.shape[0]):
            out_list = []
            for io in range(predict_.shape[1]):
                if predict_[b][io] == 0:
                    break
                out_list.append(predict_[b][io])
            bout_list.append(out_list)

        tebleu_scores.append(pt.batch_norm_edit_score(outputs, bout_list, 0.95))
        tloss.append(l2)

        print('tebleu {}'.format(np.mean(tebleu_scores)))
        print('test loss {}'.format(np.mean(tloss)))


def financial_log_task():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    # ckpts_dir = os.path.join(dirname, 'checkpoints_seq2seq_moddle')
    ckpts_dir = os.path.join(dirname, 'checkpoints_seq2seq_financial_log')
    batch_size = 10

    _, _, _, char2label = pt.load_dict(dir='./data/BusinessProcess/Financial_Log')

    str_in, strain_out, stest_in, stest_out = pt.load_sequence(dir='./data/BusinessProcess/Financial_Log')

    maxlout = 0

    for out in strain_out + stest_out:
        maxlout = max(maxlout, len(out))

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))

    input_size = len(char2label) + 1
    output_size = len(char2label) + 1
    max_decoder_time_att = maxlout + 2
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, prob, mask = \
        build_seq2seq_mask(vocab_size_in=input_size, vocab_size_out=output_size, use_emb=False,
                       input_embedding_size=64, output_embedding_size=64,
                       encoder_hidden_units=100, max_decoder_time_att=max_decoder_time_att)
    sess.run(tf.global_variables_initializer())

    batches = seq_helper.mimic_sequence(str_in, strain_out, batch_size)

    print('head of the batch:')
    for seq in next(batches)[:10]:
        print('{} vs {}'.format(seq[0], seq[1]))

    def next_feed(encoder_inputs, encoder_inputs_length, decoder_targets):
        batch = next(batches)
        inputs = []
        outputs = []
        for b in batch:
            inputs.append(b[0])
            outputs.append(b[1])
        encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
        decoder_targets_, _, mask_ = seq_helper.batch_mask(
            [sequence for sequence in outputs]
        )
        # print(decoder_targets_)
        return {
                   encoder_inputs: encoder_inputs_,
                   encoder_inputs_length: encoder_input_lengths_,
                   decoder_targets: decoder_targets_,
                   mask:mask_
               }, outputs

    loss_track = []
    min_loss = 10000
    max_batches = 100000
    batches_in_epoch = 1000

    # train_writer = tf.summary.FileWriter('./data/log_moddle_seq2seq/', sess.graph)
    train_writer = tf.summary.FileWriter('./data/log_financial_log_seq2seq/', sess.graph)
    try:
        for batch in range(max_batches):
            fd, rout = next_feed(encoder_inputs, encoder_inputs_length, decoder_targets)
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            # print('xxxx {}'.format(l))
            if batch == 0 or batch % batches_in_epoch == 0:
                summary = tf.Summary()
                print('episode {} --> minibatch loss: {}'.format(batch, np.mean(loss_track)))
                summary.value.add(tag='batch_train_loss', simple_value=np.mean(loss_track))
                loss_track = []
                trbleu_scores = []
                for _ in range(100):
                    tfd, _ = next_feed(encoder_inputs, encoder_inputs_length, decoder_targets)
                    predict_ = sess.run(decoder_prediction, tfd)
                    predict_ = np.transpose(predict_, [1, 0])
                    bout_list = []
                    for b in range(predict_.shape[0]):
                        out_list = []
                        for io in range(predict_.shape[1]):
                            if predict_[b][io] == 0:
                                break
                            out_list.append(predict_[b][io])
                        bout_list.append(out_list)

                    trbleu_scores.append(pt.batch_norm_edit_score(rout, bout_list, 0.95))

                print('----')
                ret_b = seq_helper.mimic_sequence_all(batch_size, stest_in, stest_out)
                tebleu_scores = []
                tloss = []
                for bat in ret_b:
                    inputs = []
                    outputs = []
                    # print(len(bat))
                    for b in bat:
                        inputs.append(b[0])
                        outputs.append(b[1])

                    encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
                    decoder_targets_, _, mask_ = seq_helper.batch_mask(outputs)
                    tfd = {encoder_inputs: encoder_inputs_,
                           encoder_inputs_length: encoder_input_lengths_,
                           decoder_targets: decoder_targets_,
                           mask:mask_}
                    predict_, l2, prob_ = sess.run([decoder_prediction, loss, prob], tfd)
                    predict_ = np.transpose(predict_, [1, 0])
                    # prob_ = np.transpose(prob_, [1, 0, 2])
                    # pind = np.argsort(prob_, axis=-1)
                    bout_list = []
                    for b in range(predict_.shape[0]):
                        out_list = []
                        for io in range(predict_.shape[1]):
                            if predict_[b][io] == 0:
                                break
                            out_list.append(predict_[b][io])
                        bout_list.append(out_list)

                    tebleu_scores.append(pt.batch_norm_edit_score(outputs, bout_list, 0.95))
                    tloss.append(l2)

                print('trbleu score {} vs tebleu {}'.format(np.mean(trbleu_scores),
                                                            np.mean(tebleu_scores)))
                print('test loss {}'.format(np.mean(tloss)))
                summary.value.add(tag='train_bleu', simple_value=np.mean(trbleu_scores))
                summary.value.add(tag='test_bleu', simple_value=np.mean(tebleu_scores))
                summary.value.add(tag='test_loss', simple_value=np.mean(tloss))

                train_writer.add_summary(summary, batch)
                train_writer.flush()

                if min_loss > np.mean(tloss):
                    min_loss = np.mean(tloss)
                    checkpoint_dir = os.path.join(ckpts_dir, 'att{}'.format(max_decoder_time_att))

                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    print("Saving Checkpoint ... ")
                    tf.train.Saver(tf.trainable_variables()).save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))


    except KeyboardInterrupt:
        print('training interrupted')


def test_financial_log():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_seq2seq_moddle')
    batch_size = 10

    _, _, _, char2label = pt.load_dict()

    str_in, strain_out, stest_in, stest_out = pt.load_sequence()

    maxlout = 0

    for out in strain_out + stest_out:
        maxlout = max(maxlout, len(out))

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))

    input_size = len(char2label) + 1
    output_size = len(char2label) + 1
    max_decoder_time_att = maxlout + 1
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, prob = \
        build_seq2seq2(vocab_size_in=input_size, vocab_size_out=output_size, use_emb=False,
                       input_embedding_size=64, output_embedding_size=64,
                       encoder_hidden_units=100, max_decoder_time_att=max_decoder_time_att)
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(tf.trainable_variables()).restore(sess,
                                                     os.path.join(ckpts_dir, 'att{}'.format(max_decoder_time_att)))
    batches = seq_helper.mimic_sequence(str_in, strain_out, batch_size)

    print('head of the batch:')
    for seq in next(batches)[:10]:
        print('{} vs {}'.format(seq[0], seq[1]))

    def next_feed(encoder_inputs, encoder_inputs_length, decoder_targets):
        batch = next(batches)
        inputs = []
        outputs = []
        for b in batch:
            inputs.append(b[0])
            outputs.append(b[1])
        encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
        decoder_targets_, _ = seq_helper.batch(
            [sequence for sequence in outputs]
        )
        # print(decoder_targets_)
        return {
                   encoder_inputs: encoder_inputs_,
                   encoder_inputs_length: encoder_input_lengths_,
                   decoder_targets: decoder_targets_,
               }, outputs

    loss_track = []
    min_loss = 10000
    max_batches = 100000
    batches_in_epoch = 1000

    ret_b = seq_helper.mimic_sequence_all(batch_size, stest_in, stest_out)
    tebleu_scores = []
    tloss = []
    for bat in ret_b:
        inputs = []
        outputs = []
        # print(len(bat))
        for b in bat:
            inputs.append(b[0])
            outputs.append(b[1])

        encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
        decoder_targets_, _ = seq_helper.batch(outputs)
        tfd = {encoder_inputs: encoder_inputs_,
               encoder_inputs_length: encoder_input_lengths_,
               decoder_targets: decoder_targets_}
        predict_, l2, prob_ = sess.run([decoder_prediction, loss, prob], tfd)
        predict_ = np.transpose(predict_, [1, 0])
        # prob_ = np.transpose(prob_, [1, 0, 2])
        # pind = np.argsort(prob_, axis=-1)
        bout_list = []
        for b in range(predict_.shape[0]):
            out_list = []
            for io in range(predict_.shape[1]):
                if predict_[b][io] == 0:
                    break
                out_list.append(predict_[b][io])
            bout_list.append(out_list)

        tebleu_scores.append(pt.batch_norm_edit_score(outputs, bout_list, 0.95))
        tloss.append(l2)

        print('tebleu {}'.format(np.mean(tebleu_scores)))
        print('test loss {}'.format(np.mean(tloss)))


def ibm_task():
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    print(dirname)
    # ckpts_dir = os.path.join(dirname, 'checkpoints_seq2seq_moddle')
    ckpts_dir = os.path.join(dirname, 'checkpoints_seq2seq_ibm')
    batch_size = 10

    _, _, _, char2label = pt.load_dict(dir='./data/BusinessProcess/IBM_Anonymous')

    str_in, strain_out, stest_in, stest_out = pt.load_sequence(dir='./data/BusinessProcess/IBM_Anonymous')

    maxlout = 0

    for out in strain_out + stest_out:
        maxlout = max(maxlout, len(out))

    print(str_in[:10])
    print(strain_out[:10])
    print(stest_in[:10])
    print(stest_out[:10])

    print('num train {}'.format(len(str_in)))
    print('num test {}'.format(len(stest_in)))
    print('dim in  {}'.format(len(char2label)))
    print('dim out {}'.format(len(char2label)))

    input_size = len(char2label) + 1
    output_size = len(char2label) + 1
    max_decoder_time_att = maxlout + 2
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, prob, mask = \
        build_seq2seq_mask(vocab_size_in=input_size, vocab_size_out=output_size, use_emb=False,
                       input_embedding_size=64, output_embedding_size=64,
                       encoder_hidden_units=100, max_decoder_time_att=max_decoder_time_att)
    sess.run(tf.global_variables_initializer())

    batches = seq_helper.mimic_sequence(str_in, strain_out, batch_size)

    print('head of the batch:')
    for seq in next(batches)[:10]:
        print('{} vs {}'.format(seq[0], seq[1]))

    def next_feed(encoder_inputs, encoder_inputs_length, decoder_targets):
        batch = next(batches)
        inputs = []
        outputs = []
        for b in batch:
            inputs.append(b[0])
            outputs.append(b[1])
        encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
        decoder_targets_, _, mask_ = seq_helper.batch_mask(
            [sequence for sequence in outputs]
        )
        # print(decoder_targets_)
        return {
                   encoder_inputs: encoder_inputs_,
                   encoder_inputs_length: encoder_input_lengths_,
                   decoder_targets: decoder_targets_,
                   mask:mask_
               }, outputs

    loss_track = []
    min_loss = 10000
    max_batches = 100000
    batches_in_epoch = 1000

    # train_writer = tf.summary.FileWriter('./data/log_moddle_seq2seq/', sess.graph)
    train_writer = tf.summary.FileWriter('./data/log_ibm_seq2seq/', sess.graph)
    try:
        for batch in range(max_batches):
            fd, rout = next_feed(encoder_inputs, encoder_inputs_length, decoder_targets)
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            # print('xxxx {}'.format(l))
            if batch == 0 or batch % batches_in_epoch == 0:
                summary = tf.Summary()
                print('episode {} --> minibatch loss: {}'.format(batch, np.mean(loss_track)))
                summary.value.add(tag='batch_train_loss', simple_value=np.mean(loss_track))
                loss_track = []
                trbleu_scores = []
                for _ in range(100):
                    tfd, _ = next_feed(encoder_inputs, encoder_inputs_length, decoder_targets)
                    predict_ = sess.run(decoder_prediction, tfd)
                    predict_ = np.transpose(predict_, [1, 0])
                    bout_list = []
                    for b in range(predict_.shape[0]):
                        out_list = []
                        for io in range(predict_.shape[1]):
                            if predict_[b][io] == 0:
                                break
                            out_list.append(predict_[b][io])
                        bout_list.append(out_list)

                    trbleu_scores.append(pt.batch_norm_edit_score(rout, bout_list, 0.95))

                print('----')
                ret_b = seq_helper.mimic_sequence_all(batch_size, stest_in, stest_out)
                tebleu_scores = []
                tloss = []
                for bat in ret_b:
                    inputs = []
                    outputs = []
                    # print(len(bat))
                    for b in bat:
                        inputs.append(b[0])
                        outputs.append(b[1])

                    encoder_inputs_, encoder_input_lengths_ = seq_helper.batch(inputs)
                    decoder_targets_, _, mask_ = seq_helper.batch_mask(outputs)
                    tfd = {encoder_inputs: encoder_inputs_,
                           encoder_inputs_length: encoder_input_lengths_,
                           decoder_targets: decoder_targets_,
                           mask:mask_}
                    predict_, l2, prob_ = sess.run([decoder_prediction, loss, prob], tfd)
                    predict_ = np.transpose(predict_, [1, 0])
                    # prob_ = np.transpose(prob_, [1, 0, 2])
                    # pind = np.argsort(prob_, axis=-1)
                    bout_list = []
                    for b in range(predict_.shape[0]):
                        out_list = []
                        for io in range(predict_.shape[1]):
                            if predict_[b][io] == 0:
                                break
                            out_list.append(predict_[b][io])
                        bout_list.append(out_list)

                    tebleu_scores.append(pt.batch_norm_edit_score(outputs, bout_list, 0.95))
                    tloss.append(l2)

                print('trbleu score {} vs tebleu {}'.format(np.mean(trbleu_scores),
                                                            np.mean(tebleu_scores)))
                print('test loss {}'.format(np.mean(tloss)))
                summary.value.add(tag='train_bleu', simple_value=np.mean(trbleu_scores))
                summary.value.add(tag='test_bleu', simple_value=np.mean(tebleu_scores))
                summary.value.add(tag='test_loss', simple_value=np.mean(tloss))

                train_writer.add_summary(summary, batch)
                train_writer.flush()

                if min_loss > np.mean(tloss):
                    min_loss = np.mean(tloss)
                    checkpoint_dir = os.path.join(ckpts_dir, 'att{}'.format(max_decoder_time_att))

                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    print("Saving Checkpoint ... ")
                    tf.train.Saver(tf.trainable_variables()).save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))


    except KeyboardInterrupt:
        print('training interrupted')

if __name__ == '__main__':
    #moddle_task()
    #financial_log_task()
    ibm_task()