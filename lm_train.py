#!/bin/env python

import argparse
from collections import Counter
import math
import nltk
from nltk import sent_tokenize, word_tokenize
import numpy as np
from pathlib import Path
import pickle
import random
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import time

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# Tokenize input text data
def get_tokens(text, tokenizer=word_tokenize):
    tokens = tokenizer(text.lower())
    tokens = np.array(tokens)
    tokens = np.reshape(tokens, [-1, ])
    return tokens

def preprocess(tokens, replace_singletons):
    if replace_singletons:
        fdist = nltk.FreqDist(tokens)
        tokens = [token if fdist[token] > 1 else '<UNK>' for token in tokens]

    reverse_dictionary = dict(enumerate(nltk.FreqDist(tokens).keys()))
    if not replace_singletons:
        reverse_dictionary[len(reverse_dictionary)] = '<UNK>'
    dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
    
    return tokens, dictionary, reverse_dictionary
    
# Build forward and backward dictionaries
def build_dataset(words):
    count = Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def RNN(x, n_input, n_hidden, weights, biases):

    # reshape to [?, n_input]
    batch_size = -1 # unkown batch size
    x = tf.reshape(x, [batch_size, n_input])

    # Generate a n_input element sequence of inputs
    # (e.g. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)

    # 1-layer LSTM with n_hidden units
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # 2-layer LSTM with n_hidden units each
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']




if __name__ == '__main__':
    parser = argparse.ArgumentParser("LSTM Language Model Practice -- Train Script")
    parser.add_argument('--input-file', type=str, required=True,
            help="The input text file")
    parser.add_argument('--model-file', type=str, required=False,
            help="Where to save the trained model")
    parser.add_argument('--n-input', type=int, default=3,
            help="Number of input nodes")
    parser.add_argument('--n-hidden', type=int, default=512,
            help="Number of hidden units")
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--training-iters', type=int, default=50000)
    parser.add_argument('--display-step', type=int, default=1000)
    parser.add_argument('--replace-singletons', action='store_true')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        text = f.read().replace('\n', ' ')

    # Data preprocessing
    tokens = get_tokens(text)
    # dictionary, reverse_dictionary = build_dataset(tokens)
    tokens, dictionary, reverse_dictionary = preprocess(tokens, args.replace_singletons)
    vocab_size = len(dictionary)

    # Constants and training parameters
    display_step   = args.display_step
    learning_rate  = args.learning_rate
    training_iters = args.training_iters
    n_input        = args.n_input
    n_hidden       = args.n_hidden

    # Tensorflow variables
    x = tf.placeholder("float", [None, n_input, 1], name="x")
    y = tf.placeholder("float", [None, vocab_size], name="y")
    weights = { 'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]), name="weights") }
    biases  = { 'out': tf.Variable(tf.random_normal([vocab_size]), name="biases") }

    # LSTM Prediction
    pred = RNN(x, n_input, n_hidden, weights, biases)

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), name="cost")
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # row-wise mean

    # Initialize variables
    init = tf.global_variables_initializer()

    # Training
    with tf.Session() as session:
        session.run(init)

        step = 0
        offset = random.randint(0, n_input + 1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        while step < training_iters:
            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(tokens) - end_offset):
                offset = random.randint(0, n_input + 1)

            minibatch_tokens = [tokens[i] for i in range(offset, offset+n_input)]
            symbols_in_keys  = [ [ dictionary[token] ] for token in minibatch_tokens]
            symbols_in_keys  = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

            next_token = tokens[offset + n_input]
            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[next_token]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                            feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

            loss_total += loss
            acc_total += acc

            if (step + 1) % display_step == 0:
                disp_vars = (step+1, loss_total/display_step, 100*acc_total/display_step)
                print("Iter= {}, Avg.Loss= {:.6f}, Avg.Acc= {:.2f}%".format(*disp_vars)) 
                acc_total = 0
                loss_total = 0

                pred_symbol = int(tf.argmax(onehot_pred, 1).eval())
                if reverse_dictionary[pred_symbol] == '<UNK>':
                    onehot_pred[0][pred_symbol] = np.nan
                    pred_symbol = int(tf.argmax(onehot_pred, 1).eval())

                symbols_in = minibatch_tokens
                symbols_out = next_token
                symbols_out_pred = reverse_dictionary[pred_symbol]
                print("{} - [{}] vs [{}]".format(symbols_in, symbols_out, symbols_out_pred))
            step += 1
            offset += (n_input + 1)

        print("Optimization Finished! - {} elapsed".format(elapsed(time.time() - start_time)))

        # Save the model
        if args.model_file:
            saver = tf.train.Saver()
            save_path = saver.save(session, args.model_file)
            pickle_vars = {
                "dictionary": dictionary,
                "reverse_dictionary": reverse_dictionary,
                "n_input": n_input, "n_hidden": n_hidden,
            }
            pickle_path = Path(args.model_file).parent.absolute().as_posix()
            pickle.dump(pickle_vars, open(pickle_path + '/vars.pkl', 'wb+'))
        print("Model saved to: ", args.model_file)



    
