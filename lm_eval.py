#!/bin/env python

import argparse
import math
import numpy as np
from pathlib import Path
import pickle
import tensorflow as tf

from lm_train import get_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser("LSTM Language Model Practice -- Eval Script")
    parser.add_argument('--model', type=str, required=True,
            help="Parent directory of the tensorflow model to test")
    parser.add_argument('--test-data', type=str, required=True,
            help="Test data to evaluate on")
    args = parser.parse_args()

    model = args.model
    with open(args.test_data, 'r') as f:
        text = f.read().replace('\n', ' ')
    tokens = get_tokens(text)

    pickle_path = Path(model).joinpath('vars.pkl').absolute().as_posix()
    pickle_vars = pickle.load(open(pickle_path, 'rb'))   
    n_input            = pickle_vars['n_input']
    n_hidden           = pickle_vars['n_hidden'] 
    dictionary         = pickle_vars['dictionary']
    reverse_dictionary = pickle_vars['reverse_dictionary']
    vocab_size         = len(dictionary)

    with tf.Session() as session:
        saver = tf.train.import_meta_graph(model + "/model.ckpt.meta")
        saver.restore(session, tf.train.latest_checkpoint(model))

        graph = tf.get_default_graph()
        pred = graph.get_tensor_by_name("add:0")
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        ops = graph.get_operations()
        cost = {op.name: op for op in graph.get_operations()}["cost"]

        total_loss = 0
        offset = 0

        while offset < len(tokens) - n_input - 1:
            unk_symbol = dictionary['<UNK>']
            minibatch_tokens = [tokens[i] for i in range(offset, offset+n_input)]
            symbols_in_keys  = [ [dictionary.get(token, unk_symbol)] for token in minibatch_tokens]
            symbols_in_keys  = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

            next_token = tokens[offset + n_input]
            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary.get(next_token, unk_symbol)] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            session.run(cost, feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
            loss = session.run(cost.outputs[0], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
            total_loss += loss
            offset += 1

        total_loss = float(total_loss)
        avg_loss   = float(total_loss / len(tokens))
        perplexity = math.exp(avg_loss)

        print("Total loss: {:.3f}, Avg. Loss: {:.3f}".format(total_loss, avg_loss))
        print("Model Perplexity: {:.3f}".format(perplexity))

        



