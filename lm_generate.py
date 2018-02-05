#!/bin/env python

import argparse
from nltk import word_tokenize
import numpy as np
from pathlib import Path
import pickle
import tensorflow as tf

from lm_train import get_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser("LSTM Language Model Practice -- Generative Script")
    parser.add_argument('--model', type=str, required=True,
            help="Parent directory of the tensorflow model to test")
    parser.add_argument('--num-generate', type=int, default=32,
            help="The number of tokens to generate from the seed")
    args = parser.parse_args()

    model = args.model
    num_generate = args.num_generate

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

        unk_symbol = dictionary['<UNK>']

        print("Loaded model at: {}".format(model))
        print()

        while True:
            prompt = "{} words: ".format(n_input)
            sentence = input(prompt)
            tokens = get_tokens(sentence)

            if len(tokens) != n_input:
                continue

            symbols_in_keys = [ dictionary.get(token, unk_symbol) for token in tokens]
            for i in range(num_generate):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())

                if reverse_dictionary[onehot_pred_index] == '<UNK>':
                    onehot_pred[0][onehot_pred_index] = np.nan
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())

                predicted_token = reverse_dictionary[onehot_pred_index]
                sentence = "{} {}".format(sentence, predicted_token)

                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
            print()
