from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit

from colorama import Fore
from sklearn.metrics import auc, roc_curve, precision_score, recall_score

from vocab import Vocabulary
from reader import Data
from utils import print_progress, create_checkpoints_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

params = {
    "batch_size": 128,
    "embed_size": 64,
    "hidden_size": 64,
    "num_layers": 2,
    "checkpoints": "./checkpoints/",
    "std_factor": 6.,
    "dropout": 0.7,
}

path_normal_data = "datasets/vulnbank_train.txt"
path_anomaly_data = "datasets/vulnbank_anomaly.txt"

create_checkpoints_dir(params["checkpoints"])

vocab = Vocabulary()
params["vocab"] = vocab

d = Data(path_normal_data)

#MODELLLLLLLLLL
class Seq2Seq():
    def __init__(self, args):
        tf.reset_default_graph()

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.max_seq_len = tf.placeholder(tf.int32, [], name='max_seq_len')
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.lengths = tf.placeholder(tf.int32, [None, ], name='lengths')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        
        self.num_layers = args['num_layers']
        self.hidden_size = args['hidden_size']
        self.vocab = args['vocab']

        dec_input = self._process_decoder_input(
            self.targets,
            self.vocab.vocab,
            tf.to_int32(self.batch_size))

        vocab_size = len(self.vocab.vocab)

        # Embeddings for inputs
        embed_initializer = tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3))

        with tf.variable_scope('embedding'):
            embeds = tf.get_variable(
                'embed_matrix',
                [vocab_size, args['embed_size']],
                initializer=embed_initializer,
                dtype=tf.float32)

            enc_embed_input = tf.nn.embedding_lookup(embeds, self.inputs)
            
        enc_state = self._encoder(enc_embed_input)
        
        # Embeddings for outputs
        with tf.variable_scope('embedding', reuse=True):
            dec_embed_input = tf.nn.embedding_lookup(embeds, dec_input)

        dec_outputs = self._decoder(enc_state, dec_embed_input)

        weight, bias = self._weight_and_bias(args['hidden_size'], vocab_size)
        outputs = tf.reshape(dec_outputs[0].rnn_output, [-1, args['hidden_size']])
        logits = tf.matmul(outputs, weight) + bias

        logits = tf.reshape(logits, [-1, self.max_seq_len, vocab_size], name='logits')
        self.probs = tf.nn.softmax(logits, name='probs')
        self.decoder_outputs = tf.argmax(logits, axis=2)

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=self.targets,
            name='cross_entropy')
        self.batch_loss = tf.identity(tf.reduce_mean(self.cross_entropy, axis=1), name='batch_loss')
        self.loss = tf.reduce_mean(self.cross_entropy)

        self.train_optimizer = self._optimizer(self.loss)

        # Saver
        self.saver = tf.train.Saver()
        
    def _encoder(self, enc_embed_input):
        """
        Adds an encoder to the model architecture.
        """
        cells = [self._lstm_cell(self.hidden_size) for _ in range(self.num_layers)]
        multilstm = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _, enc_state = tf.nn.dynamic_rnn(
            multilstm,
            enc_embed_input,
            sequence_length=self.lengths,
            swap_memory=True,
            dtype=tf.float32)
        
        return enc_state
    
    def _decoder(self, enc_state, dec_embed_input):
        """
        Adds a decoder to the model architecture.
        """
        output_lengths = tf.ones([self.batch_size], tf.int32) * self.max_seq_len
        helper = tf.contrib.seq2seq.TrainingHelper(
            dec_embed_input,
            output_lengths,
            time_major=False)

        cells = [self._lstm_cell(self.hidden_size) for _ in range(self.num_layers)]
        dec_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, enc_state)

        dec_outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=self.max_seq_len, swap_memory=True)
        
        return dec_outputs
    
    def _optimizer(self, loss,):
        """
        Optimizes weights given a loss. 
        """
        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        starting_lr = 0.001
        starting_global_step = tf.Variable(0, trainable=False)
        optimizer = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=starting_global_step,
            learning_rate=starting_lr,
            optimizer=tf.train.AdamOptimizer,
            learning_rate_decay_fn=lambda lr, gs: _learning_rate_decay_fn(lr, gs),
            clip_gradients=5.0)
        
        return optimizer
    
    def _process_decoder_input(self, target_data, char_to_code, batch_size):
        """
        Concatenates the <GO> to the begining of each batch.
        """
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], char_to_code['<GO>']), ending], 1)

        return dec_input

    def _lstm_cell(self, hidden_size):
        """
        Returns LSTM cell with dropout.
        """
        cell = tf.contrib.rnn.LSTMCell(
            hidden_size,
            initializer=tf.contrib.layers.xavier_initializer())

        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)

        return cell

    def _weight_and_bias(self, in_size, out_size):
        """
        Initializes weights and biases.
        """
        weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(1., shape=[out_size]))

        return weight, bias

class Trainer():

    def __init__(self, batch_size, checkpoints_path, dropout):
        self.batch_size = batch_size
        self.checkpoints = checkpoints_path
        self.path_to_graph = checkpoints_path + 'seq2seq'
        self.dropout = dropout

    def train(self, model, train_data, train_size, num_steps, num_epochs, min_loss=0.3):
        """
        Trains a given model architecture with given train data.
        """
        tf.set_random_seed(1234)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss = []
            timings = []
            steps_per_epoch = int(train_size / self.batch_size)
            num_epoch = 1
            
            for step in range(1, num_steps):
                beg_t = timeit.default_timer()
                X, L = train_data.next()
                seq_len = np.max(L)

                # For anomaly detection problem we reconstruct input data, so
                # targets and inputs are identical.
                feed_dict = {
                    model.inputs: X,
                    model.targets: X,
                    model.lengths: L,
                    model.dropout: self.dropout,
                    model.batch_size: self.batch_size,
                    model.max_seq_len: seq_len}
                
                fetches = [model.loss, model.decoder_outputs, model.train_optimizer]
                step_loss, _, _ = sess.run(fetches, feed_dict)

                total_loss.append(step_loss)
                timings.append(timeit.default_timer() - beg_t)

                if step % steps_per_epoch == 0:
                    num_epoch += 1

                if step % 200 == 0 or step == 1:
                    print_progress(
                        int(step / 200),
                        num_epoch,
                        np.mean(total_loss),
                        np.mean(step_loss),
                        np.sum(timings))
                    timings = []

                if step == 1:
                    _ = tf.train.export_meta_graph(filename=self.path_to_graph + '.meta')
                
                if np.mean(total_loss) < min_loss or num_epoch > num_epochs:
                    model.saver.save(sess, self.path_to_graph, global_step=step)
                    print("Training is finished.")
                    break

model = Seq2Seq(params)
t = Trainer(params["batch_size"], params["checkpoints"], params["dropout"])

num_steps = 10 ** 6
num_epochs = 60

train_gen = d.train_generator(params["batch_size"], num_epochs)
train_size = d.train_size

t.train(model, train_gen, train_size, num_steps, num_epochs)