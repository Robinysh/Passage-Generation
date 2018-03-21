import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np


class Model():
    def __init__(self, args, infer=False):
        self.args = args

        #for sampling
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        #declare placeholders
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.input_length])
        self.target_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        #declare batch and epoch pointer
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)

        #declare softmax variables
        softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [args.vocab_size])


        #embedding
        self.embedding_matrix = tf.Variable(tf.constant(0.0, shape=[args.vocab_size, args.embed_dim]),
                        trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, [args.vocab_size, args.embed_dim])
        self.embedding_init = self.embedding_matrix.assign(self.embedding_placeholder)

        with tf.device("/cpu:0"): #select one cpu for embedding lookup
            #inputs = tf.split(tf.nn.embedding_lookup(self.embedding_matrix, self.input_data), args.seq_length, 1)
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_data)
            targets = tf.nn.embedding_lookup(self.embedding_matrix, self.target_data)
            #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def encoder(inputs, args):
            cells = []
            for _ in range(args.num_layers):
                cell = cell_fn(args.rnn_size, activation=tf.nn.relu)
                cell = rnn.DropoutWrapper(cell, args.dropout)
                cells.append(cell)
            self.initial_state = cell.zero_state(args.batch_size, tf.float32), cell.zero_state(args.batch_size, tf.float32)
            #self.initial_state = cell.zero_state(args.batch_size, tf.float32), cell.zero_state(args.batch_size, tf.float32)
            self.cell = cell = rnn.MultiRNNCell(cells)

            #   encoder_outputs: [max_time, batch_size, num_units]
            #   encoder_state: [batch_size, num_units]
            print(inputs)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell, inputs, initial_state=self.initial_state,
                sequence_length=tf.tile([args.input_length], [args.batch_size]), time_major=False)
            return encoder_outputs, encoder_state

        encoder_outputs, encoder_state = encoder(inputs, args)
        
        #declare decoder cells
        cells = []
        for _ in range(args.num_layers):
        		cell = cell_fn(args.rnn_size, activation=tf.nn.relu)
        		cell = rnn.DropoutWrapper(cell, args.dropout)
        		cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells)

        """
        #   decoder_outputs: [max_time, batch_size, num_units]
        #   decoder_state: [batch_size, num_units]
        outputs, last_state = tf.nn.dynamic_rnn(
            cell, encoder_outputs, initial_state=encoder_state, time_major=True)
        """

        if not infer: #train
          helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=targets,
            sequence_length=tf.tile([args.seq_length], [args.batch_size]))
        else:
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              embedding=embedding,
              start_tokens=tf.tile(self.embedding_matrix[1], [args.batch_size]),
              end_token=self.embedding_matrix[2])

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=cell.zero_state(args.batch_size, tf.float32))
        outputs, last_state, _ = tf.contrib.seq2seq.dynamic_decode(
           decoder=decoder,
           output_time_major=False,
           impute_finished=True,
           maximum_iterations=args.max_length)

        print(outputs)
        output = tf.reshape(tf.concat(outputs[0], 1), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        self.text_argmax = tf.stop_gradient(tf.argmax(self.probs, 1))
        self.text_output = tf.nn.embedding_lookup(self.embedding_matrix, self.text_argmax)

        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.target_data, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime=' '):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = ''
        state = sess.run(self.cell.zero_state(1, tf.float32))
        if not len(prime) or prime == ' ':
            prime  = random.choice(list(vocab.keys()))

        print (prime)
        for word in prime.split()[:-1]:
            print (word)
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word,0)
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        ret = prime
        word = prime.split()[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word, 0)
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)

            p = probs[0]
            sample = weighted_pick(p)
            pred = words[sample]
            ret += pred
            word = pred

        return ret
