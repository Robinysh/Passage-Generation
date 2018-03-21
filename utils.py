# -*- coding: utf-8 -*-
import os
import codecs
import collections
import pickle as pkl
import numpy as np
import re
import itertools

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, input_length, embed_dim, embed_dir):
        self.data_dir   = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_length = input_length
        self.embed_dim  = embed_dim
        self.embed_dir  = embed_dir

        input_file  = os.path.join(data_dir, "corpus.pkl")
        vocab_file  = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")
        embed_file  = os.path.join(embed_dir, "glove.6B.{}d.txt".format(embed_dim))

        print("reading text file")
        self.preprocess(input_file, vocab_file, tensor_file, embed_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file, embed_file):
        with open(input_file, "rb") as f:
            x_text = pkl.load(f)

        # Optional text cleaning or make them lower case, etc.
        x_text = self.clean_str(x_text)
        x_text = x_text.split()

        self.embedding_matrix, self.vocab, self.words = self.build_vocab(x_text, embed_file)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            pkl.dump(self.words, f)

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        
        self.tensor = np.array(list(map(lambda x: self.vocab.get(x) if x in self.vocab else self.vocab.get('UNK'), x_text))) # Save the data to data.npy
        np.save(tensor_file, self.tensor)


    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def build_vocab(self, sentences, embed_file):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns pre-trained embedding, vocabulary mapping and inverse vocabulary mapping.
        """
        """
        W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                        trainable=False, name="W")

        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        embedding_init = W.assign(embedding_placeholder)

        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
        """
        PAD_TOKEN = 0
        SOS_TOKEN = 1
        EOS_TOKEN = 2

        word2idx = { 'PAD': PAD_TOKEN, 'SOS': SOS_TOKEN, 'EOS': EOS_TOKEN } # dict so we can lookup indices for tokenising our text later from string to sequence of integers
        idx2word = { PAD_TOKEN:'PAD', SOS_TOKEN:'SOS', EOS_TOKEN:'EOS' } # dict so we can lookup words from indicies
        weights = []

        with open(embed_file, 'r') as file:
            for index, line in enumerate(file):
                values = line.split() # Word and weights separated by space
                word = values[0] # Word is first symbol on each line
                word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word
                word2idx[word] = index + 3 # we have 3 special characters so we shift by 3
                idx2word[index+3] = word
                weights.append(word_weights)
                
                if index + 3 == 40_000:
                    # Limit vocabulary to top 40k terms
                    break

        EMBEDDING_DIMENSION = len(weights[0])
        # Insert the random weights at special indicies now we know the embedding dimension
        weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))
        weights.insert(1, np.random.randn(EMBEDDING_DIMENSION))
        weights.insert(2, np.random.randn(EMBEDDING_DIMENSION))

        # Append unknown to end of vocab and initialize as random
        UNKNOWN_TOKEN=len(weights)
        word2idx['UNK'] = UNKNOWN_TOKEN
        idx2word[UNKNOWN_TOKEN] = 'UNK'
        weights.append(np.random.randn(EMBEDDING_DIMENSION))

        # Construct our final vocab
        weights = np.asarray(weights, dtype=np.float32)

        VOCAB_SIZE=weights.shape[0]

        return weights, word2idx, idx2word 

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = pkl.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   (self.seq_length+self.input_length)))

    def create_batches(self):
        #features = {}
        #features['word_indices'] = nltk.word_tokenize('hello world') # ['hello', 'world']
        #features['word_indices'] = [word2idx.get(word, UNKNOWN_TOKEN) for word in features['word_indices']]
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   (self.seq_length+self.input_length)))
        assert self.num_batches!=0, "Not enough data. Make seq_length and batch_size smaller."
        self.tensor = self.tensor[:self.num_batches * self.batch_size * (self.seq_length+self.input_length)]

        batches = self.tensor.reshape(self.num_batches, self.batch_size, -1)
        self.x_batches = list(batches[:,:,:self.input_length])
        self.y_batches = list(batches[:,:,self.input_length:])
        assert self.y_batches[0].shape[1] == self.seq_length, 'length of y data doesnt equal to sequence length'

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
