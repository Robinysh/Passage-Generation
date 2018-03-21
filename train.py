from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
import pickle as pkl

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='data directory containing corpus.pkl')
    parser.add_argument('--save_dir', type=str, default='./saves',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=256,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--embed_dir', type=str, default='./glove',
                       help='directory containing GloVe txt files')
    parser.add_argument('--embed_dim', type=int, default=100,
                       help='dimensions of word embedding')
    parser.add_argument('--model', type=str, default='gru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='minibatch size')
    parser.add_argument('--max_length', type=int, default=100,
                       help='generated sequence max length')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--input_length', type=int, default=10,
                       help='RNN input length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop, learning rate decay by decay_rate**epoch_num')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='probablity of dropout for all rnn layers')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def train(args):
    assert os.path.isfile(os.path.join(args.embed_dir, "glove.6B.{}d.txt".format(args.embed_dim))), 'No GloVe file with {} dimensions found'.format(args.embed_dim) 

    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.input_length, args.embed_dim, args.embed_dir)
    args.vocab_size = data_loader.vocab_size
    embedding_matrix = data_loader.embedding_matrix 
    

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = pkl.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = pkl.load(f)
        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    #save configs 
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pkl.dump(args, f)
    #save word list
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        pkl.dump((data_loader.words, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #feed pretrained embeddings
        sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embedding_matrix})
        
        #loop through epoches
        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0

            if args.init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)

            if args.init_from is not None:
                data_loader.pointer = model.batch_pointer.eval()
                args.init_from = None
            
            #loop through batches
            for b in range(data_loader.pointer, data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
                train_loss, state, _, _ = sess.run([model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                speed = time.time() - start
                #print data
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                #save model
                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
