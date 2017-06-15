#!/bin/bash
~/tensorflow.sh train.py --data_dir=./data --save_dir=./saves/test_save --rnn_size 256 --learning_rate 0.0004 --dropout 0.8 --num_layers 2 --batch_size 50 --num_epochs 50 --decay_rate 0.98
