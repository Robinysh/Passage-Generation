#!/bin/bash
python train.py --data_dir=./data --save_dir=./saves/test_save --rnn_size 1024 --learning_rate 0.0004 --dropout 0.2 --num_layers 2 --batch_size 128 --num_epochs 50 --decay_rate 0.98
