#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=2
~/tensorflow.sh sample.py --prime "神說" --save_dir="./saves/test_save" -n 400
