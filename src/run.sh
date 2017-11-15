#!/usr/bin/env bash
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_iden_tr/cn_sep_1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_iden_tr/cn_sep_1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_iden_tr/cn_sep_1_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=cn_sep --log_dir=/tmp/logs/mnist_iden_tr/cn_sep_1_1.0 --lr=1.0
#
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=wn_sep --log_dir=/tmp/logs/mnist_iden_tr/wn_sep_0.1_1.0 --lr=1.0
#
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=ln_sep --log_dir=/tmp/logs/mnist_iden_tr/ln_sep_0.1_1.0 --lr=1.0
#
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.001  --lr=0.001
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.01  --lr=0.01
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_0.1 --lr=0.1
#python mnist/test_784*1.py --cell=bn_sep --log_dir=/tmp/logs/mnist_iden_tr/bn_sep_0.01_1.0 --lr=1.0


#python ptb/ptb_word_lm.py --lr=0.1 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb/cn_sep_1.0
#python ptb/ptb_word_lm.py --lr=1.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb/cn_sep_1.0
#python ptb/ptb_word_lm.py --lr=10.0 --rnn_mode=cn_sep --num_gpus=1 --save_path=/tmp/log/ptb/cn_sep_10.0
