#!/bin/bash

seed=42                       # --seed
num_epochs=5                  # --num_epochs
batch_size_train=16           # --batch_size_train
batch_size_val=8              # --batch_size_val
A=cls
B=cls
C=descr
D=descr
encoding=swap_fc_plus         # -e
target_fc=$                   # -tfc
hyps_starting_fc=$            # -hfc1
hyps_ending_fc=$              # -hfc2
permutation_invariance=False  # -pi
output_dir=results            # --output_dir

python source/train.py -a $A $B $C $D -e $encoding -pi $permutation_invariance