#!/bin/bash

A=tgt
B=hyps
C=def
D=hyps
encoding=swap_fc              # -e
target_fc=$                   # -tfc
hyps_start_em=$               # -hem1
hyps_end_em=$                 # -hem2
permutation_invariance=False  # -pi

seed=42                       # --seed
num_epochs=5                  # --num_epochs
batch_size_train=16           # --batch_size_train
batch_size_val=8              # --batch_size_val
output_dir=results            # --output_dir

python source/train.py -a $A $B $C $D -e $encoding -pi $permutation_invariance
