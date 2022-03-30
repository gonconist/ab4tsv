#!/bin/bash

A=tgt
B=hyps
C=def
D=hyps
dataset=dev                     # -d
encoding=swap_fc               	# -e
target_fc=$                   	# -tfc
hyps_starting_fc=$            	# -hfc1
hyps_ending_fc=$              	# -hfc2
permutation_invariance=False	  # -pi
save_preds=True			            # --save_preds
out_binary_preds=False		      #--out_binary_preds	
output_dir=results            	# --output_dir

python source/eval.py -a $A $B $C $D -d $dataset -e $encoding -pi $permutation_invariance
