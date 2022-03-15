# AB4TSV
### *Analogy and BERT for Target Sense Verification*

AB4TSV is a hybrid architecture that combines BERT with a CNN classifier for tackling Target Sense Verification (TSV). In this repository we provide scripts for traning and evaluating AB4TSV on the WiC-TSV evaluation benchmark.

![alt text](https://github.com/gonconist/ab4tsv/blob/main/ab4tsv.png)

## Getting Started

```shell
pip install -r requirements.txt
```

## Training your own model

There are two equivalent ways to finetune AB4TSV on WiC-TSV.

### Run (2 steps)

Initialize the training parameters inside `scripts/train.sh`.
```shell
#!/bin/bash

seed=42                         # --seed
num_epochs=5                    # --num_epochs
batch_size_train=16             # --batch_size_train
batch_size_val=8                # --batch_size_val
A=def
B=ctx
C=cls
D=hyps
encoding=swap_fc_plus         	# -e
target_fc=$                     # -tfc
hyps_starting_fc=$              # -hfc1
hyps_ending_fc=$                # -hfc2
permutation_invariance=False    # -pi
output_dir=results              # --output_dir

python src/train.py -a $A $B $C $D -e $encoding -pi $permutation_invariance
```
Then simply run the following command:
```shell
bash ./scripts/train.sh
```
### Run (1 step)

Alternatively, pass the arguments of interest directly to `src/train.py` like this:
```shell
python src/train.py \
    --analogy def ctx cls hyps \
    --encoding swap_fc_plus \
    --permutation_invariance False
```

## Evaluating finetuned model

```shell
#!/bin/bash

A=cls
B=cls
C=descr				
D=descr
encoding=swap_fc_plus           # -e
target_fc=$                     # -tfc
hyps_starting_fc=$              # -hfc1
hyps_ending_fc=$              	# -hfc2
permutation_invariance=False    # -pi
is_test_set=False               # --is_test_set
save_preds=True                 # --save_preds
out_binary_preds=False          # --out_binary_preds	
output_dir=results              # --output_dir

python source/eval.py -a $A $B $C $D -e $encoding -pi $permutation_invariance --is_test_set $is_test_set --save_preds $save_preds --out_binary_preds $out_binary_preds
```
