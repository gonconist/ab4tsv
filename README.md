# AB4TSV
### *Analogy and BERT for Target Sense Verification*

AB4TSV is a hybrid architecture that combines BERT with analogies for tackling Target Sense Verification (TSV). In this repository we provide scripts for training and evaluating AB4TSV on the WiC-TSV evaluation benchmark.

![alt text](https://github.com/gonconist/ab4tsv/blob/main/ab4tsv.png)

## Getting Started

```shell
pip install -r requirements.txt
```

## WiC-TSV data & baseline models

The WiC-TSV dataset as well as training and evaluation scripts for HyperBertCLS and HyperBert3 are available [here](https://github.com/semantic-web-company/wic-tsv).

## Training your own model

There are two equivalent ways to finetune AB4TSV on WiC-TSV.

### Run (2 steps)

Initialize the training parameters inside `scripts/train.sh`.
```shell
#!/bin/bash

A=tgt
B=hyps
C=def
D=hyps
encoding=swap_fc              # -e
target_fc=$                   # -tfc
hyps_start_fc=$               # -hfc1
hyps_end_fc=$                 # -hfc2
permutation_invariance=False  # -pi
seed=42                       # --seed
num_epochs=5                  # --num_epochs
batch_size_train=16           # --batch_size_train
batch_size_val=8              # --batch_size_val
output_dir=results            # --output_dir

python source/train.py -a $A $B $C $D -e $encoding -pi $permutation_invariance
```
Then simply run the following command:
```shell
bash ./scripts/train.sh
```
### Run (1 step)

Alternatively, pass the arguments of interest directly to `src/train.py`:
```shell
python src/train.py \
    --analogy tgt hyps def hyps \
    --encoding swap_fc \
    --permutation_invariance False
```

## Evaluating your finetuned model

Like training, there are two ways to evaluate the performance of your AB4TSV model.
*Note that performance results are obtained only on the __development set__ since the test set is private. For test set results submit your predictions at [codalab](https://competitions.codalab.org/competitions/23683).*

### Run (2 steps)

Initialize the training parameters inside `scripts/eval.sh`.
```shell
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
permutation_invariance=False	# -pi
save_preds=True			        # --save_preds
out_binary_preds=False		    #--out_binary_preds	
output_dir=results            	# --output_dir

python src/eval.py -a $A $B $C $D -d $dataset -e $encoding -pi $permutation_invariance
```
Then simply run the following command:
```shell
bash ./scripts/eval.sh
```

### Run (1 step)
Alternatively, pass the arguments of interest directly to `src/eval.py`:
```shell
python src/eval.py \
    --analogy tgt hyps def hyps \
    --encoding swap_fc \
    --permutation_invariance False
    --save_preds True
```
