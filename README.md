# AB4TSV
### *Analogy and BERT for Target Sense Verification*

AB4TSV is a hybrid architecture that combines BERT with a CNN classifier for tackling Target Sense Verification (TSV). In this repository we provide scripts for traning and evaluating AB4TSV on the WiC-TSV evaluation benchmark.

## Getting Started

```shell
pip install -r requirements.txt
```

## Training your own model

First we need to set up the main parameters of training by modifying `scripts/finetune.sh`

```shell
#!/bin/bash
seed=200
encoding=swap_fc_plus
A=def
B=ctx
C=cls
D=hyps
permute=False
dir=encoding_results
python src/finetuning.py -s $seed -e $encoding -a $A $B $C $D -p $permute --path $dir
```
#### Hyperparameters

Extra training parameterers can be directly modified in `src/finetune.py`
```shell
# Hyperparameters
num_epochs = 5
batch_size_train = 16
batch_size_val = 8
num_warmup_steps = 0
```




### Run command
```shell
bash ./scripts/finetune.sh
```


## Performance evaluation
