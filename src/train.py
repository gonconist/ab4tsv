import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from dataloader import WiCTSVDatasetEncodingOptions, WiCTSVDataset, WiCTSVDataLoader, read_wic_tsv
from dataloader import str2enum, str2bool
from ab4tsv import AB4TSV


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size_train", type=int, default=16)
parser.add_argument("--batch_size_val", type=int, default=8)
parser.add_argument("-a", "--analogy", nargs=4, metavar=('A', 'B', 'C', 'D'), default=('def', 'ctx', 'cls', 'hyps'))
parser.add_argument("-e", "--encoding", type=str2enum, nargs='?', default=WiCTSVDatasetEncodingOptions.SWAP_FC)
parser.add_argument("-tfc", "--target_focus_char", type=str, default='$')
parser.add_argument("-hfc1", "--hypernyms_focus_char_1", type=str, default='$')
parser.add_argument("-hfc2", "--hypernyms_focus_char_2", type=str, default='$')
parser.add_argument("-pi", "--permutation_invariance", type=str2bool, nargs='?', default=False)
parser.add_argument("--output_dir", type=str, default='results')
args = parser.parse_args()


# Set the seed value all over the place to make this reproducible
seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using CPU instead.')
    device = torch.device("cpu")

if args.permutation_invariance:
    path = '{}/{}/ab4tsv/permutation/{}/'.format(args.output_dir, args.encoding.name, '_'.join(args.analogy))
else:
    path = '{}/{}/ab4tsv/no_permutation/{}/'.format(args.output_dir, args.encoding.name, '_'.join(args.analogy))

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

# Load training and development sets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv(Path('data/Training'))
train_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions,
                         tokenizer=tokenizer,
                         target_focus_char=args.target_focus_char,
                         hypernyms_focus_char_1=args.hypernyms_focus_char_1,
                         hypernyms_focus_char_2=args.hypernyms_focus_char_2,
                         labels=labels,
                         encoding_type=args.encoding)
contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv(Path('data/Development'))
dev_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions,
                       tokenizer=tokenizer,
                       target_focus_char=args.target_focus_char,
                       hypernyms_focus_char_1=args.hypernyms_focus_char_1,
                       hypernyms_focus_char_2=args.hypernyms_focus_char_2,
                       labels=labels,
                       encoding_type=args.encoding)
train_dataloader = WiCTSVDataLoader(train_ds, 'Training', batch_size=args.batch_size_train)
dev_dataloader = WiCTSVDataLoader(dev_ds, 'Development', batch_size=args.batch_size_val)

# Instantiate model
model = AB4TSV.from_pretrained('bert-base-uncased', permutation_invariance=args.permutation_invariance)
if torch.cuda.is_available():
    model.cuda()
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-6)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * args.num_epochs)

# Save training params into .txt file
with open(path + 'output.txt', 'wt') as f:
    f.write("seed:\t {seed}\n"
            "num_epochs:\t {n_epochs}\n"
            "train_batch_size:\t {batch_size_train}\n"
            "dev_batch_size:\t {batch_size_val}\n"
            "optimizer:\t {optimizer}\n"
            "analogy:\t A : B :: C : D  = {params}\n"
            "encoding:\t {enc}\n"
            "target_focus_char:\t {tfc}\n"
            "hypernyms_starting_focus_char:\t {hfc1}\n"
            "hypernyms_ending_focus_char:\t {hfc2}\n"
            "permutation_invariance:\t {pi}\n\n".format(seed=seed_val,
                                                        n_epochs=args.num_epochs,
                                                        batch_size_train=args.batch_size_train,
                                                        batch_size_val=args.batch_size_val,
                                                        optimizer=optimizer.defaults,
                                                        params=args.analogy,
                                                        enc=args.encoding.name,
                                                        tfc=args.target_focus_char,
                                                        hfc1=args.hypernyms_focus_char_1,
                                                        hfc2=args.hypernyms_focus_char_2,
                                                        pi=args.permutation_invariance))
avg_train_loss, avg_val_loss = [], []


def train(epoch, global_step, best_acc, f1, best_epoch, step):
    train_loss = 0
    for batch_idx, features in enumerate(tqdm(train_dataloader)):
        input_ids = features["input_ids"].to(device)
        input_mask = features["attention_mask"].to(device)
        segment_ids = features["token_type_ids"].to(device)
        label_ids = features["labels"].to(device)
        target_start_len = features["target_start_len"].to(device)
        descr_start_len = features["descr_start_len"].to(device)
        def_start_len = features["def_start_len"].to(device)
        hyps_start_len = features["hyps_start_len"].to(device)
        model.zero_grad()
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            labels=label_ids,
            target_start_len=target_start_len,
            descr_start_len=descr_start_len,
            def_start_len=def_start_len,
            hyps_start_len=hyps_start_len,
            analogy=args.analogy)
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        if batch_idx % 10 == 0:
            print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, args.num_epochs,
                (batch_idx + 1) * len(label_ids), len(train_dataloader.dataset),
                100. * (batch_idx+1) / len(train_dataloader),
                loss.item() / len(label_ids)))
        if global_step % 10 == 0:
            metrics = validate()
            if metrics['acc'] > best_acc:
                best_acc = metrics['acc']
                f1 = metrics['f1']
                step = global_step
                print('====> Current Top Val set Accuracy: {:.2%}'.format(best_acc))
                print('====> Current Val set F1-score: {:.2%}'.format(f1))
                best_epoch = epoch
                # Save model to output dir
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(path)
    avg_train_loss.append(train_loss / len(train_dataloader))
    print('====> Epoch: [{}/{}]\tAverage training loss: {:.6f}'.format(epoch, args.num_epochs, avg_train_loss[-1]))
    return global_step, best_acc, f1, best_epoch, step

def validate():
    val_loss = 0
    all_targets, all_outputs = [], []
    model.eval()
    print("***** Evaluating model *****")
    for batch_idx, features in enumerate(tqdm(dev_dataloader)):
        input_ids = features["input_ids"].to(device)
        input_mask = features["attention_mask"].to(device)
        segment_ids = features["token_type_ids"].to(device)
        label_ids = features["labels"].to(device)
        target_start_len = features["target_start_len"].to(device)
        descr_start_len = features["descr_start_len"].to(device)
        def_start_len = features["def_start_len"].to(device)
        hyps_start_len = features["hyps_start_len"].to(device)
        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=label_ids,
                target_start_len=target_start_len,
                descr_start_len=descr_start_len,
                def_start_len=def_start_len,
                hyps_start_len=hyps_start_len,
                analogy=args.analogy)
        if args.permutation_invariance: logits = logits[:, 0].unsqueeze(-1).to(device)
        val_loss += loss.item()
        all_targets.extend(label_ids.flatten().cpu().detach().numpy().tolist())
        all_outputs.extend((logits > 0).float().flatten().cpu().detach().numpy().tolist())
    avg_val_loss.append(val_loss / len(dev_dataloader))
    print('====> Val set loss: {:.6f}'.format(avg_val_loss[-1]))
    acc = accuracy_score(all_targets, all_outputs)
    pre, rec, f1, _ = precision_recall_fscore_support(all_targets, all_outputs, average='binary')
    print('====> Val set Accuracy: {:.2%}'.format(acc))
    print('====> Val set Recall: {:.2%}'.format(rec))
    print('====> Val set Precesion: {:.2%}'.format(pre))
    print('====> Val set F1-score: {:.2%}'.format(f1))
    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1}

def main():
    global_step = 0
    best_acc, f1, best_epoch, step = 0., 0., 1, 0
    model.train()
    for epoch in range(1, args.num_epochs + 1):
        global_step, best_acc, f1, best_epoch, step = train(epoch, global_step, best_acc, f1, best_epoch, step)
        print("***** Evaluating model at the end of epoch {} *****".format(epoch))
        validate()
    with open(path + 'output.txt'.format(args.seed), 'a') as file:
        file.write("Top val set accuracy:\t {}\n"
                   "F1-score:\t {}\n"
                   "Training finished at epoch:\t {}\n"
                   "Total training global steps:\t {}".format(best_acc, f1, best_epoch, step))

if __name__ == '__main__':

   main()
