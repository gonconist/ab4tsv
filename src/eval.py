import os
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer
from dataloader import WiCTSVDatasetEncodingOptions, WiCTSVDataset, WiCTSVDataLoader, read_wic_tsv
from dataloader import str2enum, str2bool
from ab4tsv import AB4TSV


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default='dev')
parser.add_argument("-a", "--analogy", nargs=4, metavar=('A', 'B', 'C', 'D'), default=('tgt', 'hyps', 'def', 'hyps'))
parser.add_argument("-e", "--encoding", type=str2enum, nargs='?', default=WiCTSVDatasetEncodingOptions.SWAP_FC)
parser.add_argument("-tfc", "--target_focus_char", type=str, default='$')
parser.add_argument("-hfc1", "--hypernyms_focus_char_1", type=str, default='$')
parser.add_argument("-hfc2", "--hypernyms_focus_char_2", type=str, default='$')
parser.add_argument("-pi", "--permutation_invariance", type=str2bool, nargs='?', default=True)
parser.add_argument("--save_preds", type=str2bool, nargs='?', default=True)
parser.add_argument("--out_binary_preds", type=str2bool, nargs='?', default=True)
parser.add_argument("--output_dir", type=str, default='results')
args = parser.parse_args()


# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using CPU instead.')
    device = torch.device("cpu")

if args.permutation_invariance:
    path = '{}/{}/ab4tsv/permutation_invariance/{}/'.format(args.output_dir, args.encoding.name, '_'.join(args.analogy))
else:
    path = '{}/{}/ab4tsv/no_permutation_invariance/{}/'.format(args.output_dir, args.encoding.name, '_'.join(args.analogy))

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

# Load development and test sets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
if args.dataset == 'dev':
    contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv(Path('data/Development'))
    dev_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions,
                           tokenizer=tokenizer,
                           target_focus_char=args.target_focus_char,
                           hypernyms_focus_char_1=args.hypernyms_focus_char_1,
                           hypernyms_focus_char_2=args.hypernyms_focus_char_2,
                           labels=labels,
                           encoding_type=args.encoding)
    dataloader = WiCTSVDataLoader(dev_ds, 'Development')
elif args.dataset == 'test':
    contexts, target_ses, hypernyms, definitions, labels = read_wic_tsv(Path('data/Test'))
    test_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions,
                           tokenizer=tokenizer,
                           target_focus_char=args.target_focus_char,
                           hypernyms_focus_char_1=args.hypernyms_focus_char_1,
                           hypernyms_focus_char_2=args.hypernyms_focus_char_2,
                           labels=labels,
                           encoding_type=args.encoding)
    dataloader = WiCTSVDataLoader(test_ds, 'Test')

# Load model
model = AB4TSV.from_pretrained(path, permutation_invariance=args.permutation_invariance)

if torch.cuda.is_available():
    model.cuda()


def evaluate_model(save_preds=False, out_binary_preds=False):
    """
    Evaluates the performance of the finetuned AB4TSV in terms of accuracy, recall, precision and f1-score.
    For test set submit your predictions in `codalab <https://competitions.codalab.org/competitions/23683>`__
    :param save_preds:  saves the predictions of the model into a text file.
    :param out_binary_preds:  transforms the predictions of the model into binary (T/F).
    """
    model.eval()
    if not args.permutation_invariance:
        filepath = path + 'predictions.txt'
        all_targets, all_outputs = [], []
        num_written_lines, num_actual_predict_examples = 0, 0
    else:
        AP = ['base', 'symmetry', 'central_permutation']
        filepath = [path + 'predictions_{}.txt'.format(p) for p in AP]
        all_targets, all_outputs = [], [[], [], []]
        num_written_lines, num_actual_predict_examples = [0, 0, 0], [0, 0, 0]
    print("***** Testing model *****")
    for batch_idx, features in enumerate(tqdm(dataloader)):
        input_ids = features["input_ids"].to(device)
        input_mask = features["attention_mask"].to(device)
        segment_ids = features["token_type_ids"].to(device)
        label_ids = None if args.dataset == 'test' else features["labels"].to(device)
        target_start_len = features["target_start_len"].to(device)
        descr_start_len = features["descr_start_len"].to(device)
        def_start_len = features["def_start_len"].to(device)
        hyps_start_len = features["hyps_start_len"].to(device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=label_ids,
                target_start_len=target_start_len,
                descr_start_len=descr_start_len,
                def_start_len=def_start_len,
                hyps_start_len=hyps_start_len,
                analogy=args.analogy)
            logits = out[-1]
        if not args.permutation_invariance:
            all_outputs.extend((logits > 0).float().flatten().cpu().detach().numpy().tolist())
        else:
            for i in range(len(all_outputs)):
                all_outputs[i].extend((logits[:, i] > 0).float().flatten().cpu().detach().numpy().tolist())
        all_targets.extend(label_ids.flatten().cpu().detach().numpy().tolist()) if label_ids is not None else None

        if save_preds:
            if not args.permutation_invariance:
                with open(filepath, 'a+') as writer:
                    num_actual_predict_examples += len(input_ids)
                    for ex_probabilities in logits.data.cpu().numpy():
                        output_line = "\t".join(str(class_probability) for class_probability in ex_probabilities) + "\n"
                        if out_binary_preds:
                            if float(output_line) > 0:
                                writer.write("T\n")
                            else:
                                writer.write("F\n")
                        else:
                            writer.write(output_line)
                        num_written_lines += 1
                assert num_written_lines == num_actual_predict_examples
            else:
                for i in range(len(all_outputs)):
                    with open(filepath[i], 'a+') as writer:
                        num_actual_predict_examples[i] += len(input_ids)
                        for ex_probabilities in logits[:, i].unsqueeze(1).data.cpu().numpy():
                            output_line = "\t".join(str(class_probability) for class_probability in ex_probabilities) + "\n"
                            if out_binary_preds:
                                if float(output_line) > 0:
                                    writer.write("T\n")
                                else:
                                    writer.write("F\n")
                            else:
                                writer.write(output_line)
                            num_written_lines[i] += 1
                    assert num_written_lines[i] == num_actual_predict_examples[i]

    if label_ids is not None:
        if not args.permutation_invariance:
            acc = accuracy_score(all_targets, all_outputs)
            pre, rec, f1, _ = precision_recall_fscore_support(all_targets, all_outputs, average='binary')
            print('====> val set Accuracy: {:.2%}'.format(acc))
            print('====> val set Recall: {:.2%}'.format(rec))
            print('====> val set Precesion: {:.2%}'.format(pre))
            print('====> val set F1-score: {:.2%}'.format(f1))
        else:
            for idx, perm_outputs in enumerate(all_outputs):
                acc = accuracy_score(all_targets, perm_outputs)
                pre, rec, f1, _ = precision_recall_fscore_support(all_targets, perm_outputs, average='binary')
                print('Permutation: {}-results\n'.format(AP[idx]))
                print('====> val set Accuracy: {:.2%}'.format(acc))
                print('====> val set Recall: {:.2%}'.format(rec))
                print('====> val set Precesion: {:.2%}'.format(pre))
                print('====> val set F1-score: {:.2%}\n'.format(f1))


def main():
   evaluate_model( save_preds=args.save_preds, out_binary_preds=args.out_binary_preds)

if __name__ == '__main__':

   main()
