import csv
import argparse
from enum import Enum
from pathlib import Path
from collections import defaultdict
import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


class WiCTSVDatasetEncodingOptions(Enum):
    """
    Enum is a class in python for creating enumerations, which are a set of symbolic names (members) bound to unique,
    constant values. The members of an enumeration can be compared by these symbolic names, and the enumeration itself
    can be iterated over. An enum has the following characteristics.
    The enums are evaluatable string representation of an object also called repr().
    The name of the enum is displayed using ‘name’ keyword.
    Using type() we can check the enum types.
    The __members__ attribute is an ordered mapping of the names of the enums to their respective enum objects.
    """
    DEFAULT = '[CLS] context [SEP] definition; hypernyms [SEP]'
    DEFAULT_FC = '[CLS] context [SEP] definition; $ hypernyms $ [SEP]'
    DEFAULT_EM = '[CLS] context [SEP] definition; [H] hypernyms [\H] [SEP]'
    SWAP = '[CLS] context [SEP] hypenyms; definition [SEP]'
    SWAP_FC = '[CLS] context [SEP] $ hypernyms $; definition [SEP]'
    SWAP_EM = '[CLS] context [SEP] [H] hypernyms [\H]; definition [SEP]'

def float_or_None(value):
    try:
        return float(value)
    except:
        return None

def str2enum(v):
    if isinstance(v, WiCTSVDatasetEncodingOptions):
        return v
    if v.lower() == 'default':
        return WiCTSVDatasetEncodingOptions.DEFAULT
    elif v.lower() == 'default_fc':
        return WiCTSVDatasetEncodingOptions.DEFAULT_FC
    elif v.lower() == 'default_em':
        return WiCTSVDatasetEncodingOptions.DEFAULT_EM
    elif v.lower() == 'swap':
        return WiCTSVDatasetEncodingOptions.SWAP
    elif v.lower() == 'swap_fc':
        return WiCTSVDatasetEncodingOptions.SWAP_FC
    elif v.lower() == 'swap_em':
        return WiCTSVDatasetEncodingOptions.SWAP_EM
    else:
        raise argparse.ArgumentTypeError('Wrong value.')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class WiCTSVDataset(torch.utils.data.Dataset):
    def __init__(self,
                 contexts,
                 target_inds,
                 hypernyms,
                 definitions,
                 tokenizer: PreTrainedTokenizer,
                 labels=None,
                 target_focus_char=None,
                 hypernyms_focus_char_1=None,
                 hypernyms_focus_char_2=None,
                 encoding_type=None):
        self.len = len(contexts)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float)
        else:
            self.labels = None
        self.tokenizer = tokenizer
        if target_focus_char is not None:
            contexts, target_inds = self.mark_target_in_context(contexts=contexts,
                                                                target_inds=target_inds,
                                                                focus_char=target_focus_char)
        hypernyms = [', '.join(hyps) for hyps in hypernyms]
        if encoding_type.name not in ['DEFAULT', 'SWAP']:
            if (hypernyms_focus_char_1 and hypernyms_focus_char_2) is not None:
                hypernyms = self.mark_start_end_of_hyperyms(hypernyms=hypernyms,
                                                            focus_char_1=hypernyms_focus_char_1,
                                                            focus_char_2=hypernyms_focus_char_2)

        targets = [cxt.split(' ')[tgt_ind] for cxt, tgt_ind in zip(contexts, target_inds)]
        self.tgt_start_len = []
        self.descr_start_len = []
        sense_ids_strs = []
        self.def_start_len = []
        self.hyps_start_len = []
        for ctx, tgt_ind, def_, hyps, tgt in zip(contexts, target_inds, definitions, hypernyms, targets):
            ctx_index_map, ctx_index_list = self._get_token_index_map_and_list(ctx, tokenizer)
            def_index_map, def_index_list = self._get_token_index_map_and_list(def_, tokenizer)
            hyps_index_map, hyps_index_list = self._get_token_index_map_and_list(hyps, tokenizer)

            target_start_ind = ctx_index_map[tgt_ind][0] + len(['[CLS]'])
            target_len = len(ctx_index_map[tgt_ind])
            self.tgt_start_len.append((target_start_ind, target_len))

            if encoding_type.name == 'DEFAULT':
                sense_identifiers_str = def_ + '; ' + hyps
                sense_ids_strs.append(sense_identifiers_str)

                def_start_ind = len(ctx_index_list) + len(['[CLS]', '[SEP]'])
                def_len = len(def_index_list)
                self.def_start_len.append((def_start_ind, def_len))

                hyps_start_ind = def_start_ind + def_len + len([';'])
                hyps_len = len(hyps_index_list)
                self.hyps_start_len.append((hyps_start_ind, hyps_len))

            elif encoding_type.name == 'DEFAULT_FC' or encoding_type.name == 'DEFAULT_EM':
                sense_identifiers_str = def_ + '; ' + hyps
                sense_ids_strs.append(sense_identifiers_str)
                def_start_ind = len(ctx_index_list) + len(['[CLS]', '[SEP]'])
                def_len = len(def_index_list)
                self.def_start_len.append((def_start_ind, def_len))

                hyps_start_ind = def_start_ind + def_len + len([';'])
                hyps_len = len(hyps_index_list)
                self.hyps_start_len.append((hyps_start_ind, hyps_len))

            elif encoding_type.name == 'SWAP':
                sense_identifiers_str = hyps + '; ' + def_
                sense_ids_strs.append(sense_identifiers_str)

                hyps_start_ind = len(ctx_index_list) + len(['[CLS]', '[SEP]'])
                hyps_len = len(hyps_index_list)
                self.hyps_start_len.append((hyps_start_ind, hyps_len))

                def_start_ind = hyps_start_ind + hyps_len + len([';'])
                def_len = len(def_index_list)
                self.def_start_len.append((def_start_ind, def_len))

            elif encoding_type.name == 'SWAP_FC' or encoding_type.name == 'SWAP_EM':
                sense_identifiers_str = hyps + '; ' + def_
                sense_ids_strs.append(sense_identifiers_str)

                hyps_start_ind = len(ctx_index_list) + len(['[CLS]', '[SEP]'])
                hyps_len = len(hyps_index_list)
                self.hyps_start_len.append((hyps_start_ind, hyps_len))

                def_start_ind = len(ctx_index_list) + len(hyps_index_list) + len(['[CLS]', '[SEP]', ';'])
                def_len = len(def_index_list)
                self.def_start_len.append((def_start_ind, def_len))

            else:
                raise NotImplementedError

            descrs_start_ind = len(ctx_index_list) + len(['[CLS]', '[SEP]'])
            descrs_len = def_len + len(hyps_index_list) + len([';'])
            self.descr_start_len.append((descrs_start_ind, descrs_len))

        tokenizer_input = [[context, sense_ids] for context, sense_ids in zip(contexts, sense_ids_strs)]
        self.encodings = tokenizer(tokenizer_input, return_tensors='pt', truncation=True, padding=True)

    @staticmethod
    def _get_token_index_map_and_list(text, tokenizer):
        """
        creates a mapping between indices of original tokens and indices of tokens after bert tokenization
        :param text: text to be tokenized
        :return: dict of the format { original_index : [bert_indices]}, list original token index for each bert token
        """
        original_tokens = text.split(' ')
        index_list = []
        index_map = defaultdict(list)

        for original_index in range(len(original_tokens)):
            bert_tokens = tokenizer.tokenize(original_tokens[original_index])
            index_list += [original_index] * len(bert_tokens)
        for bert_index, original_index in enumerate(index_list):
            index_map[original_index].append(bert_index)

        bert_tokens = tokenizer.tokenize(text)
        assert len(bert_tokens) == len(sum(index_map.values(), [])), (bert_tokens, index_map)

        return index_map, index_list

    @staticmethod
    def mark_target_in_context(contexts: list, target_inds: list, focus_char: str = "$"):
        """
        This method will mark the target word in a context with a special character before and after it,
        e.g. "This is the target in this sentence" --> "This is the $ target $ in this sentence"
        :param contexts: list of context strings
        :param target_inds: list of target indices
        :param focus_char: character which should be taken to mark the target
        :return: list of marked context strings, updated target indices
        """
        marked_contexts = []
        marked_target_inds = []
        for context, target_i in zip(contexts, target_inds):
            context_tokens = context.split()
            before_target = context_tokens[:target_i]
            after_target = context_tokens[target_i + 1:]
            marked_contexts.append(' '.join(before_target +
                                            [focus_char] +
                                            context_tokens[target_i:target_i + 1] +
                                            [focus_char] +
                                            after_target))
            marked_target_inds.append(target_i + 1)
        assert len(marked_contexts) == len(marked_target_inds)

        return marked_contexts, marked_target_inds

    @staticmethod
    def mark_start_end_of_hyperyms(hypernyms: list, focus_char_1: str = "$", focus_char_2: str = "$"):
        """
        This method will mark the them with special starting and ending focus characters (before and after it),
        e.g., "conflict, struggle, battle" becomes "$ conflict, struggle, battle $"
        :param hypernyms: list of context strings
        :param focus_char_1: character which marks the beginning of the hypernyms
        :param focus_char_2: character which marks the end of the hypernyms
        :return: list of marked hypernyms
        """
        marked_hypernyms = []
        for hyps in hypernyms:
            marked_hypernyms.append(' '.join([focus_char_1,
                                              hyps,
                                              focus_char_2]))
        assert len(marked_hypernyms) == len(hypernyms)

        return marked_hypernyms

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['target_start_len'] = torch.tensor(self.tgt_start_len[idx])
        item['def_start_len'] = torch.tensor(self.def_start_len[idx])
        item['hyps_start_len'] = torch.tensor(self.hyps_start_len[idx])
        item['descr_start_len'] = torch.tensor(self.descr_start_len[idx])

        if self.labels is not None:
            item['labels'] = self.labels[idx]

        return item

    def __len__(self):
        return self.len

def WiCTSVDataLoader(ds, mode, batch_size=8, num_workers=0):

    if mode in ['Training', 'Development', 'Test']:
        sampler = RandomSampler(ds) if mode == 'Training' else SequentialSampler(ds)
        data_params = {'batch_size': batch_size,
                       'sampler': sampler,
                       'num_workers': num_workers,
                       }
        return DataLoader(ds, **data_params)
    else:
        raise ValueError

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def read_wic_tsv(wic_tsv_folder: Path,
                 tgt_column=0,
                 tgt_ind_column=1,
                 cxt_column=2):
    targets = []
    contexts = []
    target_inds = []
    examples_path = next(wic_tsv_folder.glob('*_examples.txt'))
    for line in _read_tsv(examples_path):
        target_inds.append(int(line[tgt_ind_column].strip()))
        contexts.append(line[cxt_column].strip())
        targets.append(line[tgt_column].strip())

    hypernyms = []
    hypernyms_path = next(wic_tsv_folder.glob('*_hypernyms.txt'))

    for line in _read_tsv(hypernyms_path):
        hypernyms.append([hypernym.replace('_', ' ').strip() for hypernym in line])

    defs_path = next(wic_tsv_folder.glob('*_definitions.txt'))
    definitions = [definition[0] for definition in _read_tsv(defs_path)]

    try:
        labels_path = next(wic_tsv_folder.glob('*_labels.txt'))
        labels = [int(x[0].strip() == 'T') for x in _read_tsv(labels_path)]
    except Exception as e:
        print(e)
        labels = None
    assert len(contexts) == len(hypernyms) == len(definitions), (len(contexts), len(hypernyms), len(definitions))
    for cxt, t_ind, tgt in zip(contexts, target_inds, targets):
        if not cxt.split(' ')[t_ind].lower().startswith(tgt[:-1].lower()):
            assert False, (tgt.lower(), t_ind, cxt.split(' '), cxt.split(' ')[t_ind].lower())
    return contexts, target_inds, hypernyms, definitions, labels
