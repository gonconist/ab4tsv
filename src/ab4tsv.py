import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel


class AB4TSV(BertPreTrainedModel):
    """
    BERT model + CNN + Linear
    This model uses BERT to compute a representation for [tgt], [ctx], [def], [hyps] and [desrc]
    which serve as input to the CNN-Analogy Classifier:
    ...
    1st layer (convolutional): 128 filters (= kernels) of size h × w = 1 × 2 with strides (1, 2)
    and relu activation
    2nd layer (convolutional): 64 filters of size (2, 2) with strides (2, 2) and relu activation
    3rd layer (dense, equivalent to linear for PyTorch): one output and sigmoid activation
    * sigmoid is implicitly applied within the BCEWithLogitsLoss for better numerical stability

    Conv2d(in_channels, out_channels, kernel_size, stride):
       needs 4 dim tensors as input (batch_size, channels, height, width)
   """
    def __init__(self, config, permutation_invariance=False):
        super().__init__(config)

        self.bert = BertModel(config)
        self.emb_size = config.hidden_size
        self.permutation_invariance = permutation_invariance
        self.conv1 = nn.Conv2d(1, 128, (1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, (2, 2), stride=(2, 2))
        self.linear = nn.Linear(64 * (self.emb_size // 2), 1)

        self.init_weights()

    def flatten(self, t):
        t = t.reshape(t.size()[0], -1)
        return t

    def is_analogy(self, a, b, c, d):

        a = torch.unsqueeze(a, 1)  # (bs, 1, dim)
        b = torch.unsqueeze(b, 1)  # (bs, 1, dim)
        c = torch.unsqueeze(c, 1)  # (bs, 1, dim)
        d = torch.unsqueeze(d, 1)  # (bs, 1, dim)
        image = torch.stack([a, b, c, d], dim=3)
        x = self.conv1(image)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def forward(
            self,
            input_ids=None,
            target_start_len=None,
            descr_start_len=None,
            def_start_len=None,
            hyps_start_len=None,
            analogy=None,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            *args,
            **kwargs
    ):
        tgt_inds, tgt_embeds = [], []
        ctx_ids, ctx_embeds = [], []
        def_inds, def_embeds = [], []
        hyps_inds, hyps_embeds = [], []
        descr_inds, descr_embeds = [], []
        for row in target_start_len.split(1):
            row_inds = range(row[0, 0], row.sum())
            tgt_inds.append(list(row_inds))
        for row in descr_start_len.split(1):
            row_inds = range(row[0, 0], row.sum())
            descr_inds.append(list(row_inds))
            ctx_ids.append(list(range(1, row[0, 0] - 1)))
        for row in def_start_len.split(1):
            row_inds = range(row[0, 0], row.sum())
            def_inds.append(list(row_inds))
        for row in hyps_start_len.split(1):
            row_inds = range(row[0, 0], row.sum())
            hyps_inds.append(list(row_inds))

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = bert_output[0]       # (bs, seq_len, dim)

        for i, seq_out in enumerate(hidden_state.split(1, dim=0)):
            seq_out = seq_out.squeeze()
            row_tgt_embeds = seq_out[tgt_inds[i]]
            row_tgt_mean_embeds = torch.mean(row_tgt_embeds, dim=0).squeeze()       # (1, dim)
            row_descr_embeds = seq_out[descr_inds[i]]
            row_descr_mean_embeds = torch.mean(row_descr_embeds, dim=0).squeeze()   # (1, dim)
            row_ctx_embeds = seq_out[ctx_ids[i]]
            row_ctx_mean_embeds = torch.mean(row_ctx_embeds, dim=0).squeeze()       # (1, dim)
            row_def_embeds = seq_out[def_inds[i]]
            row_def_mean_embeds = torch.mean(row_def_embeds, dim=0).squeeze()       # (1, dim)
            row_hyps_embeds = seq_out[hyps_inds[i]]
            row_hyps_mean_embeds = torch.mean(row_hyps_embeds, dim=0).squeeze()     # (1, dim)
            tgt_embeds.append(row_tgt_mean_embeds)
            descr_embeds.append(row_descr_mean_embeds)
            ctx_embeds.append(row_ctx_mean_embeds)
            def_embeds.append(row_def_mean_embeds)
            hyps_embeds.append(row_hyps_mean_embeds)

        cls_output = bert_output[1]                 # (bs, dim)
        tgt_output = torch.stack(tgt_embeds)        # (bs, dim)
        ctx_output = torch.stack(ctx_embeds)        # (bs, dim)
        def_output = torch.stack(def_embeds)        # (bs, dim)
        hyps_output = torch.stack(hyps_embeds)      # (bs, dim)
        descr_output = torch.stack(descr_embeds)    # (bs, dim)

        (A, B, C, D) = analogy
        obj = {'cls': cls_output,
                'tgt': tgt_output,
                'ctx': ctx_output,
                'def': def_output,
                'hyps': hyps_output,
                'descr': descr_output}

        if self.permutation_invariance:
            # assert not labels is None
            loss = torch.tensor(0, device=input_ids.device).float()
            loss_fct = BCEWithLogitsLoss()
            X = torch.empty(size=(input_ids.shape[0], 3), device=input_ids.device)
            pstl_to_emb = {'base': (obj[A], obj[B], obj[C], obj[D]),
                           'sym': (obj[C], obj[D], obj[A], obj[B]),
                           'cp': (obj[A], obj[C], obj[B], obj[D])}
            pstl_to_idx = {key: idx for idx, key in enumerate(pstl_to_emb.keys())}
            pstl_keys = list(pstl_to_emb.keys())
            random.shuffle(pstl_keys)
            for key in pstl_keys:
                (a, b, c, d) = pstl_to_emb[key]
                x = self.is_analogy(a, b, c, d)
                X[:, pstl_to_idx[key]] = x.squeeze()
                outputs = (X,)
                if labels is not None:
                    loss += loss_fct(x, labels.unsqueeze(1))
                    assert not torch.isnan(loss).any()
                    outputs = (loss,) + (X,)
        else:
            x = self.is_analogy(obj[A], obj[B], obj[C], obj[D])
            outputs = (x,)
            if labels is not None:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(x, labels.unsqueeze(1))
                assert not torch.isnan(loss).any()
                outputs = (loss,) + outputs

        return outputs  # (loss) + logits