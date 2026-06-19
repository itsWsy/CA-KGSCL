"Knowledge-Guided Semantically Consistent Contrastive Learning for Sequential Recommendation"

import copy
import logging
import random
import time
import numpy as np
import torch.nn.functional as F
from easydict import EasyDict
from src.model.abstract_recommeder import AbstractRecommender
import argparse
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_

from src.model.loss import (InfoNCELoss, multi_positive_view_target_loss, compute_position_importance,
                            compute_kg_reliability_for_sequence, compute_adaptive_position_scores,
                            select_adaptive_positions)
from src.model.sequential_encoder import Transformer
from src.utils.utils import HyperParamDict


class KGSCL(AbstractRecommender):
    def __init__(self, num_items, config, kg_map):
        super(KGSCL, self).__init__(num_items, config)
        self.embed_size = config.embed_size
        self.tem1 = config.tem1
        self.tem2 = config.tem2
        self.lamda1 = config.lamda1
        self.lamda2 = config.lamda2
        self.use_mp_vt = config.use_mp_vt
        self.mp_vt_top_m = config.mp_vt_top_m
        self.mp_vt_tau = config.mp_vt_tau if config.mp_vt_tau is not None else self.tem2
        self.insert_ratio = config.insert_ratio
        self.substitute_ratio = config.substitute_ratio
        self.use_adaptive_position_aug = config.use_adaptive_position_aug
        self.position_select_mode = config.position_select_mode
        self.position_temperature = config.position_temperature
        self.position_importance_sim = config.position_importance_sim
        self.kg_reliability_agg = config.kg_reliability_agg
        self.adaptive_position_warmup_epochs = config.adaptive_position_warmup_epochs
        self.adaptive_position_fallback = config.adaptive_position_fallback
        self.adaptive_substitute_position = config.adaptive_substitute_position
        self.adaptive_insert_position = config.adaptive_insert_position
        self.use_adaptive_profile = config.use_adaptive_profile
        self.adaptive_profile_batches = config.adaptive_profile_batches
        self.kg_relation_dict = kg_map.get('kg_relation_dict', {}) if isinstance(kg_map, dict) else {}
        self.co_occurrence_dict = kg_map.get('co_occurrence_dict', {}) if isinstance(kg_map, dict) else {}
        self._loss_logged_epochs = set()
        self._position_logged_epochs = set()
        self._profile_batch_count = 0
        self._profile_time = {}

        self.initializer_range = config.initializer_range
        self.item_embedding = nn.Embedding(self.num_items, self.embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.embed_size)
        self.input_layer_norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.input_dropout = nn.Dropout(config.hidden_dropout)
        self.trm_encoder = Transformer(embed_size=self.embed_size,
                                       ffn_hidden=config.ffn_hidden,
                                       num_blocks=config.num_blocks,
                                       num_heads=config.num_heads,
                                       attn_dropout=config.attn_dropout,
                                       hidden_dropout=config.hidden_dropout,
                                       layer_norm_eps=config.layer_norm_eps)
        self.nce_loss = InfoNCELoss(temperature=self.tem1,
                                    similarity_type=config.sim)
        self.cross_entropy = nn.CrossEntropyLoss()
        if self.use_adaptive_position_aug:
            start_time = time.perf_counter()
            self.register_buffer('sub_reliability_table', self._build_reliability_table('s'), persistent=False)
            self.register_buffer('ins_reliability_table', self._build_reliability_table('c'), persistent=False)
            logging.info(f'Built reliability tables once: shape={self.sub_reliability_table.size()}, '
                         f'time={time.perf_counter() - start_time:.4f}s')
        else:
            self.register_buffer('sub_reliability_table', torch.zeros(self.num_items, dtype=torch.float),
                                 persistent=False)
            self.register_buffer('ins_reliability_table', torch.zeros(self.num_items, dtype=torch.float),
                                 persistent=False)

        self.apply(self._init_weights)
        logging.info(f'use_mp_vt = {self.use_mp_vt}')
        logging.info(f'mp_vt_top_m = {self.mp_vt_top_m}')
        logging.info(f'mp_vt_tau = {self.mp_vt_tau}')
        logging.info(f'use_adaptive_position_aug = {self.use_adaptive_position_aug}')
        logging.info(f'position_select_mode = {self.position_select_mode}')
        logging.info(f'position_temperature = {self.position_temperature}')
        logging.info(f'position_importance_sim = {self.position_importance_sim}')
        logging.info(f'kg_reliability_agg = {self.kg_reliability_agg}')
        logging.info(f'adaptive_position_warmup_epochs = {self.adaptive_position_warmup_epochs}')
        logging.info(f'adaptive_position_fallback = {self.adaptive_position_fallback}')
        logging.info(f'use_adaptive_profile = {self.use_adaptive_profile}')

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def train_forward(self, item_seq, seq_len, target, *args, **kwargs):
        """
        Args:
            item_seq: [batch, max_len]
            seq_len: [batch]
            target: [batch]
        """
        v2v_aug_seq, v2v_aug_len = args[0], args[1]
        v2t_aug_seq, v2t_aug_len = args[2], args[3]
        aug_target, pos_item_set = args[4], args[5]
        epoch = kwargs.get('epoch', None)
        use_adaptive = (self.use_adaptive_position_aug and epoch is not None
                        and epoch >= self.adaptive_position_warmup_epochs)

        # original seq encoding
        if use_adaptive:
            self._profile_start('original_sequence_encoding')
            seq_hidden, seq_embedding = self.preference_encoding(item_seq, seq_len, return_all_hidden=True)
            self._profile_stop('original_sequence_encoding')
            v2v_aug_seq, v2v_aug_len, v2t_aug_seq, v2t_aug_len, position_stats = \
                self.build_adaptive_augmented_views(item_seq, seq_len, seq_hidden.detach())
        else:
            seq_embedding = self.preference_encoding(item_seq, seq_len)  # [B, D]
            position_stats = None
        # v2v aug seq encoding
        self._profile_start('augmented_views_encoding')
        v2v_seq_embedding = self.preference_encoding(v2v_aug_seq, v2v_aug_len)  # [B, D]
        # v2t aug seq encoding
        v2t_seq_embedding = self.preference_encoding(v2t_aug_seq, v2t_aug_len)  # [B, D]
        self._profile_stop('augmented_views_encoding')

        # rec loss
        logits = seq_embedding @ self.item_embedding.weight.t()
        rec_loss = self.cross_entropy(logits, target)
        # view-view (v2v) CL loss
        v2v_loss = self.nce_loss(seq_embedding, v2t_seq_embedding)
        # view-target (v2t) CL loss
        if self.use_mp_vt:
            mp_vt_pos_set, mp_vt_pos_size, mp_vt_no_sub = args[6], args[7], args[8]
            pos_sets = []
            for row, size in zip(mp_vt_pos_set.detach().cpu().tolist(), mp_vt_pos_size.detach().cpu().tolist()):
                pos_sets.append(row[:int(size)])
            v2t_loss = multi_positive_view_target_loss(
                h_view=v2v_seq_embedding,
                pos_sets=pos_sets,
                item_embedding=self.item_embedding.weight,
                tau=self.mp_vt_tau
            )
        else:
            v2t_logits = v2v_seq_embedding @ self.item_embedding.weight.t()
            # mask positive items
            v2t_logits = torch.scatter(v2t_logits, 1, pos_item_set, float('-inf'))
            v2t_loss = self.cross_entropy(v2t_logits / self.tem2, aug_target)

        if epoch is not None and epoch not in self._loss_logged_epochs:
            self._loss_logged_epochs.add(epoch)
            position_mode = 'adaptive' if use_adaptive else 'random'
            if self.use_mp_vt:
                avg_pos_size = mp_vt_pos_size.float().mean().item()
                no_sub_ratio = mp_vt_no_sub.float().mean().item()
                logging.info(f'MP-VT epoch {epoch}: position_mode={position_mode}, '
                             f'avg_positive_set_size={avg_pos_size:.4f}, '
                             f'no_substitute_ratio={no_sub_ratio:.4f}, '
                             f'loss_rec={rec_loss.item():.4f}, loss_v2v={v2v_loss.item():.4f}, '
                             f'loss_v2t={v2t_loss.item():.4f}')
            else:
                logging.info(f'KGSCL epoch {epoch}: position_mode={position_mode}, loss_rec={rec_loss.item():.4f}, '
                             f'loss_v2v={v2v_loss.item():.4f}, loss_v2t={v2t_loss.item():.4f}')
            if position_stats is not None:
                logging.info(
                    f'PAKA epoch {epoch}: avg_sub_selected_importance={position_stats["sub_importance"]:.4f}, '
                    f'avg_sub_selected_reliability={position_stats["sub_reliability"]:.4f}, '
                    f'avg_sub_selected_score={position_stats["sub_score"]:.4f}, '
                    f'avg_ins_selected_importance={position_stats["ins_importance"]:.4f}, '
                    f'avg_ins_selected_reliability={position_stats["ins_reliability"]:.4f}, '
                    f'avg_ins_selected_score={position_stats["ins_score"]:.4f}, '
                    f'sub_fallback_ratio={position_stats["sub_fallback_ratio"]:.4f}, '
                    f'ins_fallback_ratio={position_stats["ins_fallback_ratio"]:.4f}, '
                    f'avg_sub_position_from_end={position_stats["sub_from_end"]:.4f}, '
                    f'avg_ins_position_from_end={position_stats["ins_from_end"]:.4f}'
                )

        if use_adaptive:
            self._profile_report()
        return rec_loss + self.lamda1 * v2v_loss + self.lamda2 * v2t_loss

    def _build_reliability_table(self, relation_type):
        """PAKA: precompute item-level KG reliability from relation and co-occurrence data."""
        table = torch.zeros(self.num_items, dtype=torch.float)
        for item_id in range(1, self.num_items):
            raw_item = item_id - 1
            relation_info = self.kg_relation_dict.get(raw_item, {})
            neighbors = relation_info.get(relation_type, []) if isinstance(relation_info, dict) else []
            if len(neighbors) == 0:
                continue
            score_info = self.co_occurrence_dict.get(raw_item, {})
            aligned_scores = score_info.get(relation_type, None) if isinstance(score_info, dict) else None
            scores = []
            for idx, neighbor in enumerate(neighbors):
                if int(neighbor) == raw_item:
                    continue
                if aligned_scores is not None and idx < len(aligned_scores):
                    scores.append(float(aligned_scores[idx]))
                else:
                    scores.append(0.)
            if len(scores) == 0:
                continue
            if self.kg_reliability_agg == 'mean':
                table[item_id] = float(sum(scores) / len(scores))
            else:
                table[item_id] = float(max(scores))
        return table

    def build_adaptive_augmented_views(self, item_seq, seq_len, seq_hidden):
        """PAKA: build two KG augmented views using adaptive position selection."""
        with torch.no_grad():
            seq_mask = item_seq.ne(0)
            self._profile_start('position_importance')
            importance, _ = compute_position_importance(
                seq_hidden,
                seq_mask,
                similarity=self.position_importance_sim
            )
            self._profile_stop('position_importance')
            self._profile_start('sub_reliability')
            sub_reliability = compute_kg_reliability_for_sequence(
                item_seq, seq_mask, self.sub_reliability_table, agg=self.kg_reliability_agg, pad_id=0
            )
            self._profile_stop('sub_reliability')
            self._profile_start('ins_reliability')
            ins_reliability = compute_kg_reliability_for_sequence(
                item_seq, seq_mask, self.ins_reliability_table, agg=self.kg_reliability_agg, pad_id=0
            )
            self._profile_stop('ins_reliability')
            self._profile_start('position_scores')
            sub_scores = compute_adaptive_position_scores(importance, sub_reliability, seq_mask, mode='substitute')
            ins_scores = compute_adaptive_position_scores(importance, ins_reliability, seq_mask, mode='insert')
            self._profile_stop('position_scores')

            self._profile_start('sub_position_selection')
            sub_positions, sub_fallback, sub_selected_mask = select_adaptive_positions(
                sub_scores, seq_mask, self.substitute_ratio, mode=self.position_select_mode,
                temperature=self.position_temperature, fallback=self.adaptive_position_fallback, return_mask=True
            )
            self._profile_stop('sub_position_selection')
            self._profile_start('ins_position_selection')
            ins_positions, ins_fallback, ins_selected_mask = select_adaptive_positions(
                ins_scores, seq_mask, self.insert_ratio, mode=self.position_select_mode,
                temperature=self.position_temperature, fallback=self.adaptive_position_fallback, return_mask=True
            )
            self._profile_stop('ins_position_selection')

            self._profile_start('adaptive_sequence_construction')
            view_1, len_1, types_1 = self._build_one_adaptive_view(item_seq, sub_positions, ins_positions)
            view_2, len_2, types_2 = self._build_one_adaptive_view(item_seq, sub_positions, ins_positions)
            self._profile_stop('adaptive_sequence_construction')
            stats = self._summarize_position_stats(
                importance, sub_reliability, ins_reliability, sub_scores, ins_scores,
                sub_selected_mask, ins_selected_mask, sub_fallback, ins_fallback, seq_mask
            )

        return view_1, len_1, view_2, len_2, stats

    def _build_one_adaptive_view(self, item_seq, sub_positions, ins_positions):
        seq_list = item_seq.detach().cpu().tolist()
        augmented_seq = []
        augmented_len = []
        aug_types = []
        for batch_idx, row in enumerate(seq_list):
            valid_positions = [idx for idx, item in enumerate(row) if int(item) != 0]
            valid_items = [int(row[idx]) for idx in valid_positions]
            compact_pos = {origin_pos: compact_idx for compact_idx, origin_pos in enumerate(valid_positions)}
            if random.random() < 0.5:
                positions = self._compact_selected_positions(
                    sub_positions[batch_idx], compact_pos
                ) if self.adaptive_substitute_position else None
                aug_seq = self.KG_substitute(valid_items, positions)
                aug_types.append('sub')
            else:
                positions = self._compact_selected_positions(
                    ins_positions[batch_idx], compact_pos
                ) if self.adaptive_insert_position else None
                aug_seq = self.KG_insert(valid_items, positions)
                aug_types.append('ins')
            cur_len = len(aug_seq) if len(aug_seq) < self.max_len else self.max_len
            aug_seq = aug_seq[-self.max_len:]
            aug_seq = aug_seq + [0] * (self.max_len - len(aug_seq))
            augmented_seq.append(aug_seq)
            augmented_len.append(cur_len)
        device = item_seq.device
        return (torch.tensor(augmented_seq, dtype=torch.long, device=device),
                torch.tensor(augmented_len, dtype=torch.long, device=device),
                aug_types)

    @staticmethod
    def _compact_selected_positions(selected_positions, compact_pos):
        """PAKA: map padded sequence positions to compact non-padding positions."""
        return [compact_pos[pos] for pos in selected_positions if pos in compact_pos]

    def KG_insert(self, item_seq, selected_positions=None):
        copied_item_seq = copy.deepcopy(item_seq)
        insert_num = int(self.insert_ratio * len(copied_item_seq))
        if selected_positions is None:
            insert_index = random.sample([i for i in range(len(copied_item_seq))], k=insert_num)
        else:
            insert_index = list(selected_positions)
        new_item_seq = []
        for index, item in enumerate(copied_item_seq):
            new_item_seq.append(item)
            if index in insert_index:
                shifted_item = item - 1
                insert_candidates = self.kg_relation_dict.get(shifted_item, {'c': []})['c']
                if len(insert_candidates) > 0:
                    insert_frequency = self.co_occurrence_dict.get(shifted_item, {'c': []})['c']
                    insert_item = np.random.choice(insert_candidates, size=1, p=insert_frequency)[0]
                    new_item_seq.append(insert_item + 1)
                else:
                    new_item_seq.append(item)
        return new_item_seq

    def KG_substitute(self, item_seq, selected_positions=None):
        copied_item_seq = copy.deepcopy(item_seq)
        substitute_num = int(self.substitute_ratio * len(copied_item_seq))
        if selected_positions is None:
            substitute_index = random.sample([i for i in range(len(copied_item_seq))], k=substitute_num)
        else:
            substitute_index = list(selected_positions)
        new_item_seq = []
        for index, item in enumerate(copied_item_seq):
            if index in substitute_index:
                shifted_item = item - 1
                substitute_candidates = self.kg_relation_dict.get(shifted_item, {'s': []})['s']
                if len(substitute_candidates) > 0:
                    substitute_frequency = self.co_occurrence_dict.get(shifted_item, {'s': []})['s']
                    substitute_item = np.random.choice(substitute_candidates, size=1, p=substitute_frequency)[0]
                    new_item_seq.append(substitute_item + 1)
                else:
                    new_item_seq.append(item)
                    new_item_seq.append(item)
            else:
                new_item_seq.append(item)
        return new_item_seq

    def _summarize_position_stats(self, importance, sub_reliability, ins_reliability, sub_scores, ins_scores,
                                  sub_selected_mask, ins_selected_mask, sub_fallback, ins_fallback, seq_mask):
        """PAKA: summarize selected-position diagnostics with tensor reductions."""
        def avg_selected(matrix, selected_mask):
            count = selected_mask.float().sum().clamp_min(1.0)
            return float((matrix * selected_mask.float()).sum().detach().cpu() / count.detach().cpu())

        def avg_from_end(selected_mask):
            batch_size, max_len = selected_mask.size()
            pos_ids = torch.arange(max_len, device=selected_mask.device).unsqueeze(0).expand(batch_size, max_len)
            last_pos = pos_ids.masked_fill(~seq_mask, -1).max(dim=1).values.unsqueeze(1)
            distance = (last_pos - pos_ids).float().clamp_min(0.0)
            count = selected_mask.float().sum().clamp_min(1.0)
            return float((distance * selected_mask.float()).sum().detach().cpu() / count.detach().cpu())

        sub_fallback_tensor = torch.tensor(sub_fallback, dtype=torch.float, device=importance.device)
        ins_fallback_tensor = torch.tensor(ins_fallback, dtype=torch.float, device=importance.device)
        return {
            'sub_importance': avg_selected(importance, sub_selected_mask),
            'sub_reliability': avg_selected(sub_reliability, sub_selected_mask),
            'sub_score': avg_selected(sub_scores, sub_selected_mask),
            'ins_importance': avg_selected(importance, ins_selected_mask),
            'ins_reliability': avg_selected(ins_reliability, ins_selected_mask),
            'ins_score': avg_selected(ins_scores, ins_selected_mask),
            'sub_fallback_ratio': float(sub_fallback_tensor.mean().detach().cpu()) if sub_fallback_tensor.numel() else 0.,
            'ins_fallback_ratio': float(ins_fallback_tensor.mean().detach().cpu()) if ins_fallback_tensor.numel() else 0.,
            'sub_from_end': avg_from_end(sub_selected_mask),
            'ins_from_end': avg_from_end(ins_selected_mask)
        }

    def _profile_start(self, name):
        if not self.use_adaptive_profile or self._profile_batch_count >= self.adaptive_profile_batches:
            return
        if self.item_embedding.weight.is_cuda:
            torch.cuda.synchronize(self.item_embedding.weight.device)
        self._profile_time[f'{name}_start'] = time.perf_counter()

    def _profile_stop(self, name):
        if not self.use_adaptive_profile or self._profile_batch_count >= self.adaptive_profile_batches:
            return
        if self.item_embedding.weight.is_cuda:
            torch.cuda.synchronize(self.item_embedding.weight.device)
        start = self._profile_time.pop(f'{name}_start', None)
        if start is None:
            return
        total, count = self._profile_time.get(name, (0.0, 0))
        self._profile_time[name] = (total + time.perf_counter() - start, count + 1)

    def _profile_report(self):
        if not self.use_adaptive_profile:
            return
        self._profile_batch_count += 1
        if self._profile_batch_count != self.adaptive_profile_batches:
            return
        parts = []
        for key, value in self._profile_time.items():
            if key.endswith('_start'):
                continue
            total, count = value
            parts.append(f'{key}={total / max(count, 1):.6f}s')
        logging.info('Adaptive profile average per batch: ' + ', '.join(parts))

    def forward(self, item_seq, seq_len, *args, **kwargs):
        item_vectors = self.position_encoding(item_seq)
        seq_embeddings = self.trm_encoder(item_seq, item_vectors)
        seq_embedding = self.gather_index(seq_embeddings, seq_len - 1)

        # get prediction
        logits = seq_embedding @ self.item_embedding.weight.t()

        return logits

    def position_encoding(self, item_input):
        seq_embedding = self.item_embedding(item_input)
        position = torch.arange(self.max_len, device=item_input.device).unsqueeze(0)
        position = position.expand_as(item_input).long()
        pos_embedding = self.position_embedding(position)
        seq_embedding += pos_embedding
        seq_embedding = self.input_dropout(self.input_layer_norm(seq_embedding))

        return seq_embedding

    def preference_encoding(self, item_seq, seq_len, return_all_hidden=False):
        item_vectors = self.position_encoding(item_seq)
        seq_embeddings = self.trm_encoder(item_seq, item_vectors)  # [B, L, D]
        seq_embedding = self.gather_index(seq_embeddings, seq_len - 1)  # [B, D]

        if return_all_hidden:
            return seq_embeddings, seq_embedding
        return seq_embedding


def KGSCL_config():
    parser = HyperParamDict('KGSCL default hyper-parameters')
    parser.add_argument('--model', default='KGSCL')
    parser.add_argument('--model_type', default='Knowledge', choices=['Sequential', 'Knowledge'])
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--ffn_hidden', default=512, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='dropout rate for attention')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float, help='transformer layer norm eps')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')
    parser.add_argument('--insert_ratio', default=0.2, type=float, help='KG-insert ratio')
    parser.add_argument('--substitute_ratio', default=0.7, type=float, help='KG-substitute ratio')
    parser.add_argument('--tem1', default=1., type=float, help='view-view CL temperature')
    parser.add_argument('--tem2', default=1., type=float, help='view-target CL temperature')
    parser.add_argument('--sim', default='dot', type=str, choices=['dot', 'cos'], help='InfoNCE loss similarity type')
    parser.add_argument('--lamda1', default=0.1, type=float, help='view-view CL loss weight')
    parser.add_argument('--lamda2', default=1.0, type=float, help='view-target CL loss weight')
    parser.add_argument('--use_mp_vt', default=False, action='store_true',
                        help='whether to use Multi-Positive View-Target CL')
    parser.add_argument('--mp_vt_top_m', default=3, type=int,
                        help='number of substitute neighbors used as extra MP-VT positives')
    parser.add_argument('--mp_vt_tau', default=None, type=float,
                        help='MP-VT temperature; reuse tem2 when not specified')
    parser.add_argument('--use_adaptive_position_aug', default=False, action='store_true',
                        help='whether to use position-level adaptive KG augmentation')
    parser.add_argument('--position_select_mode', default='sample', type=str, choices=['sample', 'topk'])
    parser.add_argument('--position_temperature', default=0.5, type=float)
    parser.add_argument('--position_importance_sim', default='cosine', type=str, choices=['cosine', 'dot'])
    parser.add_argument('--kg_reliability_agg', default='max', type=str, choices=['max', 'mean'])
    parser.add_argument('--adaptive_position_warmup_epochs', default=5, type=int)
    parser.add_argument('--adaptive_position_fallback', default='random', type=str, choices=['random', 'original'])
    parser.add_argument('--adaptive_substitute_position', default=True, action='store_false')
    parser.add_argument('--adaptive_insert_position', default=True, action='store_false')
    parser.add_argument('--use_adaptive_profile', default=False, action='store_true')
    parser.add_argument('--adaptive_profile_batches', default=20, type=int)
    parser.add_argument('--kg_data_type', default='other', type=str, choices=['pretrain', 'jointly_train', 'other'])
    parser.add_argument('--loss_type', default='CUSTOM', type=str, choices=['CE', 'BPR', 'BCE', 'CUSTOM'])

    return parser


if __name__ == '__main__':
    a = torch.randn(1, 3)
    b = torch.randn(5, 3)
    res = a.expand_as(b)
    print(res.size())
