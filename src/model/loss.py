import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss
    """

    def __init__(self, temperature, similarity_type):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature  # temperature
        self.sim_type = similarity_type  # cos or dot
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden1, aug_hidden2):
        """
        Args:
            aug_hidden1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden1.ndim > 2:
            # flatten tensor
            aug_hidden1 = aug_hidden1.view(aug_hidden1.size(0), -1)
            aug_hidden2 = aug_hidden2.view(aug_hidden2.size(0), -1)

        if self.sim_type not in ['cos', 'dot']:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. "
                            f"Please choose from ['cos', 'dot']")

        if self.sim_type == 'cos':
            sim11 = self.cosinesim(aug_hidden1, aug_hidden1)
            sim22 = self.cosinesim(aug_hidden2, aug_hidden2)
            sim12 = self.cosinesim(aug_hidden1, aug_hidden2)
        elif self.sim_type == 'dot':
            # calc similarity
            sim11 = aug_hidden1 @ aug_hidden1.t()
            sim22 = aug_hidden2 @ aug_hidden2.t()
            sim12 = aug_hidden1 @ aug_hidden2.t()
        # mask non-calc value
        sim11[..., range(sim11.size(0)), range(sim11.size(0))] = float('-inf')
        sim22[..., range(sim22.size(0)), range(sim22.size(0))] = float('-inf')

        cl_logits1 = torch.cat([sim12, sim11], -1)
        cl_logits2 = torch.cat([sim22, sim12.t()], -1)
        cl_logits = torch.cat([cl_logits1, cl_logits2], 0) / self.temperature
        target = torch.arange(cl_logits.size(0)).long().to(aug_hidden1.device)
        cl_loss = self.criterion(cl_logits, target)

        return cl_loss

    def cosinesim(self, aug_hidden1, aug_hidden2):
        h = torch.matmul(aug_hidden1, aug_hidden2.T)
        h1_norm2 = aug_hidden1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = aug_hidden2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)


def get_substitute_neighbors(item_id, sub_neighbors):
    """MP-VT: return substitute neighbors from dict/list style relation sets."""
    if sub_neighbors is None:
        return []
    if isinstance(sub_neighbors, dict):
        neighbors = sub_neighbors.get(item_id, None)
        if neighbors is None and item_id > 0:
            neighbors = sub_neighbors.get(item_id - 1, None)
        if isinstance(neighbors, dict):
            neighbors = neighbors.get('s', [])
        return list(neighbors) if neighbors is not None else []
    if isinstance(sub_neighbors, (list, tuple)):
        if 0 <= item_id < len(sub_neighbors):
            neighbors = sub_neighbors[item_id]
        elif item_id > 0 and item_id - 1 < len(sub_neighbors):
            neighbors = sub_neighbors[item_id - 1]
        else:
            return []
        if isinstance(neighbors, dict):
            neighbors = neighbors.get('s', [])
        return list(neighbors) if neighbors is not None else []
    return []


def get_corr_score(src, dst, corr_score):
    """MP-VT: read a correlation score from dict, nested dict/list, or matrix."""
    if corr_score is None:
        return 0.
    if isinstance(corr_score, dict):
        if (src, dst) in corr_score:
            return corr_score.get((src, dst), 0.)
        pair_key = f'{src}-{dst}' if src < dst else f'{dst}-{src}'
        if pair_key in corr_score:
            return corr_score.get(pair_key, 0.)
        src_score = corr_score.get(src, None)
        if src_score is None and src > 0:
            src_score = corr_score.get(src - 1, None)
        if isinstance(src_score, dict):
            if dst in src_score:
                return src_score.get(dst, 0.)
            if 's' not in src_score:
                return 0.
        return 0.
    try:
        return corr_score[src][dst]
    except Exception:
        return 0.


def get_last_valid_indices(seq_mask):
    """PAKA: return the last non-padding index for left- or right-padded sequences."""
    batch_size, max_len = seq_mask.size()
    position_ids = torch.arange(max_len, device=seq_mask.device).unsqueeze(0).expand(batch_size, max_len)
    last_indices = position_ids.masked_fill(~seq_mask, -1).max(dim=1).values
    return last_indices.clamp_min(0)


def masked_minmax_normalize(values, mask, eps=1e-8, constant_last_indices=None, positive_constant_one=False):
    """PAKA: vectorized per-row min-max normalization for [B, L] tensors."""
    pos_inf = torch.finfo(values.dtype).max
    neg_inf = torch.finfo(values.dtype).min
    masked_min = values.masked_fill(~mask, pos_inf).min(dim=1, keepdim=True).values
    masked_max = values.masked_fill(~mask, neg_inf).max(dim=1, keepdim=True).values
    valid_rows = mask.any(dim=1, keepdim=True)
    denom = masked_max - masked_min
    normalized = (values - masked_min) / (denom + eps)
    normalized = torch.where(valid_rows, normalized, torch.zeros_like(normalized))
    normalized = normalized.masked_fill(~mask, 0.0)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    constant_rows = valid_rows.squeeze(1) & (denom.squeeze(1).abs() < eps)
    if constant_last_indices is not None and constant_rows.any():
        normalized[constant_rows] = 0.0
        row_index = torch.nonzero(constant_rows, as_tuple=False).view(-1)
        normalized[row_index, constant_last_indices[row_index]] = 1.0
    elif positive_constant_one and constant_rows.any():
        positive_rows = constant_rows.unsqueeze(1) & values.gt(0) & mask
        normalized = torch.where(positive_rows, torch.ones_like(normalized), normalized)

    return normalized.masked_fill(~mask, 0.0)


def compute_position_importance(hidden_states, seq_mask, seq_lengths=None, similarity="cosine", eps=1e-8):
    """PAKA: compute normalized position importance, hidden_states is [B, L, D]."""
    batch_size = hidden_states.size(0)
    if seq_lengths is not None:
        last_indices = (seq_lengths.long() - 1).clamp_min(0)
    else:
        last_indices = get_last_valid_indices(seq_mask)
    batch_index = torch.arange(batch_size, device=hidden_states.device)
    user_repr = hidden_states[batch_index, last_indices]

    if similarity == "cosine":
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
        user_norm = F.normalize(user_repr, p=2, dim=-1)
        raw_score = torch.sum(hidden_norm * user_norm.unsqueeze(1), dim=-1)
    elif similarity == "dot":
        raw_score = torch.sum(hidden_states * user_repr.unsqueeze(1), dim=-1)
    else:
        raise ValueError("position_importance_sim should be cosine or dot")

    raw_score = raw_score.masked_fill(~seq_mask, 0.0)
    importance = masked_minmax_normalize(raw_score, seq_mask, eps=eps, constant_last_indices=last_indices)
    return importance, user_repr


def compute_kg_reliability_for_sequence(item_seq, seq_mask, reliability_table, agg="max", pad_id=0):
    """PAKA: index precomputed item-level reliability and normalize per sequence."""
    reliability = reliability_table[item_seq.clamp_min(0)].float()
    reliability = reliability.masked_fill(~seq_mask, 0.0)
    normalized = masked_minmax_normalize(reliability, seq_mask, positive_constant_one=True)
    zero_rows = reliability.masked_fill(~seq_mask, 0.0).max(dim=1, keepdim=True).values <= 0
    normalized = torch.where(zero_rows, torch.zeros_like(normalized), normalized)
    return normalized.masked_fill(item_seq.eq(pad_id), 0.0)


def compute_adaptive_position_scores(importance, reliability, seq_mask, mode, eps=1e-8):
    """PAKA: score substitute as K*(1-I), insert as K*I."""
    if mode == "substitute":
        scores = reliability * (1.0 - importance)
    elif mode == "insert":
        scores = reliability * importance
    else:
        raise ValueError("mode should be substitute or insert")
    scores = torch.clamp(scores, min=0.0)
    scores = scores.masked_fill(~seq_mask, 0.0)
    return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def select_adaptive_positions(position_scores, seq_mask, ratio, mode="sample", temperature=0.5, fallback="random",
                              return_mask=False):
    """PAKA: select non-padding positions by top-k or sampling without replacement."""
    selected_positions = []
    fallback_flags = []
    selected_mask = torch.zeros_like(seq_mask, dtype=torch.bool)
    temperature = max(float(temperature), 1e-8)

    with torch.no_grad():
        scores = position_scores.detach()
        mask = seq_mask.detach()
        for batch_idx in range(scores.size(0)):
            valid_positions = torch.nonzero(mask[batch_idx], as_tuple=False).view(-1)
            valid_len = int(valid_positions.numel())
            if valid_len == 0 or ratio <= 0:
                selected_positions.append([])
                fallback_flags.append(0)
                continue

            raw_select_num = float(ratio) * valid_len
            select_num = int(raw_select_num)
            if raw_select_num > select_num:
                select_num += 1
            select_num = max(1, min(select_num, valid_len))
            valid_scores = scores[batch_idx][valid_positions]
            valid_scores = torch.nan_to_num(valid_scores, nan=0.0, posinf=0.0, neginf=0.0)
            positive_mask = valid_scores > 0
            use_fallback = bool((not torch.isfinite(valid_scores).all()) or valid_scores.max() <= 0)

            if use_fallback:
                perm = torch.randperm(valid_len)[:select_num]
                chosen = valid_positions[perm]
                fallback_flags.append(1)
            else:
                fallback_flags.append(0)
                if int(positive_mask.sum().item()) < select_num:
                    positive_positions = valid_positions[positive_mask]
                    zero_positions = valid_positions[~positive_mask]
                    fill_num = select_num - int(positive_positions.numel())
                    if fill_num > 0 and zero_positions.numel() > 0:
                        fill = zero_positions[torch.randperm(zero_positions.numel())[:fill_num]]
                        chosen = torch.cat([positive_positions, fill], dim=0)
                    else:
                        chosen = positive_positions
                elif mode == "topk":
                    _, top_indices = torch.topk(valid_scores, k=select_num)
                    chosen = valid_positions[top_indices]
                elif mode == "sample":
                    prob = torch.softmax(valid_scores / temperature, dim=0)
                    sample_indices = torch.multinomial(prob, num_samples=select_num, replacement=False)
                    chosen = valid_positions[sample_indices]
                else:
                    raise ValueError("position_select_mode should be sample or topk")

            selected_mask[batch_idx, chosen] = True
            selected_positions.append([int(pos) for pos in chosen.detach().cpu().tolist()])

    if return_mask:
        return selected_positions, fallback_flags, selected_mask
    return selected_positions, fallback_flags


def _get_aligned_substitute_scores(item_id, corr_score):
    if not isinstance(corr_score, dict):
        return None
    item_score = corr_score.get(item_id, None)
    if item_score is None and item_id > 0:
        item_score = corr_score.get(item_id - 1, None)
    if isinstance(item_score, dict):
        return item_score.get('s', None)
    return item_score


def build_target_positive_sets(targets, sub_neighbors, corr_score=None, top_m=3, num_items=None, pad_id=0):
    """MP-VT: build {target item + TopM substitute neighbors} for each target."""
    pos_sets = []
    target_list = targets.detach().cpu().tolist() if torch.is_tensor(targets) else list(targets)

    for target in target_list:
        target = int(target)
        pos = [target]
        shifted_from_raw_kg = False

        if isinstance(sub_neighbors, dict) and target in sub_neighbors:
            key_item = target
            neighs = get_substitute_neighbors(key_item, sub_neighbors)
        elif isinstance(sub_neighbors, dict) and target > 0 and (target - 1) in sub_neighbors:
            key_item = target - 1
            neighs = get_substitute_neighbors(key_item, sub_neighbors)
            shifted_from_raw_kg = True
        else:
            key_item = target
            neighs = get_substitute_neighbors(key_item, sub_neighbors)
            if len(neighs) == 0 and target > 0:
                key_item = target - 1
                neighs = get_substitute_neighbors(key_item, sub_neighbors)
                shifted_from_raw_kg = len(neighs) > 0

        aligned_scores = _get_aligned_substitute_scores(key_item, corr_score)
        valid_pairs = []
        seen = {target}
        for idx, nb in enumerate(neighs):
            raw_nb = int(nb)
            model_nb = raw_nb + 1 if shifted_from_raw_kg else raw_nb
            if model_nb == pad_id:
                continue
            if num_items is not None and (model_nb < 0 or model_nb >= num_items):
                continue
            if model_nb in seen:
                continue
            seen.add(model_nb)
            if aligned_scores is not None and idx < len(aligned_scores):
                score = aligned_scores[idx]
            else:
                score = get_corr_score(key_item, raw_nb, corr_score)
            valid_pairs.append((model_nb, score))

        valid_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
        pos.extend([nb for nb, _ in valid_pairs[:top_m]])
        pos_sets.append(pos)

    return pos_sets


def multi_positive_view_target_loss(h_view, pos_sets, item_embedding, tau):
    """MP-VT: multi-positive view-target contrastive loss with stable logsumexp."""
    logits = torch.matmul(h_view, item_embedding.transpose(0, 1)) / tau
    batch_size, num_items = logits.size()
    device = logits.device
    pos_mask = torch.zeros((batch_size, num_items), dtype=torch.bool, device=device)

    if torch.is_tensor(pos_sets):
        pos_sets = pos_sets.detach().cpu().tolist()

    for batch_idx, pos_items in enumerate(pos_sets):
        if not isinstance(pos_items, (list, tuple)):
            pos_items = [pos_items]
        for item in pos_items:
            item = int(item)
            if 0 <= item < num_items:
                pos_mask[batch_idx, item] = True

    neg_inf = torch.finfo(logits.dtype).min
    pos_logits = logits.masked_fill(~pos_mask, neg_inf)
    log_pos = torch.logsumexp(pos_logits, dim=1)
    log_all = torch.logsumexp(logits, dim=1)
    return -(log_pos - log_all).mean()


class InfoNCELoss_2(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss, another implementation.
    """

    def __init__(self, temperature, similarity_type, batch_size):
        super(InfoNCELoss_2, self).__init__()
        self.tem = temperature  # temperature
        self.sim_type = similarity_type  # cos or dot
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden1, aug_hidden2):
        """
        Args:
            aug_hidden1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden1.ndim > 2:
            # flatten tensor
            aug_hidden1 = aug_hidden1.view(aug_hidden1.size(0), -1)
            aug_hidden2 = aug_hidden2.view(aug_hidden2.size(0), -1)

        current_batch = aug_hidden1.size(0)
        N = 2 * current_batch
        all_hidden = torch.cat((aug_hidden1, aug_hidden2), dim=0)  # [2*B, D]

        if self.sim_type == 'cos':
            sim = F.cosine_similarity(all_hidden.unsqueeze(1), all_hidden.unsqueeze(0), dim=2) / self.tem
        elif self.sim_type == 'dot':
            sim = torch.mm(all_hidden, all_hidden.T) / self.tem
        else:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. "
                            f"Please choose from ['cos', 'dot']")

        sim_i_j = torch.diag(sim, current_batch)
        sim_j_i = torch.diag(sim, -current_batch)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if self.batch_size != current_batch:
            mask = self.mask_correlated_samples(current_batch)
        else:
            mask = self.mask
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        nce_loss = self.criterion(logits, labels)

        return nce_loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N)).bool()
        mask = mask.fill_diagonal_(0)
        index1 = torch.arange(batch_size) + batch_size
        index2 = torch.arange(batch_size)
        index = torch.cat([index1, index2], 0).unsqueeze(-1)  # [2*B, 1]
        mask = torch.scatter(mask, -1, index, 0)
        return mask
