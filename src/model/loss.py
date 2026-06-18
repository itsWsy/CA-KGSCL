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
