import argparse
from src.train.trainer import Trainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', default='KGSCL')
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--ffn_hidden', default=512, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='dropout rate for attention')
    parser.add_argument('--insert_ratio', default=0.2, type=float, help='KG-insert ratio')
    parser.add_argument('--substitute_ratio', default=0.7, type=float, help='KG-substitute ratio')
    parser.add_argument('--tem1', default=1., type=float, help='view-view CL temperature')
    parser.add_argument('--tem2', default=1., type=float, help='view-target CL temperature')
    parser.add_argument('--sim', default='dot', type=str, choices=['dot', 'cos'], help='InfoNCE loss similarity type')
    parser.add_argument('--lamda1', default=0.1, type=float, help='view-view CL loss weight')
    parser.add_argument('--lamda2', default=1.0, type=float, help='view-target CL loss weight')
    parser.add_argument('--use_mp_vt', nargs='?', const=True, default=False, type=str2bool,
                        help='whether to use Multi-Positive View-Target CL')
    parser.add_argument('--mp_vt_top_m', default=3, type=int,
                        help='number of substitute neighbors used as extra MP-VT positives')
    parser.add_argument('--mp_vt_tau', default=None, type=float,
                        help='MP-VT temperature; reuse tem2 when not specified')
    parser.add_argument('--use_adaptive_position_aug', nargs='?', const=True, default=False, type=str2bool,
                        help='whether to use position-level adaptive KG augmentation')
    parser.add_argument('--position_select_mode', default='sample', type=str, choices=['sample', 'topk'])
    parser.add_argument('--position_temperature', default=0.5, type=float)
    parser.add_argument('--position_importance_sim', default='cosine', type=str, choices=['cosine', 'dot'])
    parser.add_argument('--kg_reliability_agg', default='max', type=str, choices=['max', 'mean'])
    parser.add_argument('--adaptive_position_warmup_epochs', default=5, type=int)
    parser.add_argument('--adaptive_position_fallback', default='random', type=str, choices=['random', 'original'])
    parser.add_argument('--adaptive_substitute_position', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--adaptive_insert_position', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--use_adaptive_profile', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--adaptive_profile_batches', default=20, type=int)
    parser.add_argument('--adaptive_position_selection_impl', default='cpu', type=str, choices=['cpu', 'mask', 'legacy'])
    # Data
    parser.add_argument('--dataset', default='toys', type=str)
    parser.add_argument('--data_aug', action='store_false', help='data augmentation')
    parser.add_argument('--separator', default=' ', type=str, help='separator to split item sequence')
    # Training
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--seed', default=-1, type=int, help="random seed: -1 means don't set random seed")
    parser.add_argument('--train_batch', default=512, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--l2', default=0., type=float, help='l2 normalization')  # 1e-6
    parser.add_argument('--patience', default=10, help='early stop patience')
    parser.add_argument('--mark', default='', type=str)
    parser.add_argument('--sh', default='', type=str, help='shell/script file name used in log file name')
    parser.add_argument('--device', default='cuda:0', type=str)

    # Evaluation
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='LS', type=str, help='[LS, LS_R@0.x, PS]')
    parser.add_argument('--eval_mode', default='full', help='[uni100, pop100, full]')
    parser.add_argument('--k', default=[5, 10], help='rank k for each metric')
    parser.add_argument('--metric', default=['hit', 'ndcg'], help='[hit, ndcg, mrr, recall]')
    parser.add_argument('--valid_metric', default='hit@10', help='indicator to apply early stop')

    config = parser.parse_args()

    trainer = Trainer(config)
    trainer.start_training()

