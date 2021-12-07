import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something','jhmdb','diving48','finegym99','finegym288', 'tinykinetics','minikinetics'])
parser.add_argument('modality', type=str, choices=['gray', 'RGB', 'Flow', 'RGBDiff'])
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)
parser.add_argument('test_list', type=str)

# ========================= Model Configs ============================
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--mode', type=int, default=1)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--pretrained_parts', type=str, default='both',
                    choices=['scratch', '2D', '3D', 'both','finetune'])

# ========================= RSA Configs =====================
parser.add_argument('-tfm', '--transform', type=str, default="RSA", choices=['conv', 'LSA', 'RSA'])
parser.add_argument('-pos', '--position', type=str, default="[[2],[1,3],[1,3,5],[1]]")
parser.add_argument('-ks', '--kernel_size', type=str, default="[5,7,7]")
parser.add_argument('-nh', '--nh', type=int, default=8)
parser.add_argument('-dk', '--dk', type=int, default=0)
parser.add_argument('-dv', '--dv', type=int, default=0)
parser.add_argument('-dd', '--dd', type=int, default=0)
parser.add_argument('-ktype', '--kernel_type', type=str, default="VplusR", choices=['V', 'R', 'VplusR'])
parser.add_argument('-ftype', '--feat_type', type=str, default="VplusR", choices=['V', 'R', 'VplusR'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-i', '--iter-size', default=1, type=int,
                    metavar='N', help='number of iterations before on update')
parser.add_argument('--cosine_lr', default=False, action='store_true', help='Cosine lr annealing strategy.')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--nesterov',  default=False)
parser.add_argument('--warmup',  default=0, type=int)
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll', 'smooth_nll'])
parser.add_argument('--label_smoothness', type=float, default=0.0)
parser.add_argument('--stochastic_depth', type=float, default=0.0)
parser.add_argument('--mixup_alpha', type=float, default=0.0)
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')

# ========================= Multi-grid Configs ==========================
parser.add_argument('--num_long_cycles', default=0, type=int,
                    help='number of long cycle grid')
parser.add_argument('--num_short_cycles', default=0, type=int,
                    help='number of short cycle grid')
parser.add_argument('--last_cycle_tune', default=False, action='store_true', 
                    help='different meaning from the papar. NO MULTIGRID at FINETUNE stage.')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--wandb', default=False, action='store_true', help='Wandb debugging.')
parser.add_argument('--proj_name', type=str, default="Test_default")
parser.add_argument('--exp_name', type=str, default="TSM_R18_something_run0_default")
parser.add_argument('--entity_name', type=str, default="mandos")
# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--tar_ball', type=str, default="", help='path to dataset tar file (default: none)')
parser.add_argument('--log_dir', type=str, default="", help='path to log')
parser.add_argument('--val_output_folder', type=str, default="", help="folder location to store validation scores")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="img_", type=str)
parser.add_argument('--rgb_prefix', default="img_", type=str)








