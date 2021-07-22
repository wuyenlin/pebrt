import argparse


def args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser = argparse.ArgumentParser("Set PEBRT parameters", add_help=False)

    # Hyperparameters
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-04)
    parser.add_argument("--lr_backbone", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-05)
    parser.add_argument("--lr_drop", default=10, type=int)
    
    # Transformer (layers of enc and dec, dropout rate, num_heads, dim_feedforward)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate applied in transformer")
    
    # dataset
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="Loading model checkpoint for evaluation")
    parser.add_argument("--export_training_curves", action="store_true", help="Save train/val curves in .png file")
    # parser.add_argument("--dataset", type=str, default="./dataset/S1/Seq1/imageSequence/S1.npz")
    parser.add_argument("--dataset", type=str, default="./h36m/data_h36m_frame_all.npz")
    parser.add_argument("--device", default="cuda", help="device used")
    parser.add_argument("--resume", type=str, default=None, help="Loading model checkpoint")
    parser.add_argument("--distributed", action="store_true")
    
    # SLI
    parser.add_argument("--local_rank", type=int, help="local rank")
    parser.add_argument("--random_seed", type=int, help="random seed", default=0)
    
    args = parser.parse_args()
    return args
