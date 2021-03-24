import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    
    # Hyperparameters
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=51)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--lr_backbone', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-04)
    parser.add_argument('--lr_drop', default=10, type=int)


    # Transformer (layers of enc and dec, dropout rate, num_heads, dim_feedforward)
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate applied in transformer")


    # dataset
    parser.add_argument('--eval', type=bool, default="False", help="Evaluation mode")
    parser.add_argument('--lift', type=bool, default=True, help="use Lifting or End-to-end version of PETR")
    parser.add_argument('--export_training_curves', type=bool, default="False", help="Save train/val curves in .png file")
    parser.add_argument('--dataset', type=str, default="dataset/S1/Seq1/imageSequence/full_S1.npz")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--resume', type=str, default=None, help="Loading model checkpoint")
    

    args = parser.parse_args()
    return args