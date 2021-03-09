import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sess', type=int, default=0, help="session")
    parser.add_argument('--epoch', type=int, default=50, help="rounds of training (default=50)")
    parser.add_argument('--bs', type=int, default=128, help="batch size (default=128)")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate (default=0.005)")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default=0.9)")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--pretrained', type=bool, default=True, help="use pretrained ResNet50")
    parser.add_argument('--req_grad', type=bool, default=False, help="require gradients (i.e. False to freeze ResNet")

    args = parser.parse_args()
    return args