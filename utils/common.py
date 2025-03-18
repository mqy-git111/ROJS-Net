import torch

# target one-hot编码
def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.5 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_V2(optimizer, lr):
    """Sets the learning rate to a fixed number"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


