import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--gpu_id', type=list, default=[0], help='use cpu only')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

# Preprocess parametersx
parser.add_argument('--n_labels', type=int, default=2, help='number of classes')
parser.add_argument('--n_labels_seg', type=int, default=10, help='number of classes')
parser.add_argument('--task_num', type=int, default=2, help='number of task')
parser.add_argument('--n_experts', type=int, default=3, help='number of expert')
parser.add_argument('--n_experts_encoder', type=int, default=4, help='number of expert')
parser.add_argument('--k', type=int, default=2, help='number of selected expert')
parser.add_argument('--input_size', type=int, default=(64,256,256), help='input_size')
parser.add_argument('--attr', type=list, default=[1,1,1,0], help='input_size')

# data in/out and dataset
parser.add_argument('--dataset_path', default='./data/index/', help='fixed trainset root path')
parser.add_argument('--test_data_path', default='./data/index', help='Testset path')
parser.add_argument('--activation', default='sigmoid', help='activation')
parser.add_argument('--save', default="", help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=1, help='batch size of trainset')


# train
parser.add_argument('--epochs', type=int, default=2000, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--early_stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--freeze', default=True, type=int, help=' ')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')


args = parser.parse_args()
