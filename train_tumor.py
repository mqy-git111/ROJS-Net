import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from torch.autograd import Variable
from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset
from tensorboardX import SummaryWriter
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import config
from models import resunet_tumor
from utils import logger,  metrics, common
from utils.losses import get_loss
from collections import OrderedDict
import numpy as np
import random

def val(model, val_loader, loss_func, n_labels):
    model.eval()
    train_loss = metrics.LossAverage()
    train_organ_dice = metrics.DiceAverage(9, 'softmax')
    train_tumor_dice = metrics.DiceAverage(2, 'softmax')
    with torch.no_grad():
        for idx, (data, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            # target = targets[0]
            data = data.float()
            seg_array = dict()
            labels_organ = targets[0].long()
            labels_tumor = targets[1].long()
            labels_organ = common.to_one_hot_3d(labels_organ, 9).to(device)
            labels_tumor = common.to_one_hot_3d(labels_tumor, 2).to(device)
            data = data.float()
            data = Variable(data)
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, labels_tumor)
            train_loss.update(loss.item()/2, data.size(0))
            # train_organ_dice.update(output, labels_tumor)
            train_tumor_dice.update(output, labels_tumor)

    val_log = OrderedDict({'Val_Loss': train_loss.avg, 'Val_dice_1': train_tumor_dice.avg[1]})
    sum = train_tumor_dice.avg[1]
    # for i in range(2,3):
    #     val_log.update({'Val_dice_' + str(i): train_organ_dice.avg[i]})
    #     sum = sum + train_organ_dice.avg[i]
    # val_log.update({'Val_dice_' + str(9): train_tumor_dice.avg[1]})
    # sum = sum + train_tumor_dice.avg[1]
    val_log['Val_dice_avg'] = sum

    return val_log



def train(model, train_loader, optimizer, loss_func, n_labels):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_organ_dice = metrics.DiceAverage(9, 'softmax')
    train_tumor_dice = metrics.DiceAverage(2, 'softmax')
    ce = nn.CrossEntropyLoss()
    for idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # target = targets[0]
        data = data.float()
        seg_array = dict()
        labels_organ_1 = targets[0].long()
        labels_tumor_1 = targets[1].long()
        labels_organ = common.to_one_hot_3d(labels_organ_1, 9).to(device)
        labels_tumor = common.to_one_hot_3d(labels_tumor_1, 2).to(device)
        data = data.float()
        data = Variable(data)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, labels_tumor) + ce(output, labels_tumor_1.to(device))
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item()/4, data.size(0))
        # train_organ_dice.update(output, labels_tumor)
        train_tumor_dice.update(output, labels_tumor)

    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_1': train_tumor_dice.avg[1]})
    sum = train_tumor_dice.avg[1]
    # for i in range(2,8):
    #     train_log.update({'Train_dice_' + str(i): train_organ_dice.avg[i]})
    #     sum = sum + train_organ_dice.avg[i]
    # train_log.update({'Train_dice_' + str(8): train_tumor_dice.avg[1]})
    # sum = sum + train_tumor_dice.avg[1]
    train_log['Train_dice_avg'] = sum

    return train_log


if __name__ == '__main__':
    args = config.args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dataloader
    save_path = os.path.join('./results/resunet_tumor', args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # device = torch.device('cpu' if args.cpu else 'cuda')
    logging.basicConfig(filename=os.path.join(save_path,"./unet.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # 控制输出到控制台
    writer = {
        'avg':SummaryWriter(os.path.join(save_path,"avg")),
        '1':SummaryWriter(os.path.join(save_path,"1"))
    }

    # data info:
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=1, num_workers=args.n_threads,
                              shuffle=False)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)

    # model info
    model = resunet_tumor.UNet(1, args.n_labels, args.task_num, training=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # lr_decay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    common.print_network(model)
    # model = torch.nn.DataParallel(model.cuda(), device_ids=args.gpu_id)  # multi-GPU

    loss = get_loss.SegLoss('dice', 'softmax')

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels_seg)
        val_log = val(model, val_loader, loss, args.n_labels_seg)
        log.update(epoch, train_log, val_log)   #输出
        logging.info(train_log)
        logging.info(val_log)
        name = ["avg", "1"]
        writer["avg"].add_scalar('scalar/train_loss', train_log["Train_Loss"], epoch)
        writer["avg"].add_scalar('scalar/val_loss', val_log["Val_Loss"], epoch)
        for i in name:
            writer[i].add_scalar('scalar/train_dice', train_log["Train_dice_" + i], epoch)
            writer[i].add_scalar('scalar/val_dice', val_log["Val_dice_" + i], epoch)

        # Save checkpoint.2
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        # torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_avg'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model_' + str(epoch) + '.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_avg']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # lr_decay.step(val_log['Val_Loss'])
        #
        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
