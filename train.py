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
#对比算法
from models_all import pine_unet,TransUnet3d,VNet,Atten_Net3D,unetr_pp_model,swin_unetr
# #消融实验
# from models import res_2decoder,res_semmoe,res_semmoe_prompt,res_seprompt_tsp
# #不同策略
# from models import resunet_organ,resunet_tumor,pine_res_seunet
#不同backbone
from models import unet_2decoder,transmmoe_prompt,mmoe,mmoe_prompt,trans_mmoe,TransUnet3d_2decoder
from utils import logger,  metrics, common
from utils.losses import get_loss
from collections import OrderedDict
import numpy as np
import random
def load_pretrain(path,name):
    freeze_name = []
    if name == "unet":
        encoder_model = Encoder(1,12)
    else:
        encoder_model = TransUnetEncoder((64, 256, 256), 1, 6, 3)
    for name,data in encoder_model.named_parameters():
        freeze_name.append(name)
    print(encoder_model)
    ckpt = torch.load(path)
    ckpt_net = ckpt['net']
    encoder_model.load_state_dict(ckpt['net'],strict=False)
    return encoder_model.state_dict(), freeze_name
def val(model, val_loader, loss_func, n_labels):
    model.eval()
    train_loss = metrics.LossAverage()
    train_organ_dice = metrics.DiceAverage(4, 'softmax')
    train_tumor_dice = metrics.DiceAverage(2, 'softmax')
    with torch.no_grad():
        for idx, (data, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data = data.float()
            labels_organ = targets[0].long()
            labels_tumor = targets[1].long()
            labels_organ = common.to_one_hot_3d(labels_organ, 4).to(device)
            labels_tumor = common.to_one_hot_3d(labels_tumor, 2).to(device)
            data = data.float()
            data = Variable(data)
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output[0], labels_organ) + loss_func(output[1], labels_tumor)

            train_loss.update(loss.item(), data.size(0))
            train_organ_dice.update(output[0], labels_organ)
            train_tumor_dice.update(output[1], labels_tumor)

    val_log = OrderedDict({'Val_Loss': train_loss.avg, 'Val_dice_1': train_organ_dice.avg[1]})
    sum = train_organ_dice.avg[1]
    for i in range(2,4):
        val_log.update({'Val_dice_' + str(i): train_organ_dice.avg[i]})
        sum = sum + train_organ_dice.avg[i]
    val_log.update({'Val_dice_' + str(4): train_tumor_dice.avg[1]})
    sum = sum + train_tumor_dice.avg[1]
    val_log['Val_dice_avg'] = sum / 4

    return val_log
def train(model, train_loader, optimizer, loss_func, n_labels):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_organ_dice = metrics.DiceAverage(4, 'softmax')
    train_tumor_dice = metrics.DiceAverage(2, 'softmax')

    for idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # target = targets[0]
        data = data.float()
        seg_array = dict()
        labels_organ = targets[0].long()
        labels_tumor = targets[1].long()
        labels_organ = common.to_one_hot_3d(labels_organ, 4).to(device)
        labels_tumor = common.to_one_hot_3d(labels_tumor, 2).to(device)
        data = data.float()
        data = Variable(data)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        ce = nn.CrossEntropyLoss(weight=None, ignore_index=-1)
        loss = loss_func(output[0], labels_organ) + loss_func(output[1], labels_tumor) + ce(output[0], labels_organ) + ce(output[1], labels_tumor)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), data.size(0))
        train_organ_dice.update(output[0], labels_organ)
        train_tumor_dice.update(output[1], labels_tumor)

    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_1': train_organ_dice.avg[1]})
    sum = train_organ_dice.avg[1]
    for i in range(2,4):
        train_log.update({'Train_dice_' + str(i): train_organ_dice.avg[i]})
        sum = sum + train_organ_dice.avg[i]
    train_log.update({'Train_dice_' + str(4): train_tumor_dice.avg[1]})
    sum = sum + train_tumor_dice.avg[1]
    train_log['Train_dice_avg'] = sum / 4

    return train_log


if __name__ == '__main__':
    args = config.args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dataloader
    save_path = os.path.join('./results_backbone/mmoe', args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    logging.basicConfig(filename=os.path.join(save_path,"./mmoe_aug.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # 控制输出到控制台
    writer = {
        'avg':SummaryWriter(os.path.join(save_path,"avg")),
        '1':SummaryWriter(os.path.join(save_path,"1")),
        '2':SummaryWriter(os.path.join(save_path,"2")),
        '3':SummaryWriter(os.path.join(save_path,"3")),
        '4':SummaryWriter(os.path.join(save_path,"4")),
    }

    # data info:
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=1, num_workers=args.n_threads,
                              shuffle=False)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)
    # 对比算法
    # from models import pine_unet, TransUnet3d, VNet, Atten_Net3D, swin_unetr
    # 消融实验
    # from models import res_2decoder, res_semmoe, res_semmoe_prompt
    # 不同策略
    # from models import resunet_organ, resunet_tumor, pine_res_seunet
    # model info
    # model = pine_unet.UNet(1, args.n_labels, args.task_num, training=True).to(device)
    # model = TransUnet3d.TranUnet((64, 256, 256), 1, 12, 2, 4).to(device)
    # model = VNet.VNet().to(device)
    # model = Atten_Net3D.AttU_Net().to(device)
    model = swin_unetr.SwinUNETR(img_size=(64,256,256),in_channels=1,out_channels=13,feature_size = 12,drop_rate=0.0,attn_drop_rate=0.0,dropout_path_rate=False,use_checkpoint=False).to(device)
    # model = unetr_pp_model.UNETR_PP(1,2).to(device)
    #消融实验
    # model = res_2decoder.UNet(1, args.n_labels, args.task_num, training=True).to(device)
    # model = res_seprompt_tsp.resunet_prompt(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)
    # model = res_semmoe.MMoE(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)

    #不同策略
    # model = pine_res_seunet.UNet(1, args.n_labels, args.task_num, training=True).to(device)

    #不同backbone unet
    # model = unet_2decoder.UNet(1, args.n_labels, args.task_num, training=True).to(device)
    model = mmoe.MMoE(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)
    # model = mmoe_prompt.MMoE(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)


    # 不同backbone transunet
    #args.lr = 0.0005
    # model = transmmoe_prompt.Trans_MMoE((64, 256, 256), 1, 12, 2, 4).to(device)
    #model = TransUnet3d_2decoder.TranUnet((64, 256, 256), 1, 12, 2, 4).to(device)
    # model = trans_mmoe.Trans_MMoE((64, 256, 256), 1, 12, 2, 4).to(device)

    # ours
    # model = res_semmoe_prompt.MMoE(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)
    #加载预训练权重
    # pretrain_model = r'./pre_train_models/best_model.pth'
    # _, freeze_namelist = load_pretrain(pretrain_model, "unet")
    # ckpt = torch.load(pretrain_model)
    # model_dict = ckpt["net"]
    # pretrained_dict = {key: value for key, value in model_dict.items() if
    #                    (key[8:] in freeze_namelist)}
    # for key, value in model_dict.items():
    #     if key.find("dtp") != -1 or key.find("uni") != -1:
    #     # if key.find("dtp") != -1 or key.find("uni") != -1 or key.find("w_gate") != -1:
    #         pretrained_dict[key] = value
    # model.load_state_dict(pretrained_dict, strict=False)
    # #
    #
    # pretrain_model = r'./results/unetr_pp2222_0001/best_model_99.pth'   #84高
    # # _, freeze_namelist = load_pretrain(pretrain_model, "unet")
    # ckpt = torch.load(pretrain_model)
    # model_dict = ckpt["net"]
    # model.load_state_dict(model_dict, strict=False)

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
        name = ["avg", "1", "2", "3", "4"]
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
