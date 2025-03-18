#测试代码并且保存预测图为.nii格式
import numpy
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import config
import torch
import os
import csv
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#对比算法
#from models import pine_unet,TransUnet3d,VNet,Atten_Net3D,unetr_pp_model
#消融实验
#from models import res_2decoder,res_semmoe,res_semmoe_prompt,res_seprompt_tsp
#不同策略
#from models import resunet_organ,resunet_tumor,pine_res_seunet
#不同backbone
from models import unet_2decoder,transmmoe_prompt,mmoe,mmoe_prompt,trans_mmoe,TransUnet3d_2decoder,unet_prompt,transunet_prompt

from utils import common
def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list

def test(ct_array, model, mode=False):   #这里

    model.eval()
    data = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        data = data.float()
        data = data.to(device)
        output = model(data)
        pre_organ = output[0]
        pre_tumor = output[1]
        pre_organ = F.softmax(pre_organ,dim=1)
        pre_tumor = F.softmax(pre_tumor,dim=1)
        pre_organ = torch.argmax(pre_organ, dim=1).unsqueeze(0)
        pre_tumor = torch.argmax(pre_tumor, dim=1).unsqueeze(0)
    return [pre_organ,pre_tumor]

def getdice(logits,targets):
    inter = torch.sum(logits * targets)
    union = torch.sum(logits) + torch.sum(targets)
    dice = (2. * inter + 1e-8) / (union + 1e-8)
    return dice

if __name__ == '__main__':
    args = config.args
    name = "unet"
    #不同backbone unet
    model = unet_2decoder.UNet(1, args.n_labels, args.task_num, training=True).to(device)
    # model = mmoe.MMoE(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)
    # model = mmoe_prompt.MMoE(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)
    # model = unet_prompt.unet_prompt(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num,args.attr).to(device)

    # 不同backbone transunet
    # args.lr = 0.0005
    # model = transmmoe_prompt.Trans_MMoE((64, 256, 256), 1, 12, 2, 4).to(device)
    # model = TransUnet3d_2decoder.TranUnet((64, 256, 256), 1, 12, 2, 4).to(device)
    # model = trans_mmoe.Trans_MMoE((64, 256, 256), 1, 12, 2, 4).to(device)
    # model = transunet_prompt.Trans_MMoE((64, 256, 256), 1, 12, 2, 4).to(device)


    # pine_model_path = 'unetr_pp2222_0005./results/best_model_114.pth'
    pine_model_path = './results_backbone/' + name + '/best_model.pth'  #99
    save_path = os.path.join("./results_backbone",os.path.join(name,"prediction"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    head = ["name", 'oral_cavity', 'neck_right', 'neck_left', 'parotid_right', 'parotid_left', 'submandibular_right', 'submandibular_left','larynx',"ptv"]
    # model = pine_unet.UNet(1, args.n_labels, args.task_num, training=True).to(device)
    #model = TransUnet3d.TranUnet((64, 256, 256), 1, 12, 2, 4).to(device)
    # model = VNet.VNet().to(device)
    # model = Atten_Net3D.AttU_Net().to(device)
    # model = swin_unetr.SwinUNETR(img_size=(64,256,256),in_channels=1,out_channels=13,feature_size = 12,drop_rate=0.0,attn_drop_rate=0.0,dropout_path_rate=False,use_checkpoint=False).to(device)
    # model = unetr_pp_model.UNETR_PP(1, 2).to(device)
    # 消融实验
    # model = res_2decoder.UNet(1, args.n_labels, args.task_num, training=True).to(device)
    # model = res_seprompt_tsp.resunet_prompt(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)
    # model = res_semmoe.MMoE(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)

    # 不同策略
    # model = pine_res_seunet.UNet(1, args.n_labels, args.task_num, training=True).to(device)



    # ours
    # model = res_semmoe_prompt.MMoE(args.input_size, 1, 2, args.n_experts, True, args.k, args.task_num, args.attr).to(device)

    ckpt = torch.load(pine_model_path)
    model.load_state_dict(ckpt['net'])
    # model.load_state_dict(ckpt)
    model.to(device)
    filename_list = load_file_name_list(os.path.join(args.test_data_path, 'test_path_list.txt'))
    dices = [0, 0, 0, 0, 0]
    num = 0
    with open("predict.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for singleImg in filename_list:
            name = singleImg[0].split("/")[-1][:]
            img = sitk.ReadImage(singleImg[0])
            img_fdata = sitk.GetArrayFromImage(img)
            organ = sitk.ReadImage(singleImg[1])
            organ_fdata = sitk.GetArrayFromImage(organ)
            tumor = sitk.ReadImage(singleImg[2])
            tumor_fdata = sitk.GetArrayFromImage(tumor)
            organ_fdata = torch.FloatTensor(organ_fdata.astype(float)).unsqueeze(0)
            tumor_fdata = torch.FloatTensor(tumor_fdata.astype(float)).unsqueeze(0)
            organ_label = common.to_one_hot_3d(organ_fdata.long(), 4)
            tumor_label = common.to_one_hot_3d(tumor_fdata.long(), 2)
            organ_label = np.asarray(organ_label.cpu().numpy(), dtype='uint8')
            tumor_label = np.asarray(tumor_label.cpu().numpy(), dtype='uint8')
            # predLabel_array是多个列表
            predLabel_array = test(img_fdata, model, False)
            preorgan_fdata = predLabel_array[0].squeeze()
            pretumor_fdata = predLabel_array[1].squeeze()
            # 保存
            if not os.path.exists(os.path.join(save_path, name)):
                os.makedirs(os.path.join(save_path, name))
            preorgan_fdata = np.asarray(preorgan_fdata.cpu().numpy(), dtype='uint8')
            pretumor_fdata = np.asarray(pretumor_fdata.cpu().numpy(), dtype='uint8')
            predorgan = sitk.GetImageFromArray(preorgan_fdata)
            predorgan.SetSpacing(img.GetSpacing())
            predorgan.SetDirection(img.GetDirection())
            predorgan.SetOrigin(img.GetOrigin())
            predtumor = sitk.GetImageFromArray(pretumor_fdata)
            predtumor.SetSpacing(img.GetSpacing())
            predtumor.SetDirection(img.GetDirection())
            predtumor.SetOrigin(img.GetOrigin())
            sitk.WriteImage(predorgan, os.path.join(save_path, os.path.join(name, "organ.nii.gz")))
            sitk.WriteImage(predtumor, os.path.join(save_path, os.path.join(name, "tumor.nii.gz")))
            # 计算dice
            preorgan_fdata = predLabel_array[0].squeeze()
            pretumor_fdata = predLabel_array[1].squeeze()
            preorgan_fdata = preorgan_fdata.unsqueeze(0).cpu()
            pretumor_fdata = pretumor_fdata.unsqueeze(0).cpu()
            preorgan_labels = common.to_one_hot_3d(preorgan_fdata.long(), 4).numpy()
            pretumor_labels = common.to_one_hot_3d(pretumor_fdata.long(), 2).numpy()
            # organ_label = predLabel_array
            temp_dice = [name]
            for i in range(1, 4):
                dice = getdice(torch.FloatTensor(organ_label[:, i, :, :, :]),
                               torch.FloatTensor(preorgan_labels[:, i, :, :, :])) * 100
                dice = np.array(dice)
                temp_dice.append(dice)
                dices[i] = dices[i] + dice
            dice = getdice(torch.FloatTensor(tumor_label[:, 1, :, :, :]),
                           torch.FloatTensor(pretumor_labels[:, 1, :, :, :])) * 100
            dice = np.array(dice)
            dices[4] = dices[4] + dice
            temp_dice.append(dice)
            # temp_dice = numpy.array(temp_dice)
            writer.writerow(temp_dice)
            print(temp_dice)
            num = num + 1
        all = [dices[i] / num for i in range(0, 5)]
        writer.writerow(all)
        # # dice_numpy = numpy.array(all)
        print(all)
        avg_organ =  np.sum(all[0:4]) / 3
        print(avg_organ)
        avg = np.sum(all) / 4
        writer.writerow([avg])
        print(avg)
    # print("img:{},sucessful!".format(singleImg))

    print("finish!")
    # pine_model = torch.nn.DataParallel(pine_model, device_ids=args.gpu_id)  # multi-GPU
