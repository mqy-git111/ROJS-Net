import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)


class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self, class_num, activation):
        self.class_num = class_num
        self.activation = activation
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0
        self.precision = 0
        self.recall = 0
        self.F1 = 0
        self.list = []

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets, self.activation)
        self.list.append(self.value.tolist())
        output = DiceAverage.get_froc(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        self.precision = output[0]
        self.recall = output[1]
        self.F1 = output[2]
        # print(self.value)

    def update_1(self, logits, targets, task_num):
        self.value = [0]
        for i in range(task_num) :
            self.value.append(DiceAverage.get_dices(logits[i], targets[i], self.activation)[1])
        # self.list.append(self.value.tolist())
        # output = DiceAverage.get_froc(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # self.precision = output[0]
        # self.recall = output[1]
        # self.F1 = output[2]
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets, activation):
        logits = logits.float()
        targets = targets.float()

        if activation == 'sigmoid':
            logits = F.sigmoid(logits)
            logits[logits < 0.5] = 0
            logits[logits >= 0.5] = 1
        elif activation == 'softmax':
            logits = F.softmax(logits, dim=1)
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1e-8) / (union + 1e-8)
            dices.append(dice.item())
        return np.asarray(dices)

    @staticmethod
    def get_froc(logits, targets):
        # froc = []
        predict = np.array(list(np.where(logits.cpu().detach().numpy() < 0.5, 0, 1)[:])).astype(dtype=int)
        targets = np.array(targets.cpu().detach().numpy()).astype(dtype=int)

        TN = np.array(targets[predict == 0] == 0).sum()
        FP = np.array(targets[predict == 1] == 0).sum()
        FN = np.array(targets[predict == 0] == 1).sum()
        TP = np.array(targets[predict == 1] == 1).sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        return precision, recall, F1



#按有病没病
class LossAverage_1(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)


class DiceAverage_1(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self, class_num, activation):
        self.class_num = class_num
        self.activation = activation
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg_sick = np.asarray([0] * self.class_num, dtype='float64')
        self.avg_unsick = np.asarray([0] * self.class_num, dtype='float64')
        self.sum_sick = np.asarray([0] * self.class_num, dtype='float64')
        self.sum_unsick = np.asarray([0] * self.class_num, dtype='float64')
        self.count_sick = np.asarray([0] * self.class_num, dtype='float64')
        self.count_unsick = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0
        self.precision = 0
        self.recall = 0
        self.F1 = 0
        self.list = []

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets, self.activation)
        if self.count < 32:
            flag = 2
        elif self.count < 52:
            flag = 1
        else:
            flag = 3
        for i in range(4):
            if i == flag:
                self.sum_sick[i] = self.sum_sick[i] + self.value[i]
                self.count_sick[i] = self.count_sick[i] + 1
                self.avg_sick[i] = self.sum_sick[i] / self.count_sick[i]
            else:
                self.sum_unsick[i] = self.sum_unsick[i] + self.value[i]
                self.count_unsick[i] = self.count_unsick[i] + 1
                self.avg_unsick[i] = self.sum_unsick[i] / self.count_unsick[i]

        # self.list.append(self.value.tolist())
        # output = DiceAverage.get_froc(logits, targets)
        # self.sum += self.value
        self.count += 1
        # self.avg = np.around(self.sum / self.count, 4)
        # self.precision = output[0]
        # self.recall = output[1]
        # self.F1 = output[2]
        # print(self.value)


    def update_1(self, logits, targets):
        self.value = [0, DiceAverage.get_dices(logits[0], targets[0], self.activation)[1],DiceAverage.get_dices(logits[1], targets[1], self.activation)[1],DiceAverage.get_dices(logits[2], targets[2], self.activation)[1]]
        if self.count < 32:
            flag = 2
        elif self.count < 52:
            flag = 1
        else:
            flag = 3
        for i in range(4):
            if i == flag:
                self.sum_sick[i] = self.sum_sick[i] + self.value[i]
                self.count_sick[i] = self.count_sick[i] + 1
                self.avg_sick[i] = self.sum_sick[i] / self.count_sick[i]
            else:
                self.sum_unsick[i] = self.sum_unsick[i] + self.value[i]
                self.count_unsick[i] = self.count_unsick[i] + 1
                self.avg_unsick[i] = self.sum_unsick[i] / self.count_unsick[i]

        # self.list.append(self.value.tolist())
        # output = DiceAverage.get_froc(logits, targets)
        # self.sum += self.value
        self.count += 1
        # self.avg = np.around(self.sum / self.count, 4)
        # self.precision = output[0]
        # self.recall = output[1]
        # self.F1 = output[2]
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets, activation):
        logits = logits.float()
        targets = targets.float()

        if activation == 'sigmoid':
            logits = F.sigmoid(logits)
            logits[logits < 0.5] = 0
            logits[logits >= 0.5] = 1
        elif activation == 'softmax':
            logits = F.softmax(logits, dim=1)
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1e-8) / (union + 1e-8)
            dices.append(dice.item())
        return np.asarray(dices)

    @staticmethod
    def get_froc(logits, targets):
        # froc = []
        predict = np.array(list(np.where(logits.cpu().detach().numpy() < 0.5, 0, 1)[:])).astype(dtype=int)
        targets = np.array(targets.cpu().detach().numpy()).astype(dtype=int)

        TN = np.array(targets[predict == 0] == 0).sum()
        FP = np.array(targets[predict == 1] == 0).sum()
        FN = np.array(targets[predict == 0] == 1).sum()
        TP = np.array(targets[predict == 1] == 1).sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        return precision, recall, F1
