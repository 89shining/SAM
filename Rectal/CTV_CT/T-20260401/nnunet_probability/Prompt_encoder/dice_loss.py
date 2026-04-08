import torch.nn as nn
import torch
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        # 定义一个小常数，避免计算中的除零错误
        self.epsilon = 1e-5

    def forward(self, preds, targets):
        # 取目标张量的第一个维度大小，通常为批次大小
        N = targets.size()[0]
        # sigmoid函数将任意实数值映射到0-1的连续概率值
        predsigmoid = F.sigmoid(preds)
        # 将预测和目标掩码展开为二维数组，将图像的空间维度 [C, H, W] 展平为一维
        preds_flat = predsigmoid.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算预测与真实值的交集即重叠部分
        intersection = preds_flat * targets_flat
        # 对每个交集求和rp
        rp = intersection.sum(1)
        # 真实掩码中目标类别的像素总数wl
        wl = targets_flat.sum(1)  # cls
        # 进行归一化处理，计算权重wl
        wl = 1.0 / (wl * wl + 1e-5)
        # 计算预测掩码与真实掩码的总和r_p
        r_p = (preds_flat + targets_flat).sum(1)
        # wl * rp加权的交集   wl * r_p加权的并集
        # Dice = 2 * (weighted intersection) / (weighted union)   重叠度指标，取值[0,1]
        # Dice损失（最小化）= 1 - Dice
        loss = 1 - 2 * (wl * rp).sum() / (wl * r_p + self.epsilon).sum()
        return loss


def Sigmoid_preds(preds):
    preds_sigmoid = F.sigmoid(preds)
    return preds_sigmoid

# 计算dice系数：衡量预测值与真实值之间的重叠度，值越接近1重叠度越好
def dice_coeff(preds, targets):
    """Dice coeff for batches"""
    import torch
    predbinary = torch.round(preds)
    loss = dice_loss(predbinary, targets)
    return loss


def dice_loss(preds, trues):
    eps = 1e-6
    # 确保pred和true连续
    preds = preds.contiguous()
    trues = trues.contiguous()
    # 获取size第一位即批次大小 batchsize
    num = preds.size()[0]
    # 将预测和真实值展平为二维数组，形状为[num, h * w]
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    # 逐元素相乘，得到 预测与真实标签的交集
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    # 计算平均dice系数
    score = scores.sum() / num
    # 将dice系数限制在[0,1]范围内
    score = torch.clamp(score, 0., 1.)
    return score


def dice_loss_sum(preds, trues):
    eps = 1e-6
    #
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size()[0]
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)

    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    score = scores.sum() / num
    score = torch.clamp(score, 0., 1.)
    # 返回加权dice损失和批次大小num
    return score * num, num

# 针对不平衡数据集，引入一个调节因子
class FocalLoss2d(nn.Module):
    # gamma值越高，模型更关注难分类的样本，默认值为2
    def __init__(self, gamma=2):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma

    def forward(self, preds, target):
        # 将真实标签从二维张量转换为一维，long类型
        target = target.view(-1, 1).long()

        # 类别权重，此处默认相等
        class_weight = [1] * 2  # [0.5, 0.5]
        prob = F.sigmoid(preds)
        prob = prob.view(-1, 1)
        # 将预测结果拼接，得到每个样本属于背景类和目标类的概率。
        # 1 - prob 表示样本属于背景类的概率，prob 表示样本属于目标类的概率。此时，prob 的形状变为 [batch_size, 2]，每个样本包含两个概率值（背景类和目标类）
        prob = torch.cat((1 - prob, prob), 1)
        # len(prob) 批次大小
        select = torch.FloatTensor(len(prob), 2).zero_().cuda()
        select.scatter_(1, target, 1.)

        # 为每个类别分配权重
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)
        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        batch_loss = - class_weight * (torch.pow((1 - prob), self.gamma)) * prob.log()
        loss = batch_loss.mean()
        return loss


def ComputeDSC2D(preds, targets):

    eps = 1e-6
    preds = preds.contiguous()
    targets = targets.contiguous()

    # num批次大小
    num = preds.size()[0]
    preds = preds.view(num, -1)
    targets = targets.view(num, -1)

    intersection = (preds * targets).sum(1)
    pred_sum = preds.sum(1)
    target_sum = targets.sum(1)

    # 获取批次中样本的数量
    nSlices = intersection.shape[0]
    dice_sum = 0
    dice_num = 0
    for ni in range(nSlices):
        # 转为浮动型
        finter = float(intersection[ni].item())
        fpred = float(pred_sum[ni].item())
        ftargets = float(target_sum[ni].item())
        # 如果预测和真实值像素都为0，跳过计算
        if fpred + ftargets > 0:
            dice_tmp = (2. * finter + eps) / (fpred + ftargets + eps)
            dice_sum += dice_tmp
            dice_num += 1

    return dice_sum, dice_num

# 计算多类别交叉熵损失
def multi_class_bce(preds, trues):
    bceAll = []
    for idx in range(0, preds.shape[1]):
        # 计算单个类别的二进制交叉熵损失
        bceAll.append(F.binary_cross_entropy_with_logits(preds[:, idx, ...].contiguous(), trues[:, idx, ...].contiguous()))
    return sum(bceAll) / len(bceAll)

# 计算多类别Dice损失
# preds b,c,W,H    trues:b,c,W,H
def multi_class_dice(preds, trues):
    diceAll = []
    for idx in range(0, preds.shape[1]):
        diceAll.append(dice_loss(preds[:, idx, ...].contiguous(), trues[:, idx, ...].contiguous()))
    return sum(diceAll) / len(diceAll)

# 用sigmoid函数转换为概率值后，计算多类别Dice损失
# preds b,c,W,H    trues:b,c,W,H
def multi_class_sigmoid_dice(preds, trues):
    diceAll = []
    preds_sigmoid = F.sigmoid(preds)
    for idx in range(0, preds.shape[1]):
        diceAll.append(dice_loss(preds_sigmoid[:, idx, ...].contiguous(), trues[:, idx, ...].contiguous()))
    return sum(diceAll) / len(diceAll)

# 结合了 Focal Loss 和 Dice Loss，用于不平衡数据集的图像分割
class HaNLoss(nn.Module):

    def __init__(self):
        super(HaNLoss, self).__init__()

    def forward(self, preds, targets):
        output = F.sigmoid(preds)
        dice = dice_loss(output, targets)

        cfl = FocalLoss2d()
        # fl：计算得到的Focal Loss
        fl = cfl(output, targets)
        # HaNLoss： Focal Loss 和 Dice 损失的对数 相减
        loss = fl - torch.log(dice)
        return loss


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, preds, targets):
        # 计算BCE时的pred是未经过sigmoid激活的模型输出
        # multi_class_bce函数直接接受logits作为输入，不需要先经过sigmoid激活
        ce = multi_class_bce(preds, targets)
        output = torch.sigmoid(preds)
        dice = multi_class_dice(output, targets)

        loss = 0.5 * ce + 0.5 * (1.0 - dice)
        return loss


class BCE1Loss(nn.Module):

    def __init__(self):
        super(BCE1Loss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, preds, targets):

        loss = multi_class_bce(preds, targets)
        return loss


class DiceLoss1(nn.Module):

    def __init__(self):
        super(DiceLoss1, self).__init__()

    def forward(self, preds, targets):

        output = F.sigmoid(preds)
        dice = multi_class_dice(output, targets)
        loss = (1.0 - dice)
        return loss

class Tckersky_loss(nn.Module):
    def __init__(self):
        super(Tckersky_loss,self).__init__()

    def forward(self, inputs,targets,beta=0.7,weights=None):
        batch_size = targets.size(0)
        loss = 0.0

        for i in range(batch_size):
            prob = inputs[i]   # 预测结果
            ref = targets[i]

            # alpha 调节假阳性FP
            alpha = 1.0 - beta
            # tp 真正例 即预测为目标类且真实值也为目标类的像素数量
            tp = (ref * prob).sum()
            # fp 假阳性
            fp = ((1 - ref) * prob).sum()
            # fn 假阴性
            fn = (ref * (1 - prob)).sum()
            # tversky指数越大，模型表现越好
            tversky = tp / (tp + alpha * fp + beta * fn)
            loss = loss + (1 - tversky)
        return loss / batch_size

# def tversky_loss(inputs, targets, beta=0.7, weights=None):
#     batch_size = targets.size(0)
#     loss = 0.0
#
#     for i in range(batch_size):
#         prob = inputs[i]
#         ref = targets[i]
#
#         alpha = 1.0-beta
#
#         tp = (ref*prob).sum()
#         fp = ((1-ref)*prob).sum()
#         fn = (ref*(1-prob)).sum()
#         tversky = tp/(tp + alpha*fp + beta*fn)
#         loss = loss + (1-tversky)
#     return loss/batch_size




