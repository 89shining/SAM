# -*- coding: utf-8 -*-
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import pandas as pd
from Mycodes.dice_loss import BCEDiceLoss
from Mydataset import SAMDataset
from segment_anything import sam_model_registry
from tensorboardX import SummaryWriter
import logging
import torch.nn.functional as F


# tensorboard训练日志保存目录：./runs/exp1
writer=SummaryWriter('../runs/exp1')

# 训练网络
def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp=True,
              root_dir: str=None,
              csv_path: str=None,
              save_dir: str=None,
              target_size=(1024, 1024),
              num_pos: int=None,
              num_neg: int=None):

    # 数据集划分
    dataset = SAMDataset(csv_path, root_dir, target_size, num_pos, num_neg)
    ntotal = len(dataset)
    n_val = int(ntotal * val_percent)
    n_train = ntotal - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    # 加载CSV，注意 header=None（因为你自己在SAMDataset里也是这么读取的）
    csv_data = pd.read_csv(csv_path, header=None, names=["image", "mask"])

    # 提取所有图像路径
    all_image_paths = csv_data["image"].tolist()

    # 提取训练集ID
    train_ids = []
    for idx in train_dataset.indices:
        image_path = all_image_paths[idx]  # eg: /images/p_0/32.tiff
        id = image_path.replace('/images/', '').replace('.tiff', '').replace('.tif', '')
        train_ids.append(id)

    # 提取验证集ID
    val_ids = []
    for idx in val_dataset.indices:
        image_path = all_image_paths[idx]
        id = image_path.replace('/images/', '').replace('.tiff', '').replace('.tif', '')
        val_ids.append(id)

    # 保存到txt
    train_id_path = os.path.join(save_dir, 'train_ids.txt')
    val_id_path = os.path.join(save_dir, 'val_ids.txt')

    with open(train_id_path, 'w') as f:
        for id in train_ids:
            f.write(f"{id}\n")
    with open(val_id_path, 'w') as f:
        for id in val_ids:
            f.write(f"{id}\n")

    # 同时保存到日志文件
    logging.info(f"Train IDs ({len(train_ids)} samples): {train_ids}")
    logging.info(f"Val IDs ({len(val_ids)} samples): {val_ids}")

    # 数据集加载
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 日志文件和模型的储存地址
    dir_checkpoint = save_dir + '/checkpoint'  # 保存训练过程的模型检查点
    os.makedirs(dir_checkpoint, exist_ok=True)  # exist_ok=True 目录存在时不报错，目录不存在时创建目录
    os.makedirs(save_dir + '/logs', exist_ok=True)  # 保存训练过程中的日志文件
    save_path = save_dir + '/weights'  # 保存模型权重文件
    os.makedirs(save_path, exist_ok=True)

    # 日志信息
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp} 
        Device:          {device.type}
    ''')


    #损失函数
    #  BCEDiceLoss：二元交叉熵损失（BCE） 和 Dice系数损失 的复合损失函数
    criterion = BCEDiceLoss()

    # 学习率
    scalelr = lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=scalelr, weight_decay=1e-8)
    """
    学习率调度器ReduceLROnPlateau：根据验证损失动态调整学习率
    optimizer：训练优化器
    多分类任务最小化验证损失，二分类任务最大化损失指标
    patience=2，监控指标在连续两个epoch内没有改善时，降低学习率
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    # 每个epoch的平均损失，每个的验证集评分，，每一轮的验证集损失
    trainLoss = []
    valLoss = []
    bestloss = 1e10  # 初始化成很大很大的值

    for epoch in range(epochs):
        print(f"\n{'-' * 10} Epoch {epoch + 1}/{epochs} {'-' * 10}")
        net.train()    # 训练模式

        train_epoch_loss = 0    # 当前epoch的总损失

        LOSS =[]     # 保存当前epoch的每个batch的损失值

        train_n_loss = 0   # 处理的batch计数
        # 进度条和信息（总batch数=训练数/批次大小，字符串提示：当前epoch/总epoch，进度条单位）
        # batch_size+1：batch为小数时将最后一个进度条补上
        with tqdm(total=len(train_loader), desc='[Train]', unit='batch', disable=False) as pbar:
            # 传入一个batch
            for image, mask, point_coords, point_labels, box, mask_input in train_loader:
                imgs = image.to(device).float()
                # print(imgs.shape)
                # print(len(imgs))
                true_masks = mask.to(device).float()
                # print(true_masks.shape)
                points_coords = point_coords.to(device).float()
                # print(points_coords.shape)
                points_labels = point_labels.to(device).long()
                # print(points_labels.shape)
                bbox = box.to(device).float()
                # print(bbox.shape)
                mask_inputs = mask_input.to(device).float()
                # print(mask_inputs.shape)

                # 手动传入
                input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                image_embeddings = net.image_encoder(input_images)

                logits_list = []
                for i in range(len(imgs)):
                    sparse_embeddings, dense_embeddings = net.prompt_encoder(
                        points=(points_coords[i].unsqueeze(0), points_labels[i].unsqueeze(0)),
                        boxes=bbox[i].unsqueeze(0),
                        masks=mask_inputs
                    )
                    low_res_masks, _ = net.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    logits_list.append(low_res_masks)

                masks_pred = torch.stack([x.squeeze(0) for x in logits_list], dim=0)
                # print(masks_pred.shape)
                masks_pred = masks_pred.to(device)

                if true_masks.dim() == 3:
                    true_masks = true_masks.unsqueeze(1)

                true_masks = F.interpolate(true_masks, size=masks_pred.shape[-2:], mode='bilinear', align_corners=False)
                train_loss = criterion(masks_pred, true_masks)
                # 返回当前batch的loss
                train_loss_batch = float(train_loss.item())
                # 当前epoch总loss
                train_epoch_loss += train_loss_batch
                train_n_loss += 1
                #当前batch的loss反向传播
                optimizer.zero_grad()
                train_loss.backward()

                # 梯度裁剪：将梯度值限制在某个指定的范围内，防止梯度值过大导致训练不稳定
                # clip_value：梯度剪裁阈值，使梯度最大绝对值不超过0.1
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # 优化器参数更新
                optimizer.step()
                # 更新进度条右侧的附加信息:当前epoch的平均loss dice bce
                pbar.set_postfix({'TrainLoss': f"{train_epoch_loss / train_n_loss:.4f}"})
                # 更新进度条（进度条前进1步）
                pbar.update(1)

        train_meanLoss = train_epoch_loss / train_n_loss    # 当前epoch每个batch的平均损失
        LOSS.append(train_meanLoss)       # LOSS列表保存平均损失
        trainLoss.append(LOSS[-1])
        writer.add_scalar('Loss/train_epoch_avg', train_meanLoss, epoch + 1)

        # 验证阶段
        net.eval()
        val_epoch_loss = 0  # 当前epoch的总损失
        LOSS = []  # 保存当前epoch的每个batch的损失值
        val_n_loss = 0  # 处理的batch计数
        # 进度条和信息（总batch数=训练数/批次大小，字符串提示：当前epoch/总epoch，进度条单位）
        with torch.no_grad():
            # batch_size+1：batch为小数时将最后一个进度条补上
            with tqdm(total=len(val_loader), desc='[Val]', unit='batch', disable=False) as pbar:
                # 传入一个batch
                for image, mask, point_coords, point_labels, box, mask_input in val_loader:
                    imgs = image.to(device).float()
                    # print(imgs.shape)
                    # print(len(imgs))
                    true_masks = mask.to(device).float()
                    # print(true_masks.shape)
                    points_coords = point_coords.to(device).float()
                    # print(points_coords.shape)
                    points_labels = point_labels.to(device).float()
                    # print(points_labels.shape)
                    bbox = box.to(device).float()
                    # print(bbox.shape)
                    mask_inputs = mask_input.to(device).float()
                    # print(mask_inputs.shape)

                    # 手动传入
                    input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                    image_embeddings = net.image_encoder(input_images)

                    logits_list = []
                    for i in range(len(imgs)):
                        sparse_embeddings, dense_embeddings = net.prompt_encoder(
                            points=(points_coords[i].unsqueeze(0), points_labels[i].unsqueeze(0)),
                            boxes=bbox[i].unsqueeze(0),
                            masks=mask_inputs
                        )
                        low_res_masks, _ = net.mask_decoder(
                            image_embeddings=image_embeddings[i].unsqueeze(0),
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        logits_list.append(low_res_masks)

                    masks_pred = torch.stack([x.squeeze(0) for x in logits_list], dim=0)
                    # print(masks_pred.shape)
                    masks_pred = masks_pred.to(device)

                    if true_masks.dim() == 3:
                        true_masks = true_masks.unsqueeze(1)

                    true_masks = F.interpolate(true_masks, size=masks_pred.shape[-2:], mode='bilinear', align_corners=False)
                    val_loss = criterion(masks_pred, true_masks)

                    # 返回当前batch的loss
                    val_loss_batch = float(val_loss.item())
                    # 当前epoch总loss
                    val_epoch_loss += val_loss_batch
                    # print("epoch_loss")
                    # print(epoch_loss)
                    val_n_loss += 1

                    # 更新进度条右侧的附加信息:当前epoch的平均loss
                    pbar.set_postfix({'ValLoss': f"{val_epoch_loss / val_n_loss:.4f}"})
                    # 更新进度条（进度条前进1步）
                    pbar.update(1)

        val_meanLoss = val_epoch_loss / val_n_loss  # 当前epoch每个batch的平均损失
        LOSS.append(val_meanLoss)  # LOSS列表保存平均损失
        valLoss.append(LOSS[-1])
        writer.add_scalar('Loss/Val_epoch_avg', val_meanLoss, epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
            f'Epoch {epoch + 1}: Train Loss={trainLoss[-1]:.4f}, Val Loss={valLoss[-1]:.4f}, lr={current_lr:.8f}')

        # 调整学习率
        scheduler.step(val_meanLoss)        # 根据验证集每个epoch平均损失来调整学习率

        if bestloss > val_meanLoss:
            bestloss = val_meanLoss
            torch.save(net.state_dict(),
                       save_path + f'/best.pth')    # 将最优模型权重保存为best.pth
            logging.info(f'Best model updated with loss={bestloss:.4f}')

        if save_cp and epoch % 1 == 0:       # 每10次保存一次模型权重
            torch.save(net.state_dict(),
                       dir_checkpoint + f'/CP_epoch{epoch + 1}.pth')
            # 在日志文件中输出一条保存信息
            logging.info(f'Checkpoint {epoch + 1} saved !')

    #绘图部分
    import matplotlib.pyplot as plt
    x1 = range(0,epochs)     # x1横坐标范围0——epochs-1
    x2 = range(0,epochs)
    y1 = trainLoss           # y1纵坐标：训练平均损失
    y2 = valLoss             # y3纵坐标：验证损失

    # 绘制损失图1
    plt.figure(1)
    plt.subplot(1, 1, 1)     # 指定图中每个子图的位置
    plt.plot(x1, y1, 'o-')   # 使用原点-连接线绘制数据
    plt.title("Train Loss")              # 图片名
    plt.ylabel("train loss")
    plt.xlabel("epoch")
    plt.savefig(save_dir + "/trainloss.jpg")     # 将图像存在指定路径

    plt.figure(2)
    plt.subplot(1, 1, 1)
    plt.plot(x2, y2, 'o-')
    plt.title("Val Loss")
    plt.ylabel("val loss")
    plt.xlabel("epoch")
    plt.savefig(save_dir + "/valloss.jpg")

    plt.show()
    writer.close()
    print("Training completed.")

if __name__ == '__main__':
    # 配置日志保存
    saveDir = "C:/Users/dell/Desktop/task/results"
    log_path = os.path.join(saveDir, 'logs', 'train.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 训练设备：gpu 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    val_percent = 0.2
    save_cp = True
    trainDir = 'C:/Users/dell/Desktop/task'
    trainFile = 'C:/Users/dell/Desktop/task/train_tiff.csv'

    BatchSize = 1
    nEpochs = 5
    flr = 0.0001

    # 加载模型
    sam_checkpoint = "D:\project\segment-anything\demo\configs\checkpoint\sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    net = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    net.to(device=device)

    # 开始训练
    try:
        train_net(net=net,
                  device=device,
                  epochs=nEpochs,
                  batch_size=BatchSize,
                  lr=flr,
                  val_percent=val_percent,
                  save_cp=True,
                  root_dir=trainDir,
                  csv_path=trainFile,
                  save_dir=saveDir,
                  target_size=(1024, 1024),
                  num_pos=2,
                  num_neg=2
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), '../INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


