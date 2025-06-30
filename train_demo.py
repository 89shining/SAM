import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from segment_anything import sam_model_registry
from dataset.dataset_box import CustomDataset
from Mycodes.dice_loss import BCEDiceLoss
import gc


def SAMtrain(weightdir, root_dir, csv_path, num_epochs, batchn):
    # 加载模型
    sam_checkpoint = weightdir + "/pretrain_model/sam_vit_b_01ec64.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = torch.nn.DataParallel(sam)  # 启用多卡并行训练
    sam.to(device=device)

    # 创建数据集和数据加载器
    dataset = CustomDataset(csv_path, root_dir, point_num=8, target_size=(1024, 1024))
    ntotal = len(dataset)
    n_val = int(ntotal * 0.2)
    n_train = ntotal - n_val
    mtrain, mval = random_split(dataset, [n_train, n_val])
    # 定义损失函数和优化器
    bce_criterion = nn.BCELoss()
    optimizer = optim.Adam(sam.parameters(), lr=0.001)
    best_loss = 1000
    train_losses = []
    val_losses = []
    valid_freq = 5

    for epoch in range(num_epochs):
        #mtrain, mval = random_split(dataset, [n_train, n_val])
        trainloader = DataLoader(mtrain, batch_size=batchn, shuffle=True)
        #print(len(trainloader))

        epoch_loss = 0
        sam.train()
        for images, points, labels, bboxes, low_res_mask, gts in trainloader:
            images = images.to(device).float()
            labels = labels.to(device).float()
            points = points.to(device).float()
            bboxes = bboxes.to(device).float()
            low_res_mask = low_res_mask.to(device).float()
            gts = gts.to(device).float()

            optimizer.zero_grad()  # 在每个批次开始时清零梯度

            # 第一次前向传播和损失计算
            outputs = sam([{"image": images, "point_coords": points, "point_labels": labels}])
            preds = outputs[0]["low_res_logits"]
            loss1 = BCEDiceLoss()(preds, gts)
            loss1.backward()  # 反向传播梯度
            optimizer.step()  # 更新模型参数

            # 第二次前向传播和损失计算
            outputs = sam([{"image": images, "boxes": bboxes}])
            preds = outputs[0]["low_res_logits"]
            loss2 = BCEDiceLoss()(preds, gts)
            loss2.backward()  # 反向传播梯度
            optimizer.step()  # 更新模型参数

            # 第三次前向传播和损失计算
            outputs = sam([{"image": images, "mask_inputs": low_res_mask}])
            preds = outputs[0]["low_res_logits"]
            loss3 = BCEDiceLoss()(preds, gts)
            loss3.backward()  # 反向传播梯度
            optimizer.step()  # 更新模型参数

            # 平均损失
            loss_mean = (loss1 + loss2 + loss3)/3
            epoch_loss += loss_mean.item()  # 每个epoch的总损失

        avg_loss = epoch_loss / len(trainloader)   # 每个epoch的平均损失
        train_losses.append(avg_loss)
        print(f"Train Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

        # 保存最后模型
        torch.save(sam.state_dict(), "../sam_epoch_last.pth")
        gc.collect()
        torch.cuda.empty_cache()

        if epoch % valid_freq != 0:
            continue

        # 验证阶段
        epoch_loss = 0
        sam.eval()
        valloader = DataLoader(mval, batch_size=batchn, shuffle=False)
        #print(len(valloader))
        with torch.no_grad():
            for images, points, labels, bboxes, low_res_mask, gts in valloader:
                images = images.to(device).float()
                labels = labels.to(device).float()
                points = points.to(device).float()
                bboxes = bboxes.to(device).float()
                low_res_mask = low_res_mask.to(device).float()
                gts = gts.to(device).float()

                outputs = sam([{"image": images, "point_coords": points, "point_labels": labels}])
                preds = outputs[0]["low_res_logits"]
                dice1 = BCEDiceLoss()(preds, gts)  # 256

                outputs = sam([{"image": images, "boxes": bboxes}])
                preds = outputs[0]["low_res_logits"]
                dice2 = BCEDiceLoss()(preds, gts)

                outputs = sam([{"image": images, "mask_inputs": low_res_mask}])
                preds = outputs[0]["low_res_logits"]
                dice3 = BCEDiceLoss()(preds, gts)
                #print(gts)
                loss_mean = (dice1 + dice2 + dice3)/3
                #print(loss_mean)
                epoch_loss += loss_mean.item()


            avg_loss = epoch_loss / len(valloader)  # 每个epoch的平均验证损失
            val_losses.append(avg_loss)
            print(f"Val Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

            # 保存最佳模型
            if best_loss > avg_loss:
                best_loss = avg_loss
                torch.save(sam.state_dict(), "../sam_best.pth")


    print("Training completed.")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

    weightdir = "/data1/liuyunhai/lyhSAM/segment-anything-main"
    csv_path = '/data1/liuyunhai/lyhSAM/traindata/train.csv'
    root_dir = '/data1/liuyunhai/lyhSAM/traindata'
    num_epochs = 300
    batch_size = 64

    SAMtrain(weightdir=weightdir, root_dir=root_dir, csv_path=csv_path, num_epochs=num_epochs, batchn=batch_size)