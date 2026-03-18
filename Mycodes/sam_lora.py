import math
import torch
import torch.nn as nn


class LoRA_qkv(nn.Module):
    """
    给原始 qkv 线性层加 LoRA，只作用于 Q 和 V。
    原始 qkv: Linear(dim, 3*dim)
    输入: [B, H, W, C]
    输出: [B, H, W, 3*C]
    """
    def __init__(self, qkv: nn.Linear, dim: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.qkv = qkv
        self.dim = dim
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Q branch
        self.lora_q_A = nn.Linear(dim, r, bias=False)
        self.lora_q_B = nn.Linear(r, dim, bias=False)

        # V branch
        self.lora_v_A = nn.Linear(dim, r, bias=False)
        self.lora_v_B = nn.Linear(r, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)

        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始 qkv 输出
        qkv = self.qkv(x)   # [B, H, W, 3*dim]

        # LoRA 增量
        delta_q = self.lora_q_B(self.lora_q_A(x)) * self.scaling   # [B,H,W,dim]
        delta_v = self.lora_v_B(self.lora_v_A(x)) * self.scaling   # [B,H,W,dim]

        # qkv布局: [q | k | v]
        qkv[..., :self.dim] = qkv[..., :self.dim] + delta_q
        qkv[..., -self.dim:] = qkv[..., -self.dim:] + delta_v

        return qkv


def inject_lora_to_sam_image_encoder(sam_model, r=4, alpha=8):
    """
    将 SAM image encoder 中每个 block 的 attn.qkv 替换为 LoRA_qkv
    """
    for blk in sam_model.image_encoder.blocks:
        old_qkv = blk.attn.qkv
        dim = old_qkv.in_features
        blk.attn.qkv = LoRA_qkv(old_qkv, dim=dim, r=r, alpha=alpha)
    return sam_model


def freeze_image_encoder_except_lora(sam_model):
    """
    冻结整个 image encoder，再只打开 LoRA 参数
    """
    for p in sam_model.image_encoder.parameters():
        p.requires_grad = False

    for blk in sam_model.image_encoder.blocks:
        for name, p in blk.attn.qkv.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

# 不冻结图像编码器
def unfreeze_image_encoder(sam_model):
    for p in sam_model.image_encoder.parameters():
        p.requires_grad = True

# 只对最后n个block（+ lora）训练
def unfreeze_image_encoder_last_n_blocks(sam_model, n=6):
    blocks = sam_model.image_encoder.blocks
    total_blocks = len(blocks)

    # 默认全冻结
    for p in sam_model.image_encoder.parameters():
        p.requires_grad = False

    # 打开最后 n 个 block
    for blk in blocks[total_blocks - n:]:
        for p in blk.parameters():
            p.requires_grad = True

def unfreeze_mask_decoder(sam_model):
    for p in sam_model.mask_decoder.parameters():
        p.requires_grad = True

def freeze_mask_decoder(sam_model):
    for p in sam_model.mask_decoder.parameters():
        p.requires_grad = False

def unfreeze_prompt_encoder(sam_model):
    for p in sam_model.prompt_encoder.parameters():
        p.requires_grad = True

def freeze_prompt_encoder(sam_model):
    for p in sam_model.prompt_encoder.parameters():
        p.requires_grad = False

def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total}")
    print(f"Trainable params: {trainable}")
    print(f"Trainable ratio: {trainable / total:.4%}")