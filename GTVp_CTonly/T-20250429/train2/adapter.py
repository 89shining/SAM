import torch.nn as nn

class Adapter_Layer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        shortcut = x      # [B,H,W,C]
        x = self.bottleneck(x)
        # 跳跃连接
        x = shortcut + x
        return x