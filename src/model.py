import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = [
    "maize_northern_blight", "maize_common_rust", "maize_healthy",
    "tomato_early_blight", "tomato_late_blight", "tomato_leaf_mold", "tomato_healthy",
    "potato_early_blight", "potato_late_blight", "potato_healthy",
]


class DepthwiseSepConv(nn.Module):
    """Depthwise-separable conv — the MobileNet-style efficiency trick."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        return F.relu(self.bn2(self.pw(x)), inplace=True)


class VisualBackbone(nn.Module):
    """AgriYouthNet backbone: 3 conv blocks -> GAP -> 128-d feature."""

    def __init__(self, feat=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = DepthwiseSepConv(32, 64, stride=2)
        self.conv3 = DepthwiseSepConv(64, feat, stride=2)
        self.gap   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.gap(x).flatten(1)


class ContextEncoder(nn.Module):
    """Tiny MLP that turns the 4-dim farmer context into a 32-dim token."""

    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class MHCAFusion(nn.Module):
    """Multi-Head Cross-Attention fusion (visual Q, context K/V)."""

    def __init__(self, visual_dim=128, ctx_dim=32, n_heads=4, n_tokens=4):
        super().__init__()
        assert visual_dim % n_heads == 0
        self.h     = n_heads
        self.t     = n_tokens
        self.d     = visual_dim // n_heads
        self.d_tok = visual_dim // n_tokens

        self.ctx_proj = nn.Linear(ctx_dim, visual_dim)
        self.q   = nn.Linear(self.d_tok, n_heads * self.d)
        self.k   = nn.Linear(self.d_tok, n_heads * self.d)
        self.v   = nn.Linear(self.d_tok, n_heads * self.d)
        self.out = nn.Linear(n_heads * self.d, self.d_tok)
        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, vis, ctx):
        B = vis.size(0)
        v_tok = vis.view(B, self.t, self.d_tok)
        c_tok = self.ctx_proj(ctx).view(B, self.t, self.d_tok)

        q = self.q(v_tok).view(B, self.t, self.h, self.d).transpose(1, 2)
        k = self.k(c_tok).view(B, self.t, self.h, self.d).transpose(1, 2)
        v = self.v(c_tok).view(B, self.t, self.h, self.d).transpose(1, 2)

        att = torch.softmax(q @ k.transpose(-2, -1) / (self.d ** 0.5), dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, self.t, self.h * self.d)
        out = self.out(out).reshape(B, -1)
        return self.norm(vis + out)


class AgriYouthNetContextPlus(nn.Module):
    """Full multimodal classifier. fusion in {attention, concat, gating}."""

    def __init__(self, num_classes=10, ctx_dim=4, fusion="attention", n_heads=4):
        super().__init__()
        self.backbone = VisualBackbone(feat=128)
        self.ctx_enc  = ContextEncoder(ctx_dim, hidden=32)
        self.fusion   = fusion

        if fusion == "attention":
            self.mhca = MHCAFusion(visual_dim=128, ctx_dim=32, n_heads=n_heads)
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(inplace=True),
                nn.Dropout(0.1), nn.Linear(64, num_classes))
        elif fusion == "concat":
            self.head = nn.Sequential(
                nn.Linear(128 + 32, 64), nn.ReLU(inplace=True),
                nn.Dropout(0.1), nn.Linear(64, num_classes))
        elif fusion == "gating":
            self.gate = nn.Sequential(
                nn.Linear(128 + 32, 128), nn.Sigmoid())
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(inplace=True),
                nn.Dropout(0.1), nn.Linear(64, num_classes))
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def forward(self, img, ctx):
        v = self.backbone(img)
        c = self.ctx_enc(ctx)
        if self.fusion == "attention":
            h = self.mhca(v, c)
        elif self.fusion == "concat":
            h = torch.cat([v, c], dim=1)
        elif self.fusion == "gating":
            g = self.gate(torch.cat([v, c], dim=1))
            h = g * v
        return self.head(h)
