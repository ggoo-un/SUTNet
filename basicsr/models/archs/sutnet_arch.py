import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import numpy as np

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, bias, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 获取窗口内每个 token 的成对的相对位置索引
        coords_h = torch.arange(self.window_size[0])  # 高维度上的坐标 (0, 7)
        coords_w = torch.arange(self.window_size[1])  # 宽维度上的坐标 (0, 7)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 坐标，结构为 [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # 重构张量结构为 [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 相对坐标，结构为 [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 交换维度，结构为 [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 第1个维度移位
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 第1个维度移位
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 第1个维度的值乘以 2倍的 Ww，再减 1
        relative_position_index = relative_coords.sum(-1)  # 相对位置索引，结构为 [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)  # 保存数据，不再更新

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # nW*B, window_size, window_size, C
        x = x.permute(0, 3, 1, 2)
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)
        # attn -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(b // nW, nW, self.num_heads, mask.shape[1], mask.shape[1]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, mask.shape[1], mask.shape[1])
            # attn -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        out = (attn @ v)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.proj_drop(out)

        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


##########################################################################
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, ffn_expansion_factor, bias, LayerNorm_type, drop=0.,
                 attn_drop=0., drop_path=0.):
        super(SwinTransformerBlock, self).__init__()

        self.window_size = window_size
        self.shift_size = shift_size

        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = WindowAttention(dim, to_2tuple(self.window_size), num_heads, bias, attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        # if self.shift_size > 0:
        #     attn_mask = self.calculate_mask(to_2tuple(self.input_resolution))
        # else:
        #     attn_mask = None
        #
        # self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, attn_mask):
        b, c, h, w = x.shape

        shortcut = x
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), 'reflect')
        _, _, hp, wp = x.shape

        x = x.permute(0, 2, 3, 1)  # b,h,w,c

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, C, window_size, window_size

        # merge windows
        attn_windows = attn_windows.permute(0, 2, 3, 1)
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c) # [-1, window_size, window_size, C]
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # B H' W' C

        # 逆向循环移位
        if self.shift_size > 0:
            # 第 0 维度下移 shift_size 位，第 1 维度右移 shift_size 位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x  # 不逆向移位

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :h, :w, :].contiguous()

        x = x.permute(0, 3, 1, 2)  # b,c,h,w

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, window_size, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else self.shift_size,
                                 ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks)
        ])

        # if self.sampling == 'downsample':
        #     self.downsample = Downsample(dim)
        #
        # elif self.sampling == 'upsample':
        #     self.upsample = Downsample(dim)

        # 计算注意力 mask

    def calculate_mask(self, x, x_size):
        # H, W 特征窗口的高宽
        H, W = x_size
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 新建张量，结构为 [1, H, W, 1]
        # 以下两 slices 中的数据是索引，具体缘由尚未搞懂
        h_slices = (slice(0, -self.window_size),  # 索引 0 到索引倒数第 window_size
                    slice(-self.window_size, - self.shift_size),  # 索引倒数第 window_size 到索引倒数第 shift_size
                    slice(-self.shift_size, None))  # 索引倒数第 shift_size 后所有索引
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt  # 将 img_mask 中 h, w 对应索引范围的值置为 cnt
                cnt += 1  # 加 1

        mask_windows = window_partition(img_mask, self.window_size)  # 窗口分割，返回值结构为 [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1,self.window_size * self.window_size)  # 重构结构为二维张量 [nW, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1] # 增加第 2 维度减去增加第 3 维度的注意力 mask
        # [nW, Mh*Mw, Mh*Mw]
        # 用浮点数 -100. 填充注意力 mask 中值不为 0 的元素，再用浮点数 0. 填充注意力 mask 中值为 0 的元素
        # 每一个窗口区域中对各个patch位置的判断
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        attn_mask = self.calculate_mask(x, x_size)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            x = blk(x, attn_mask)

        # if self.sampling == 'downsample':
        #     x = self.downsample(x, H, W)
        #     H, W = (H + 1) // 2, (W + 1) // 2
        #
        # elif self.sampling == 'upsample':
        #     x = self.downsample(x, H, W)
        #     H, W = (H + 1) // 2, (W + 1) // 2

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))  # 채널 두배 확장, 사이즈 절반 줄어듬

    def forward(self, x):
        return self.body(x)


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = LayerNorm(4 * dim, 'BiasFree')

    def forward(self, x):
        """
        x: B, C, H, W
        """
        b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, f"x size ({h}*{w}) are not even."

        x = x.permute(0, 2, 3, 1)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = x.permute(0, 3, 1, 2)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- SUTNet -----------------------
class SUTNet(nn.Module):
    def __init__(self,
                 img_size=192,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 window_size=8,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(SUTNet, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = BasicLayer(dim=dim, num_heads=heads[0], num_blocks=num_blocks[0], window_size=window_size,
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type)

        self.down1_2 = PatchMerging(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = BasicLayer(dim=int(dim * 2 ** 1), num_heads=heads[1], num_blocks=num_blocks[1],
                                         window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type)

        self.down2_3 = PatchMerging(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = BasicLayer(dim=int(dim * 2 ** 2), num_heads=heads[2], num_blocks=num_blocks[2],
                                         window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type)

        self.down3_4 = PatchMerging(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = BasicLayer(dim=int(dim * 2 ** 3), num_heads=heads[3], num_blocks=num_blocks[3],
                                 window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                 LayerNorm_type=LayerNorm_type)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = BasicLayer(dim=int(dim * 2 ** 2), num_heads=heads[2], num_blocks=num_blocks[2],
                                         window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = BasicLayer(dim=int(dim * 2 ** 1), num_heads=heads[1], num_blocks=num_blocks[1],
                                         window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BasicLayer(dim=int(dim * 2 ** 1), num_heads=heads[0], num_blocks=num_blocks[0],
                                         window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type)

        self.refinement = BasicLayer(dim=int(dim * 2 ** 1), num_heads=heads[0], num_blocks=num_refinement_blocks,
                                     window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                     LayerNorm_type=LayerNorm_type)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, inp_enc_level1.shape[2:])

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, inp_enc_level2.shape[2:])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, inp_enc_level3.shape[2:])

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4, inp_enc_level4.shape[2:])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, inp_dec_level3.shape[2:])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, inp_dec_level2.shape[2:])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, inp_dec_level1.shape[2:])

        out_dec_level1 = self.refinement(out_dec_level1, inp_dec_level1.shape[2:])

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == '__main__':
    height = 128
    width = 192
    x = torch.randn((1, 3, height, width))  # .cuda()
    model = SUTNet(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',  ## Other option 'BiasFree'
        dual_pixel_task=False)  # .cuda()
    print(model)
    # print(height, width, model.flops() / 1e9)

    # x = torch.randn((1, 3, 50, 52))
    x = model(x)
    print(x.shape)


