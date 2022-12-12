import functools

import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as fc
from torch.nn.utils import spectral_norm as SpectralNorm
from .attention import spectral_norm, MultiHeadAttention


# https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment-stylegan2-pytorch/DiffAugment_pytorch.py

def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.1):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = fc.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.3):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


# Discriminator

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, stride_size=4, emb_size=384, image_size=32, batch_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            spectral_norm(nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=stride_size)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = ((image_size - patch_size + stride_size) // stride_size) ** 2 + 1
        self.positions = nn.Parameter(torch.randn(num_patches, emb_size))
        self.batch_size = batch_size

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += torch.sin(self.positions)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class DiscriminatorTransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=384,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                nn.Sequential(
                    spectral_norm(nn.Linear(emb_size, forward_expansion * emb_size)),
                    nn.GELU(),
                    nn.Dropout(forward_drop_p),
                    spectral_norm(nn.Linear(forward_expansion * emb_size, emb_size)),
                ),
                nn.Dropout(drop_p)
            )
            ))


class DiscriminatorTransformerEncoder(nn.Sequential):
    def __init__(self, depth=4, **kwargs):
        super().__init__(*[DiscriminatorTransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=384, class_size_1=4098, class_size_2=1024, class_size_3=512, n_classes=10):
        super().__init__(
            nn.LayerNorm(emb_size),
            spectral_norm(nn.Linear(emb_size, class_size_1)),
            nn.GELU(),
            spectral_norm(nn.Linear(class_size_1, class_size_2)),
            nn.GELU(),
            spectral_norm(nn.Linear(class_size_2, class_size_3)),
            nn.GELU(),
            spectral_norm(nn.Linear(class_size_3, n_classes)),
            nn.GELU(),
        )

    def forward(self, x):
        # Take only the cls token outputs
        x = x[:, 0, :]
        return super().forward(x)


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels=3,
                 patch_size=4,
                 stride_size=4,
                 emb_size=384,
                 image_size=32,
                 depth=4,
                 n_classes=1,
                 diffaugment='color,translation,cutout',
                 **kwargs):
        self.diffaugment = diffaugment
        super().__init__(
            PatchEmbedding(in_channels, patch_size, stride_size, emb_size, image_size),
            DiscriminatorTransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes=n_classes)
        )

    def forward(self, img, do_augment=True):
        if do_augment:
            img = DiffAugment(img, policy=self.diffaugment)
        return super().forward(img)


class ViTDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 4,
            stride_size: int = 4,
            emb_size: int = 192,
            image_size: int = 224,
            depth: int = 8,
            n_classes=1,
            diff_augment='color,translation,cutout',
            **kwargs,
    ):
        self.diff_augment = diff_augment
        super().__init__()

        self.embedding = PatchEmbedding(in_channels, patch_size, stride_size, emb_size, image_size)
        self.transformer = DiscriminatorTransformerEncoder(depth, emb_size=emb_size, **kwargs)
        self.classifier = ClassificationHead(
            emb_size, n_classes=n_classes
        )

    def forward(self, img, do_augment=True):
        if do_augment:
            img = DiffAugment(img, policy=self.diff_augment)
        img = self.embedding(img)
        img = self.transformer(img)
        img = self.classifier(img)
        return img


# DISCRIMINATOR CLASSES #
class PixelDiscriminator(nn.Module):

    def __init__(self, input_c, output_c, ndf=64, norm_layer=nn.InstanceNorm2d):

        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = nn.Sequential(
            nn.Conv2d(input_c + output_c, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

# non-local block #
class Attention(nn.Module):
    def __init__(self, ch):
        super(Attention, self).__init__()

        self.ch = ch
        self.theta = SpectralNorm(nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
        self.phi = SpectralNorm(nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
        self.g = SpectralNorm(nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False))
        self.o = SpectralNorm(nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False))

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])

        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, phi.shape[2] * phi.shape[3])
        g = g.view(-1, self.ch // 2, g.shape[2] * g.shape[3])

        beta = fc.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)

        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# Modified version of "Image-to-Image Translation with Conditional Adversarial Networks" paper
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_c, output_c, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if n_layers == 0:
            PixelDiscriminator(input_c, output_c, ndf)
        else:
            sequence = [nn.Conv2d(input_c + output_c, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.1, True)]
            nf_mult = 1
            nf_mult_prev = 1

            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                if n == 1:
                    sequence += [
                        SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1,
                                               bias=use_bias)),
                        nn.LeakyReLU(0.1, True),
                        Attention(128)
                    ]
                else:
                    sequence += [
                        SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1,
                                               bias=use_bias)),
                        nn.LeakyReLU(0.1, True)
                    ]

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)

            sequence += [
                SpectralNorm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias)),
                nn.LeakyReLU(0.1, True)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
            self.model = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.model(x)
        return x
