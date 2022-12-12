import itertools
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

import torch
from torch import nn, autograd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.autograd import Variable
from torch.nn import functional as fc
from models.classifiers import ClassBlock
from . import conv2d_gradfix
from .discriminator import ViTDiscriminator
from .loss import TripletLoss
from .transformers import (
    deit_micro_patch16_LS,
    deit_tiny_patch16_LS,
    deit_small_patch16_LS,
    deit_medium_patch16_LS,
    deit_base_patch16_LS,
    deit_large_patch16_LS
)
from .generator import GeneratorViT

deit_embed_dict: Dict[int, Any] = {
    128: deit_micro_patch16_LS,
    192: deit_tiny_patch16_LS,
    384: deit_small_patch16_LS,
    512: deit_medium_patch16_LS,
    768: deit_base_patch16_LS,
    1024: deit_large_patch16_LS,
}


def cal_loss(outputs, labels, loss_func):
    loss = 0
    if outputs.dim() == 3:
        for i in range(outputs.shape[0]):
            loss += loss_func(outputs[i], labels)
        loss = loss / outputs.shape[0]
    elif outputs.dim() == 2:
        loss = loss_func(outputs, labels)
    else:
        raise NotImplementedError()

    return loss


def calc_triplet_loss(outputs, outputs2, labels, loss_func):
    loss = 0
    if outputs.dim() == 3:
        for i in range(outputs.shape[0]):
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
            labels_concat = torch.cat((labels, labels), dim=0)
            loss += loss_func(out_concat, labels_concat)
        loss = loss / outputs.shape[0]
    elif outputs.dim() == 2:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels, labels), dim=0)
        loss = loss_func(out_concat, labels_concat)
    else:
        raise NotImplementedError()

    return loss


def calc_kl_loss(outputs, outputs2, loss_func):
    loss = 0
    if outputs.dim() == 3:
        for i in range(outputs.shape[0]):
            loss += loss_func(
                fc.log_softmax(outputs[i], dim=1),
                fc.softmax(Variable(outputs2[i]), dim=1))
        loss = loss / outputs.shape[0]
    elif outputs.dim() == 2:
        loss = loss_func(
            fc.log_softmax(outputs, dim=1),
            fc.softmax(Variable(outputs2), dim=1))
    else:
        raise NotImplementedError()

    return loss


class LitRelativisticGAN(pl.LightningModule):
    def __init__(
            self,
            image_size=224,
            block: int = 2,

            embed_dim: int = 768,
            class_dim: int = 1000,
            thumbnail_scale: int = 2,

            cls_batch_norm: bool = False,

            triplet_loss: float = 1.0,
            kl_loss: float = 1.0,
            # optimizer
            lr_g: float = 1e-4,
            lr_r: float = 1e-4,
            lr_d: float = 1e-4,
            beta_1: float = 0.5,
            beta_2: float = 0.999,

            transformer_share_weight: bool = False,
            generator_share_weight: bool = False,

            denormalize: Optional[Callable] = None,
            normalize: Optional[Callable] = None,
    ):
        super(LitRelativisticGAN, self).__init__()
        self.save_hyperparameters(ignore=["denormalize", "normalize"])
        self.denormalize = denormalize if denormalize is not None else nn.Identity()
        self.normalize = normalize if normalize is not None else nn.Identity()
        # model
        self.generator_transformer = deit_embed_dict[embed_dim](pretrained=True, pretrained_21k=False)
        if transformer_share_weight:
            self.retrieval_transformer = self.generator_transformer
        else:
            self.retrieval_transformer = deit_embed_dict[embed_dim](pretrained=True, pretrained_21k=False)

        self.generator_class_classifiers = ClassBlock(embed_dim, class_dim, 0.5, batch_norm=cls_batch_norm)
        self.generator_heat_classifiers = nn.ModuleList()
        for i in range(block):
            self.generator_heat_classifiers.append(ClassBlock(embed_dim, class_dim, 0.5, batch_norm=cls_batch_norm))

        self.retrieval_class_classifiers = ClassBlock(embed_dim, class_dim, 0.5, batch_norm=cls_batch_norm)
        self.retrieval_heat_classifiers = nn.ModuleList()
        for i in range(block):
            self.retrieval_heat_classifiers.append(ClassBlock(embed_dim, class_dim, 0.5, batch_norm=cls_batch_norm))

        self.generator_heads = nn.ModuleList()
        if generator_share_weight:
            generator_head = GeneratorViT(
                hidden_size=embed_dim,
                sln_parameter_size=embed_dim,
                image_size=image_size // thumbnail_scale,
                patch_size=16 // thumbnail_scale,
                out_patch_size=16 // thumbnail_scale,
            )
            for i in range(thumbnail_scale ** 2):
                self.generator_heads.append(generator_head)
        else:
            for i in range(thumbnail_scale ** 2):
                self.generator_heads.append(GeneratorViT(
                    hidden_size=embed_dim,
                    sln_parameter_size=embed_dim,
                    image_size=image_size // thumbnail_scale,
                    patch_size=16 // thumbnail_scale,
                    out_patch_size=16 // thumbnail_scale,
                ))

        self.discriminator = ViTDiscriminator(
            in_channels=3+3,
            emb_size=embed_dim // thumbnail_scale,
            image_size=image_size,
            patch_size=32 // thumbnail_scale,
            stride_size=32 // thumbnail_scale,
        )

        # loss
        self.triplet_loss = TripletLoss(margin=triplet_loss)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = torch.nn.L1Loss()

    def configure_optimizers(self):
        hp = self.hparams
        d_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=hp.lr_d, betas=(hp.beta_1, hp.beta_2))
        g_optim = torch.optim.Adam(itertools.chain(
            self.generator_heads.parameters(),
            self.generator_transformer.parameters()
        ), lr=hp.lr_g, betas=(hp.beta_1, hp.beta_2))
        r_optim = torch.optim.Adam(itertools.chain(
            self.generator_transformer.parameters(),
            self.generator_class_classifiers.parameters(),
            self.generator_heat_classifiers.parameters(),
            self.retrieval_transformer.parameters(),
            self.retrieval_class_classifiers.parameters(),
            self.retrieval_heat_classifiers.parameters(),
        ), lr=hp.lr_r, betas=(hp.beta_1, hp.beta_2))

        return [d_optim, g_optim, r_optim], []

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    @staticmethod
    def d_logistic_loss(real_pred, fake_pred):
        real_loss = fc.softplus(-real_pred)
        fake_loss = fc.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    @staticmethod
    def d_r1_loss(real_pred, real_img):
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(
                outputs=real_pred.sum(), inputs=real_img, create_graph=True
            )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def g_non_saturating_loss(fake_pred):
        loss = fc.softplus(-fake_pred)

        return loss.mean()

    def make_grid(self, output_images: List[torch.Tensor]) -> torch.Tensor:
        """
        :param output_images: shape [batch, channel, height, width]
        :return:
        """
        size = self.hparams.thumbnail_scale
        grid = torch.cat([torch.cat(output_images[i: i + size], dim=-1) for i in range(0, size * size, size)], dim=-2)
        return grid

    def get_heatmap_pool(self, img_tokens, add_global=False, other_branch=False):
        """
        :param img_tokens: [batch, num_patches, embed]
        :param add_global:
        :param other_branch:
        :return:
        """
        hp = self.hparams
        heatmap = torch.mean(img_tokens, dim=-1)
        size = img_tokens.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [img_tokens[i, arg[i], :] for i in range(img_tokens.size(0))]
        x_sort = torch.stack(x_sort, dim=0)  # [batch, num_patches, embed]

        split_each = size / hp.block
        split_list = [int(split_each) for _ in range(hp.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)  # (block, [batch, split, embed])

        split_list = [torch.mean(split, dim=1) for split in split_x]  # (block, [batch, embed])
        part_features = torch.stack(split_list, dim=2)  # [batch, embed, block]
        if add_global:
            global_feat = torch.mean(img_tokens, dim=1).view(img_tokens.size(0), -1, 1).expand(-1, -1, hp.block)
            part_features = part_features + global_feat
        if other_branch:
            other_branch = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
            return part_features, other_branch
        return part_features

    def part_classifier(self, part_features, generator=True):
        hp = self.hparams
        predicts, features = [], []
        for i in range(hp.block):
            part = part_features[:, :, i]
            if generator:
                predict, feature = self.generator_heat_classifiers[i](part)
            else:
                predict, feature = self.retrieval_heat_classifiers[i](part)
            predicts.append(predict)
            features.append(feature)
        if None in predicts:
            return None, torch.stack(features, dim=0)
        else:
            return torch.stack(predicts, dim=0), torch.stack(features, dim=0)

    def forward_generator(
            self, input_image: torch.Tensor, denormalize: bool = False,
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor):
        """
        :param denormalize: 是否将生成的图像进行反正则化操作，设为 True 的时候可以返回正常图像
        :param input_image: 输入一张卫星图片，用于生成街道视图或者无人机视图
                            输入图片形状为 [batch_size, channel, height, width]
        :return: 返回生成器生成的 缩略图，单张原图，分类结果，特征提取结果
        """
        tokens = self.generator_transformer.forward_all_tokens(input_image)
        cls_token, img_tokens = tokens[:, 0], tokens[:, 1:]
        output_images = []
        for generator_head in self.generator_heads:
            output_images.append(generator_head.forward_token(img_tokens))

        thumbnail_image = self.make_grid(output_images)
        part_features = self.get_heatmap_pool(img_tokens)
        p_cls, p_feat = self.part_classifier(part_features, True)
        t_cls, t_feat = self.generator_class_classifiers(cls_token)
        cls = torch.cat([t_cls.unsqueeze(0), p_cls], dim=0) if t_cls is not None else None
        feat = torch.cat([t_feat.unsqueeze(0), p_feat], dim=0)

        return thumbnail_image, output_images, cls, feat

    def forward_retrieval(self, input_image: torch.Tensor):
        """
        :param input_image: 输入一张无人机视图或者街道视图
                            输入图片形状为 [batch_size, channel, height, width]
        :return: 返回检索器的 分类结果，特征提取结果
        """
        tokens = self.retrieval_transformer.forward_all_tokens(input_image)
        cls_token, img_tokens = tokens[:, 0], tokens[:, 1:]

        part_features = self.get_heatmap_pool(img_tokens)
        p_cls, p_feat = self.part_classifier(part_features, False)
        t_cls, t_feat = self.retrieval_class_classifiers(cls_token)
        cls = torch.cat([t_cls.unsqueeze(0), p_cls], dim=0) if t_cls is not None else None
        feat = torch.cat([t_feat.unsqueeze(0), p_feat], dim=0)

        return cls, feat

    def forward(self, batch):
        (img_sa, ids), (img_dr, ids, thumb_dr) = batch
        thumb_fake, _, _, _ = self.forward_generator(img_sa)
        dr_cls, dr_feat = self.forward_retrieval(img_dr)
        fake_pred = self.discriminator(torch.cat([thumb_fake, img_sa], dim=1))
        return 0

    def training_step(self, batch, batch_idx, optimizer_idx) -> STEP_OUTPUT:
        """
        :param batch: 从 DataModule 加载出来的一批数据
        :param batch_idx:
        :param optimizer_idx: 优化器编号: 0 - 判别器，1 - 生成器，2 - 检索器
        :return:
        """
        # print(f"batch_idx = {batch_idx}, optimizer_idx = {optimizer_idx}")
        (img_sa, ids), (img_dr, ids, thumb_real) = batch
        hp = self.hparams

        if optimizer_idx == 0:
            # train discriminator
            self.requires_grad(self.discriminator, True)
            self.requires_grad(self.generator_transformer, False)
            self.requires_grad(self.generator_heads, False)

            thumb_fake, _, _, _ = self.forward_generator(img_sa)

            thumb_real = self.make_grid(thumb_real)
            thumb_real.requires_grad = True
            real_pred = self.discriminator(torch.cat([thumb_real, img_sa], dim=1))
            fake_pred = self.discriminator(torch.cat([thumb_fake, img_sa], dim=1))
            d_loss = self.d_logistic_loss(real_pred, fake_pred)
            d_r1 = self.d_r1_loss(real_pred, thumb_real)
            loss = d_loss + d_r1

            self.log("Score/Real", real_pred.mean(), prog_bar=True)
            self.log("Score/Fake", fake_pred.mean(), prog_bar=True)
            self.log("Loss/R1", d_r1, prog_bar=True)
            self.log("Loss/Discriminator", loss, prog_bar=True)

        elif optimizer_idx == 1:
            self.requires_grad(self.discriminator, False)
            self.requires_grad(self.generator_transformer, True)
            self.requires_grad(self.generator_heads, True)

            thumb_fake, _, _, _ = self.forward_generator(img_sa)
            fake_pred = self.discriminator(torch.cat([thumb_fake, img_sa], dim=1))

            g_loss = self.g_non_saturating_loss(fake_pred)
            loss = g_loss
            self.log("Loss/Generator", loss, prog_bar=True)

        elif optimizer_idx == 2:
            # train retrieval
            self.requires_grad(self.generator_transformer, False)
            _, _, sa_cls, sa_feat = self.forward_generator(img_sa)
            dr_cls, dr_feat = self.forward_retrieval(img_dr)
            triplet_loss = calc_triplet_loss(sa_feat, dr_feat, ids, self.triplet_loss)
            kl_loss = calc_kl_loss(sa_cls, dr_cls, self.kl_loss)
            cls_loss = cal_loss(sa_cls, ids, self.ce_loss) + cal_loss(dr_cls, ids, self.ce_loss)

            loss = hp.triplet_loss * triplet_loss + hp.kl_loss * kl_loss + cls_loss
            self.log("Loss/Triplet", triplet_loss, prog_bar=True)
            self.log("Loss/KL", kl_loss, prog_bar=True)
            self.log("Loss/Class", cls_loss, prog_bar=True)
            self.log("Loss/Retrieval", loss, prog_bar=True)

        else:
            raise NotImplementedError()

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        (img_sa, ids), (img_dr, ids, thumb_dr) = batch
        thumb_cat = self.make_grid(thumb_dr)
        thumb_gan, _, sa_cls, sa_feat = self.forward_generator(img_sa)
        dr_cls, dr_feat = self.forward_retrieval(img_dr)
        logger = self.logger
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_image("Thumb/Real", self.denormalize(thumb_cat[0]))
            logger.experiment.add_image("Thumb/Fake", self.denormalize(thumb_gan[0]))
        return None

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        ...

