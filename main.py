"""
python main.py fit --data_dir data --batch_size 4
"""
import os

import pytorch_lightning as pl
import argparse

import torch

import datasets
import models
from callbacks import TensorBoardHistogram

pl.seed_everything(15)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", metavar="<phase>", choices=["fit", "validate", "test"])
    parser.add_argument("--profiler", default='simple', choices=['simple', 'advanced'])
    parser.add_argument("--logger", choices=['tf'], default='tf')

    parser.add_argument("--model", choices=models.model_dict.keys(), default='rgan')
    parser.add_argument("--block", choices=[2, 3], default=3, type=int)
    parser.add_argument("--image_size", choices=[224], default=224, type=int)
    parser.add_argument("--embed_dim", choices=[128, 192, 384, 512, 768, 1024], default=128, type=int)
    parser.add_argument("--transformer_share_weight", action="store_true", default=False)
    parser.add_argument("--generator_share_weight", action="store_true", default=False)
    parser.add_argument("--thumbnail_scale", default=4, type=int)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--lr_r", type=float, default=1e-4)

    parser.add_argument("--datamodule", choices=datasets.datamodule_dict.keys(), default='university')
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), 'autodl-tmp/University-Release'))

    parser.add_argument("--ckpt_path", type=str, default=None, help="path to your checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulate_batches", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu", "auto"], default="gpu")
    parser.add_argument("--devices", type=str, default=[0])
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None)

    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=800000, help="total training iterations")
    parser.add_argument("--max_epochs", type=int, default=500, help="total training iterations")

    args = parser.parse_args()

    root_dir = f'{args.logger}-logs/{args.model}-{args.datamodule}'

    trainer = pl.Trainer(
        profiler=args.profiler,
        accumulate_grad_batches=args.accumulate_batches,
        precision=args.precision,
        default_root_dir=root_dir,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        callbacks=[
            pl.callbacks.LearningRateMonitor(
                logging_interval='step'
            ),
            pl.callbacks.RichProgressBar(),
            TensorBoardHistogram(),
        ],
        log_every_n_steps=args.log_every_n_steps,
    )
    datamodule = datasets.datamodule_dict[args.datamodule](
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        thumbnail_scale=args.thumbnail_scale,
    )
    datamodule.setup()
    model = models.model_dict[args.model](
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lr_r=args.lr_r,
        image_size=args.image_size,
        embed_dim=args.embed_dim,
        class_dim=len(datamodule.class_names),
        thumbnail_scale=args.thumbnail_scale,
        transformer_share_weight=args.transformer_share_weight,
        generator_share_weight=args.generator_share_weight,

        normalize=datamodule.normalize(),
        denormalize=datamodule.denormalize(),
    )

    if args.phase == 'fit':
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path,
        )
    elif args.phase == "validate":
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path,
        )
    elif args.phase == 'test':
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path,
        )
