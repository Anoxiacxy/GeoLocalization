from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

class TensorBoardHistogram(Callback):

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger = pl_module.logger
        if isinstance(logger, TensorBoardLogger):
            for name, param in pl_module.named_parameters():  # 返回网络的
                logger.experiment.add_histogram(name + '_data', param, pl_module.current_epoch)
                if param.grad is not None:
                    logger.experiment.add_histogram(name + '_grad', param.grad, pl_module.current_epoch)
