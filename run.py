import os
import math
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from data import CNNDailyMail_Module
from model import Summarizer


def argLoader():
    parser = ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="train the model")
    parser.add_argument("--mini", action="store_true", help="Only Load 500 instances")
    
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="deepspeed_stage_2")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size_per_gpu", type=int, default=5)

    parser.add_argument("--model", type=str, default="facebook/bart-large")
    parser.add_argument("--data_path", type=str, default="../../sumData/cnn_dailymail")
    parser.add_argument("--build_from_strach", action="store_true", help="Whether or not build dataset from strach")
    parser.add_argument("--load_from_cache", action="store_true", help="Whether or not load from cache file")
    parser.add_argument("--model_path", type=str, default="./model")


    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--num_train_instances", type=int, default=287113)
    parser.add_argument("--warmup_steps", type=int, default=1795)
    parser.add_argument("--valid_per_epoch", type=int, default=5)

    args = parser.parse_args()
    args.pad = 1

    args.train_steps = \
        int(math.ceil(float(args.num_train_instances / (args.batch_size_per_gpu * args.n_gpus))) * args.max_epochs)

    assert args.load_from_cache != args.build_from_strach

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    return args

if __name__ == "__main__":
    config = argLoader()
    if config.do_train:
        data = CNNDailyMail_Module(config)
        model = Summarizer(config)

        checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{step}-{valid_loss:.8f}',
            save_last=True,
            monitor="valid_loss",
            mode="min",
            save_top_k=3,
            save_weights_only=True,
        )

        if config.strategy == "deepspeed_stage_2_offload":
            config.strategy = DeepSpeedStrategy(
                offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8
            )

        trainer = pl.Trainer(
            default_root_dir=config.model_path,
            accelerator=config.accelerator,
            devices=config.n_gpus,
            strategy=config.strategy,
            precision=16,
            max_epochs=config.max_epochs,
            callbacks=[checkpoint_callback],
            terminate_on_nan=True,
            sync_batchnorm=True,
            val_check_interval=1.0 / config.valid_per_epoch,
            profiler="simple",
        )

        trainer.fit(model=model, datamodule=data)
