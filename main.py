from pathlib import Path
import os
from lightning.pytorch.callbacks import ModelCheckpoint

import hydra
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
import torch
from checkpointing import restore_best_score
from parsing_utils import make_omegaconf_resolvers


@hydra.main(version_base=None, config_path="./cli_configs", config_name="train")
def main(cfg):

    # seeding
    if cfg.seed:
        seed_everything(cfg.seed)
        cfg.trainer.benchmark = False
        cfg.trainer.deterministic = True

    experiment_name = cfg.name

    # Set up output folder
    output_dir = os.path.join(cfg.exp_dir, experiment_name)
    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # setup logger
    try:
        Path(
            "./main.log"
        ).unlink()  # gets automatically created, however logs are available in Weights and Biases so we do not need to log twice
    except:
        pass
    log_path = Path(cfg.trainer.logger.save_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    uid = cfg.output_subdir.split("/")[-1]
    cfg.trainer.logger.group = uid

    # add sync_batchnorm if multiple GPUs are used
    if cfg.trainer.devices > 1 and cfg.trainer.accelerator == "gpu":
        cfg.trainer.sync_batchnorm = True

    # remove callbacks that are not enabled
    cfg.trainer.callbacks = [i for i in cfg.trainer.callbacks.values() if i]
    if not cfg.trainer["enable_checkpointing"]:
        cfg.trainer.callbacks = [
            i
            for i in cfg.trainer.callbacks
            if i["_target_"] != "lightning.pytorch.callbacks.ModelCheckpoint"
        ]

    print(OmegaConf.to_yaml(cfg))

    # in case of Cross Validation loop over the folds (default is 1 (no Cross Validation))
    for k in range(cfg.data.cv.k):
        output_fold_dir = os.path.join(output_dir, f"fold{k}")
        if not (os.path.exists(output_fold_dir)):
            os.makedirs(output_fold_dir)
        if cfg.data.cv.k > 1:
            cfg.data.module.fold = k
        else:
            if cfg.data.module.fold is not None:
                pass
            else:
                cfg.data.module.fold = "0"

        if cfg.trainer["enable_checkpointing"]:
            for i in cfg.trainer.callbacks:
                if i["_target_"] == "lightning.pytorch.callbacks.ModelCheckpoint":
                    i["dirpath"] = os.path.join(
                        str(cfg.exp_dir),
                        str(cfg.data.module.name),
                        "checkpoints",
                        uid,
                        str(cfg.data.module.fold),
                    )

        # set up model checkpoint
        ckpt_path = os.path.join(output_fold_dir, f"model_best.ckpt")
        checkpoint_callback = ModelCheckpoint(
            monitor=f"Val/{cfg.monitor.metric}",
            mode=cfg.monitor.mode,
            save_top_k=1,
            filename=ckpt_path,
            save_weights_only=False,
            save_last=True,
        )

        restore_ckpt = None
        if os.path.exists(ckpt_path) and not (cfg.overwrite):
            restore_best_score(checkpoint_callback, ckpt_path)
            restore_ckpt = ckpt_path

        # instantiate trainer, model and dataset
        trainer = instantiate(cfg.trainer, callbacks=[checkpoint_callback])
        model = instantiate(cfg.model, output_folder=output_fold_dir)

        if cfg.model.compile:
            model = torch.compile(model, mode="default")
        dataset = instantiate(cfg.data).module

        # log hypperparams and drop stuff that shouldn't be logged
        ## Model
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["model"].pop("_target_")
        cfg_dict["model"]["model"] = cfg_dict["model"].pop("name")
        trainer.logger.log_hyperparams(cfg_dict["model"])

        ## Data
        cfg_dict["data"]["module"].pop("_target_")
        if cfg_dict["data"]["module"]["train_transforms"] is not None:
            cfg_dict["data"]["module"]["train_transforms"] = ".".join(
                cfg_dict["data"]["module"]["train_transforms"]["_target_"].split(".")[
                    -2:
                ]
            )
        if cfg_dict["data"]["module"]["test_transforms"] is not None:
            cfg_dict["data"]["module"]["test_transforms"] = ".".join(
                cfg_dict["data"]["module"]["test_transforms"]["_target_"].split(".")[
                    -2:
                ]
            )
        cfg_dict["data"]["module"].pop("name")
        trainer.logger.log_hyperparams(cfg_dict["data"]["module"])

        ## Trainer
        cfg_dict["trainer"].pop("_target_")
        cfg_dict["trainer"].pop("callbacks")
        cfg_dict["trainer"].pop("enable_checkpointing")
        cfg_dict["trainer"].pop("enable_progress_bar")
        cfg_dict["trainer"].pop("logger")
        cfg_dict["trainer"].pop("num_sanity_val_steps")
        trainer.logger.log_hyperparams(cfg_dict["trainer"])

        # start fitting
        if cfg_dict["val_only"]:
            trainer.validate(model, dataset)
        else:
            trainer.fit(model, dataset, ckpt_path=restore_ckpt)

        wandb.finish()


if __name__ == "__main__":
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    make_omegaconf_resolvers()
    main()
