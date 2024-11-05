import os
import yaml

from eggnet import lightning_modules
from eggnet.utils.loading import get_stage_module, get_trainer


def train(
    config_file, checkpoint, checkpoint_resume_dir, load_only_model_parameters
):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(yaml.dump(config))

    base_model_class = getattr(lightning_modules, config.get("base_model", "NodeEncoding"))

    os.makedirs(config["output_dir"], exist_ok=True)

    base_model, ckpt_config, default_root_dir, checkpoint = get_stage_module(
        config,
        base_model_class,
        checkpoint_path=checkpoint,
        checkpoint_resume_dir=checkpoint_resume_dir,
    )
    trainer = get_trainer(config, default_root_dir)
    if load_only_model_parameters:
        trainer.fit(base_model)
    else:
        trainer.fit(base_model, ckpt_path=checkpoint)
