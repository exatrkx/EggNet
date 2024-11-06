import click
from . import train_stage, infer_stage, eval_stage


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_file")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for training"
)
@click.option(
    "--checkpoint_resume_dir",
    default=None,
    help="Pass a default rootdir for saving model checkpoint",
)
@click.option(
    "--load_only_model_parameters",
    is_flag=True,
    type=bool,
    help="Load only model parameters from checkpoint instead of the full training states",
)
def train(**kwargs):
    return train_stage.train(**kwargs)


@cli.command()
@click.argument("config_file")
@click.option(
    "--checkpoint", "-c", required=True, help="Checkpoint to use for inference"
)
@click.option("--output_dir", "-o", default=None, help="Directory to save the output pyg files. Default to the same output_dir as in training_config if not specified.")
@click.option("--datasets", "-d", default=None, multiple=True, type=click.Choice(["trainset", "valset", "testset"]), help="Which dataset to run inference. Default is all datasets. Can specify one dataset or multiple.")
@click.option("--accelerator", "-a", default=None, type=click.Choice(["cuda", "cpu"]), help="Which device to use. Default will be what is specified in the training config.")
@click.option("--devices", "-dv", default=None, type=int, help="Number of devices. Default will be what is specified in the training config.")
@click.option("--num_nodes", "-n", default=None, type=int, help="Number of nodes. Default will be what is specified in the training config.")
def infer(**kwargs):
    return infer_stage.infer(**kwargs)


@cli.command()
@click.argument("config_file")
@click.argument("eval_config_file")
@click.option("--output_dir", "-o", default=None, help="Directory with the inference data and where to save the evaluation plots. Default to the same output_dir as in training_config if not specified.")
@click.option("--accelerator", "-a", default="cpu", type=click.Choice(["cuda", "cpu"]), help="Which device to use. Default is cpu")
@click.option("--dataset", "-d", default="valset", type=click.Choice(["trainset", "valset", "testset"]), help="Specify a dataset to run inference. Default is valset.")
def eval(**kwargs):
    return eval_stage.eval(**kwargs)


if __name__ == "__main__":
    cli()
