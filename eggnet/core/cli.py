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
    "--checkpoint", "-c", required=True, help="Checkpoint to use for training"
)
@click.option("--output_dir", "-o", default=None, help="Directory to save the output pyg files. Default to the same output dir if not specified.")
@click.option("--datasets", "-d", default=None, multiple=True)
@click.option("--accelerator", "-a", default=None)
@click.option("--devices", "-dv", default=None, type=int)
@click.option("--num_nodes", "-n", default=None, type=int)
def infer(**kwargs):
    return infer_stage.infer(**kwargs)


@cli.command()
@click.argument("config_file")
@click.argument("eval_config_file")
@click.option("--output_dir", "-o", default=None, help="Directory to save the output pyg files. Default to the same output dir if not specified.")
@click.option("--accelerator", "-a", default="cuda")
@click.option("--dataset", "-d", default="valset")
def eval(**kwargs):
    return eval_stage.eval(**kwargs)


if __name__ == "__main__":
    cli()
