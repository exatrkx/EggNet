import click
# from . import train_stage, infer_stage, eval_stage


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
@click.option("--slurm", "-s", is_flag=True, type=bool, help="Submit to slurm batch.")
def train(**kwargs):
    from . import train_stage
    return train_stage.train(**kwargs)


@cli.command()
@click.argument("config_file")
@click.option(
    "--checkpoint", "-c", required=True, help="Checkpoint to use for inference"
)
@click.option("--output_dir", "-o", default=None, help="Directory to save the output pyg files. Default to the same output_dir as in training_config if not specified.")
@click.option("--dataset", "-d", default=None, multiple=True, type=click.Choice(["trainset", "valset", "testset"]), help="Which dataset to run inference. Default is all datasets. Can specify one dataset or multiple.")
@click.option("--max-events", default=None, type=click.IntRange(min=1), help="Limit inference to the first N events from each selected dataset.")
@click.option("--accelerator", "-a", default=None, type=click.Choice(["cuda", "cpu"]), help="Which device to use. Default will be what is specified in the training config.")
@click.option("--devices", "-dv", default=None, type=int, help="Number of devices. Default will be what is specified in the training config.")
@click.option("--num_nodes", "-n", default=None, type=int, help="Number of nodes. Default will be what is specified in the training config.")
@click.option(
    "--reuse_inference_output/--no-reuse_inference_output",
    default=False,
    help="Reuse existing output_dir .pyg files from a previous run instead of rerunning embedding inference.",
)
@click.option("--slurm", "-s", is_flag=True, type=bool, help="Submit to slurm batch.")
def infer(**kwargs):
    from . import infer_stage
    return infer_stage.infer(**kwargs)


@cli.command()
@click.argument("config_file")
@click.argument("eval_config_file")
@click.option("--output_dir", "-o", default=None, help="Directory with the inference data and where to save the evaluation plots. Default to the same output_dir as in training_config if not specified.")
@click.option("--accelerator", "-a", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Which device to use. Default is cuda. Note: currently only supports cuda")
@click.option("--dataset", "-d", default="valset", type=click.Choice(["trainset", "valset", "testset"]), help="Specify a dataset to run inference. Default is valset.")
@click.option("--max-events", "max_events", default=None, type=click.IntRange(min=1), help="Limit eval to the first N events from the selected dataset.")
@click.option("--slurm", "-s", is_flag=True, type=bool, help="Submit to slurm batch.")
def eval(**kwargs):
    from . import eval_stage
    return eval_stage.eval(**kwargs)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--event-id",
    default=None,
    help="Show one event from a directory by event id instead of the aggregate summary.",
)
@click.option(
    "--prefix",
    default=None,
    help="Only include profiling stages that start with this prefix.",
)
@click.option(
    "--top",
    default=15,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of rows to print.",
)
@click.option(
    "--slowest-events",
    default=3,
    show_default=True,
    type=click.IntRange(min=0),
    help="Number of slowest event ids to print for each aggregate stage row.",
)
def profile(**kwargs):
    from eggnet.tools.profile_viewer import run_profile_viewer

    return run_profile_viewer(**kwargs)

if __name__ == "__main__":
    cli()
