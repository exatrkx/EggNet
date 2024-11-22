# EggNet
Graph-based Graph Attention Network for End-to-end Particle Track Reconstruction.

This is the repository to reproduce the EggNet results presented in:
- ICML AI4Science Workshop: https://arxiv.org/abs/2407.13925
- CHEP 2024: https://indico.cern.ch/event/1338689/contributions/6010085/

A large fraction of the codes in this respository is based on the Acorn framework (https://gitlab.cern.ch/gnn4itkteam/acorn). Any questions or bug reports please contact Jay Chan (jaychan@lbl.gov).


## Installation
The codes should be run in a conda / mamba environment. For NERSC Perlmutter, please run the following commands:
```
module load conda/Mambaforge-22.11.1-4
module load cudatoolkit/12.2
```

Assuming GPU capability with cuda version >= 12.2, run the following commands.

```
mamba create --name eggnet python=3.10 && mamba activate eggnet
pip install torch==2.1.0 && pip install --no-cache-dir -r requirements.txt
pip install -e .
```

Replace `mamba` with `conda` if using conda environment instead.

## Running EggNet pipeline
A full cycle consists of three stages: training, inference and evaluation. 

### Training
To train an EggNet model, one needs to prepare a training configuration file `.yaml`. Example training yaml can be found in `configs/trackML/eggnet_trackml.yaml`. The command to run the training:

```
> eggnet train --help
Usage: eggnet train [OPTIONS] TRAINING_CONFIG

Options:
  -c, --checkpoint TEXT         Checkpoint to use for training
  --checkpoint_resume_dir TEXT  Pass a default rootdir for saving model
                                checkpoint
  --load_only_model_parameters  Load only model parameters from checkpoint
                                instead of the full training states
  -s, --slurm                   Submit to slurm batch.
  --help                        Show this message and exit.
```

During the training, the model checkpoints will be automatically saved in the `output_dir` specified in the training yaml, which can be used to resume a training, or used for inference (next step). The training progress will also be logged on Wandb.

### Inference
To perform an inference, run the following command:

```
> eggnet infer --help
Usage: eggnet infer [OPTIONS] TRAINING_CONFIG

Options:
  -c, --checkpoint TEXT           Checkpoint to use for inference  [required]
  -o, --output_dir TEXT           Directory to save the output pyg files.
                                  Default to the same output_dir as in
                                  training_config if not specified.
  -d, --dataset [trainset|valset|testset]
                                  Which dataset to run inference. Default is
                                  all datasets. Can specify one dataset or
                                  multiple.
  -a, --accelerator [cuda|cpu]    Which device to use. Default will be what is
                                  specified in the training config.
  -dv, --devices INTEGER          Number of devices. Default will be what is
                                  specified in the training config.
  -n, --num_nodes INTEGER         Number of nodes. Default will be what is
                                  specified in the training config.
  -s, --slurm                   Submit to slurm batch.
  --help                          Show this message and exit.
```
Note that here `TRAINING_CONFIG` should be exactly the same as what was used for the training. The data with the inference results (i.e. node embedding) will be saved to the `output_dir`.

### Evaluation
To run the evaluation, one needs to run the inference first with the inference data. One then needs to provide both the training yaml as well as an evaluation yaml. An example evaluation yaml can be found in `configs/trackML/eval_trackml.yaml`. The command to run the evaluation:

```
> eggnet eval --help
Usage: eggnet eval [OPTIONS] TRAINING_CONFIG EVAL_CONFIG_FILE

Options:
  -o, --output_dir TEXT           Directory with the inference data and where
                                  to save the evaluation plots. Default to the
                                  same output_dir as in training_config if not
                                  specified.
  -a, --accelerator [cuda|cpu]    Which device to use. Default is cpu.
  -d, --dataset [trainset|valset|testset]
                                  Specify a dataset to run inference. Default
                                  is valset.
  -s, --slurm                   Submit to slurm batch.
  --help                          Show this message and exit.
```
The evaluation will run the clustering on the node embedding and calculate track efficiency, fake rate and duplication rate. It will also evaluate inferece time. Results are saved to the same output_dir where the inference data are stored.

## Slurm batch

The program also supports slurm batch submission. Simply pass `-s` or `--slurm` to your command line. Before submitting to batch, edit the `.env` to set the correct project ID for your batch job (default is `m2616`).
