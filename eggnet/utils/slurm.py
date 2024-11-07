import secrets
import subprocess
import os

from dotenv import dotenv_values


def submit_to_slurm(command, accelerator, devices, num_nodes, gpu_memory=40):

    env_config = dotenv_values()
    project_id = env_config.get("project_id", "m2616")

    job_script = f"""#!/bin/bash

#SBATCH -A {project_id}{"" if accelerator=="cpu" else "_g"}
#SBATCH -q regular
#SBATCH -C {"cpu" if accelerator=="cpu" else "gpu&hbm80g" if gpu_memory==80 else "gpu"}
#SBATCH -t 12:00:00
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={devices}
#SBATCH --gpus-per-task=1
#SBATCH -c {(2 * 64 // devices) if accelerator != "cpu" else (2 * 128// devices)}
#SBATCH -o logs/%x-%j.out
#SBATCH -J eggnet
#SBATCH --gpu-bind=none
#SBATCH --comment=96:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --requeue

# Setup
mkdir -p logs
eval "$(conda shell.bash hook)"

# module load python/3.9-anaconda-2021.11
# mamba activate eggnet

export SLURM_CPU_BIND="cores"
export WANDB__SERVICE_WAIT=300
echo -e "\n{command}\n"

# Single GPU training
srun {command}

"""

    temp_file_name = secrets.token_hex(nbytes=16) + ".sh"
    with open(temp_file_name, "w") as f:
        f.write(job_script)

    subprocess.run(["sbatch", temp_file_name])

    os.remove(temp_file_name)
