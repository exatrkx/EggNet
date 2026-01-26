import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd

from eggnet import lightning_modules
from eggnet.utils.cluster import cluster_and_match
from eggnet.utils.plotting import *
from eggnet.utils.slurm import submit_to_slurm

def eval(config_file, eval_config_file, output_dir, accelerator, dataset, slurm):
    # plot_eff_vs_eps
    # plot_eff_fixed_eps
    # plot_computing_time
    if slurm:
        eval_slurm(config_file, eval_config_file, output_dir, accelerator, dataset)
        return
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if output_dir is not None:
        config["output_dir"] = output_dir
    with open(eval_config_file, "r") as f:
        eval_config = yaml.load(f, Loader=yaml.FullLoader)
    eval_config["output_dir"] = config["output_dir"]
    base_model = getattr(lightning_modules, config.get("base_model", "NodeEncoding"))(config)
    # base_model.setup(stage="test", datasets=[dataset]) # BUG: Doesn't work
    base_model.datasets = dataset
    base_model.setup(stage="test")
    data = getattr(base_model, dataset)
    plot_computing_time_bool = eval_config.get("plot_computing_time", False)
    plot_eps_bool = any([eval_config.get(key, False) for key in eval_config.keys() if ('eps' in key and 'plot' in key)])

    if plot_eps_bool:
        eps_data = pd.DataFrame({
            "eps": np.arange(0.05, 0.51, 0.05),
            "n_particles": 0,
            "n_matched_particles": 0,
            "n_matched_tracks": 0,
            "n_matched_target_particles": 0,
            "n_matched_target_tracks": 0,
            "n_tracks": 0,
        })
    if plot_computing_time_bool:
        time_data = pd.DataFrame({
            "num_nodes": [],
            "eggnet": [],
            "knn": [],
            "dbscan": [],
        })

    if eval_config.get("pT_unit", "MeV") == "MeV":
        pt_min, pt_max = 1000, 50000
    else:
        pt_min, pt_max = 1, 50
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

    particles_pt_hist = np.histogram([], bins=pt_bins)[0]
    matched_target_particles_pt_hist = np.histogram([], bins=pt_bins)[0]
    if eval_config.get("plot_eta", True):
        eta_bins = np.linspace(-4, 4)
        particles_eta_hist = np.histogram([], bins=eta_bins)[0]
        matched_target_particles_eta_hist = np.histogram([], bins=eta_bins)[0]
    # Calculate EPS data only if we need it, since it takes a while to load all the events into memory
    if plot_computing_time_bool or plot_eps_bool:
        for event in tqdm(data):
            event = event.to(accelerator)
            if plot_eps_bool:
                for eps_i in eps_data.eps:
                    eps_data_i, particles_pt_hist_i, matched_target_particles_pt_hist_i, particles_eta_hist_i, matched_target_particles_eta_hist_i = cluster_and_match(event, eps_i, eval_config, time_yes=True if eps_i == eval_config["eps"] else False)

                    eps_data[eps_data.eps == eps_i] = eps_data[eps_data.eps == eps_i].to_numpy() + eps_data_i.to_numpy()

                    if eps_i == eval_config["eps"]:
                        particles_pt_hist += particles_pt_hist_i
                        matched_target_particles_pt_hist += matched_target_particles_pt_hist_i
                        if eval_config.get("plot_eta", True):
                            particles_eta_hist += particles_eta_hist_i
                            matched_target_particles_eta_hist += matched_target_particles_eta_hist_i

            if plot_computing_time_bool:
                time_data = pd.concat([time_data, pd.DataFrame({
                    "num_nodes": [event["num_nodes"].cpu()],
                    "eggnet": [event["BaseModule.forward"]],
                    "knn": [event[f"{config.get('knn_algorithm', 'cu_knn')}.get_graph"]],
                    "dbscan": [event["cluster"]]
                    }
            )])

    # check metric!!
    if plot_eps_bool:
        eps_data["eff"] = eps_data.n_matched_target_particles / eps_data.n_particles
        eps_data["dup"] = (
            eps_data.n_matched_target_tracks - eps_data.n_matched_target_particles
        ) / eps_data.n_matched_target_particles
        eps_data["fak"] = (eps_data.n_tracks - eps_data.n_matched_tracks) / eps_data.n_matched_particles

    if plot_computing_time_bool:
        time_data["gnn"] = time_data["eggnet"] - time_data["knn"]
        time_data["total"] = time_data["eggnet"] + time_data["dbscan"]
    if eval_config.get("plot_eff_vs_eps", False):
        plot_eff_vs_eps(eps_data, eval_config)
    if eval_config.get("plot_eff_fixed_eps", False):
        plot_eff_fixed_eps(matched_target_particles_pt_hist, particles_pt_hist, eps_data, eval_config, pt_bins, f"$p_T$ [{eval_config.get('pT_unit', 'MeV')}]", logx=True, filename="track_efficiency_pt.png")
    if eval_config.get("plot_eta", False):
        plot_eff_fixed_eps(matched_target_particles_eta_hist, particles_eta_hist, eps_data, eval_config, eta_bins, r"$\eta$", logx=False, filename="track_efficiency_eta.png")
    if eval_config.get("plot_parameter_predict", False):
        plot_hit_parameter_prediction_accuracy(data, config, eval_config)
    if plot_computing_time_bool:
        plot_computing_time(time_data, eval_config)
    if eval_config.get("plot_resolution_PT", False):
        if eval_config.get("plot_resolution_PT_x", None):
            x_range = eval_config.get("plot_resolution_PT_x", None)
        else:
            x_range = None
        if len(x_range) == 0: 
            x_range = None
        if x_range:
            if all([isinstance(x, list) for x in x_range]):
                for x in x_range:
                    plot_binned_std_of_residuals_PT(
                        data,
                        config,
                        eval_config,
                        filename=f"binned_std_of_residuals_PT.png",
                        num_bins=50,
                        x_range=x,
                    )
            else:
                plot_binned_std_of_residuals_PT(
                    data,
                    config,
                    eval_config,
                    filename="binned_std_of_residuals_PT.png",
                    num_bins=50,
                    x_range=x_range
                )
        else:
            plot_binned_std_of_residuals_PT(
                data,
                config,
                eval_config,
                filename="binned_std_of_residuals_PT.png",
                num_bins=50,
            )
    if eval_config.get("plot_resolution_ETA", False):
        plot_binned_std_of_residuals_ETA(
            data,
            config,
            eval_config,
            filename="binned_std_of_residuals_ETA.png",
            num_bins=50,
            
        )
    if eval_config.get("plot_resolution_confidence", False):
        if eval_config.get("plot_resolution_confidence_x", None):
            x_range = eval_config.get("plot_resolution_confidence_x", None)
        else:
            x_range = None
        if len(x_range) == 0: 
            x_range = None
        if x_range:
            if all([isinstance(x, list) for x in x_range]):
                for x in x_range:
                    plot_binned_std_of_residuals_confidence(
                        data,
                        config,
                        eval_config,
                        filename=f"binned_std_of_residuals_confidence.png",
                        num_bins=70,
                        x_range=x
                    )
            else:
                print("YAY")
                plot_binned_std_of_residuals_confidence(
                    data,
                    config,
                    eval_config,
                    filename="binned_std_of_residuals_confidence.png",
                    num_bins=70,
                    x_range=x_range
                )
        else:
            plot_binned_std_of_residuals_confidence(
                data,
                config,
                eval_config,
                filename="binned_std_of_residuals_confidence.png",
                num_bins=50,
            )
    if eval_config.get("plot_binned_hit_param_pred_acc", False):
        plot_binned_hit_parameter_prediction_accuracy(
            data,
            config,
            eval_config,
            num_bins=5
        )
    if eval_config.get("plot_hit_param_pred_acc_heatmap", False):
        plot_hit_parameter_prediction_accuracy_heatmap(
            data,
            config,
            eval_config,

        )

def eval_slurm(config_file, eval_config_file, output_dir, accelerator, dataset):

    command = (
        (f"eggnet eval {config_file} {eval_config_file}") +
        (f" --output_dir {output_dir}" if output_dir else "") +
        (f" --accelerator {accelerator}" if accelerator else "") +
        (f" --dataset {dataset}" if dataset else "")
    )

    submit_to_slurm(command, accelerator, 1, 1, gpu_memory=40)
