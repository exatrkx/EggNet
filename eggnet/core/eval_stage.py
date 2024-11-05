import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd

from eggnet import lightning_modules
from eggnet.utils.cluster import cluster_and_match
from eggnet.utils.plotting import plot_eff_vs_eps, plot_eff_fixed_eps, plot_computing_time


def eval(config_file, eval_config_file, output_dir, accelerator, dataset):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if output_dir is not None:
        config["output_dir"] = output_dir
    with open(eval_config_file, "r") as f:
        eval_config = yaml.load(f, Loader=yaml.FullLoader)
    eval_config["output_dir"] = config["output_dir"]

    base_model = getattr(lightning_modules, config.get("base_model", "NodeEncoding"))(config)
    base_model.setup(stage="test", datasets=[dataset])
    data = getattr(base_model, dataset)

    eps_data = pd.DataFrame({
        "eps": np.arange(0.05, 0.51, 0.05),
        "n_particles": 0,
        "n_matched_particles": 0,
        "n_matched_tracks": 0,
        "n_matched_target_particles": 0,
        "n_matched_target_tracks": 0,
        "n_tracks": 0,
    })

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

    for event in tqdm(data):
        event = event.to(accelerator)

        for eps_i in eps_data.eps:
            eps_data_i, particles_pt_hist_i, matched_target_particles_pt_hist_i, particles_eta_hist_i, matched_target_particles_eta_hist_i = cluster_and_match(event, eps_i, eval_config, time_yes=True if eps_i == eval_config["eps"] else False)

            eps_data[eps_data.eps == eps_i] = eps_data[eps_data.eps == eps_i].to_numpy() + eps_data_i.to_numpy()

            if eps_i == eval_config["eps"]:
                particles_pt_hist += particles_pt_hist_i
                matched_target_particles_pt_hist += matched_target_particles_pt_hist_i
                if eval_config.get("plot_eta", True):
                    particles_eta_hist += particles_eta_hist_i
                    matched_target_particles_eta_hist += matched_target_particles_eta_hist_i

        time_data = pd.concat([time_data, pd.DataFrame({
            "num_nodes": [event["num_nodes"].cpu()],
            "eggnet": [event["BaseModule.forward"]],
            "knn": [event["get_knn_graph"]],
            "dbscan": [event["cluster"]],
        })])

    eps_data["eff"] = eps_data.n_matched_target_particles / eps_data.n_particles
    eps_data["dup"] = (
        eps_data.n_matched_target_tracks - eps_data.n_matched_target_particles
    ) / eps_data.n_matched_target_particles
    eps_data["fak"] = (eps_data.n_tracks - eps_data.n_matched_tracks) / eps_data.n_matched_particles

    time_data["gnn"] = time_data["eggnet"] - time_data["knn"]
    time_data["total"] = time_data["eggnet"] + time_data["dbscan"]

    plot_eff_vs_eps(eps_data, eval_config)
    plot_eff_fixed_eps(matched_target_particles_pt_hist, particles_pt_hist, eps_data, eval_config, pt_bins, f"$p_T$ [{eval_config.get('pT_unit', 'MeV')}]", logx=True, filename="track_efficiency_pt.png")
    if eval_config.get("plot_eta", True):
        plot_eff_fixed_eps(matched_target_particles_eta_hist, particles_eta_hist, eps_data, eval_config, eta_bins, r"$\eta$", logx=False, filename="track_efficiency_eta.png")
    plot_computing_time(time_data, eval_config)
