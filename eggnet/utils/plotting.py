import os
from typing import List

import scipy
import numpy as np
import matplotlib.pyplot as plt
from atlasify import atlasify
import atlasify as atl
import torch


def plot_eff_vs_eps(eps_data, eval_config):

    if eval_config.get("trackML_data"):
        atl.ATLAS = "TrackML Dataset"

    base_subtext = (
        (
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
            r" $t \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
        )
        if not eval_config.get("trackML_data")
        else r"$p_T > 1$GeV" + "\n"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(eps_data.eps, eps_data.eff, color="black", marker="o", linestyle=":", label="Efficiency")
    ax.plot(
        eps_data.eps, eps_data.dup, color="red", marker="o", linestyle="-.", label="Duplication rate"
    )
    ax.plot(eps_data.eps, eps_data.fak, color="blue", marker="o", linestyle="--", label="Fake rate")
    ax.set_xlabel(r"$\epsilon$", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Efficiency (Rate)", ha="right", y=0.95, fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right", fontsize=14)
    plt.tight_layout()

    # Save the plot
    atlasify(
        atlas=True if eval_config.get("trackML_data") else "Internal",
        subtext=base_subtext + "DBSCAN (min_samples = 3)",
    )
    fig.savefig(os.path.join(eval_config["output_dir"], "track_eff_dbscan_vs_eps.png"))

    print(
        "Finish plotting. Find the plot at"
        f' {os.path.join(eval_config["output_dir"], "track_eff_dbscan_vs_eps.png")}'
    )

    plt.clf()


def plot_eff_fixed_eps(matched_target_particles_hist, particles_hist, eps_data, eval_config, bins, xlabel, logx, filename):

    if eval_config.get("trackML_data"):
        atl.ATLAS = "TrackML Dataset"

    base_subtext = (
        (
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
            r" $t \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
        )
        if not eval_config.get("trackML_data")
        else r"$p_T > 1$GeV" + "\n"
    )

    eff = float(eps_data[eps_data.eps == eval_config["eps"]].eff.iloc[0])
    dup = float(eps_data[eps_data.eps == eval_config["eps"]].dup.iloc[0])
    fak = float(eps_data[eps_data.eps == eval_config["eps"]].fak.iloc[0])

    hist, err = get_ratio(matched_target_particles_hist, particles_hist)

    fig, ax = plot_1d_histogram(
        hist,
        bins,
        err,
        xlabel,
        "Track Efficiency",
        # eval_config.get("ylim", [0.7, 1.04]),
        eval_config.get("ylim", [0., 1.04]),
        "Efficiency",
        logx=logx,
        color="black",
    )

    # Save the plot
    atlasify(
        atlas=True if eval_config.get("trackML_data") else "Internal",
        subtext=base_subtext
        + r"DBSCAN ($\epsilon$"
        + f"={eval_config['eps']}, min_samples=3)"
        + "\n"
        f"Efficiency: {eff :.4f}" + "\n"
        f"Duplication rate: {dup :.4f}" + "\n"
        f"Fake rate: {fak :.4f}" + "\n",
    )
    fig.savefig(os.path.join(eval_config["output_dir"], filename))

    print(
        "Finish plotting. Find the plot at"
        f' {os.path.join(eval_config["output_dir"], filename)}'
    )

    plt.clf()


def get_ratio(passed: List[int], total: List[int]):
    if len(passed) != len(total):
        raise ValueError(
            "Length of passed and total must be the same"
            f"({len(passed)} != {len(total)})"
        )

    res = np.array([x / y if y != 0 else 0.0 for x, y in zip(passed, total)])
    error = np.array([clopper_pearson(x, y) for x, y in zip(passed, total)]).T
    return res, error


def clopper_pearson(passed: float, total: float, level: float = 0.68):
    """
    Estimate the confidence interval for a sampled binomial random variable with Clopper-Pearson.
    `passed` = number of successes; `total` = number trials; `level` = the confidence level.
    The function returns a `(low, high)` pair of numbers indicating the lower and upper error bars.
    """
    alpha = (1 - level) / 2
    lo = scipy.stats.beta.ppf(alpha, passed, total - passed + 1) if passed > 0 else 0.0
    hi = (
        scipy.stats.beta.ppf(1 - alpha, passed + 1, total - passed)
        if passed < total
        else 1.0
    )
    average = passed / total
    return (average - lo, hi - average)


def plot_1d_histogram(
    hist,
    bins,
    err,
    xlabel,
    ylabel,
    ylim,
    label,
    canvas=None,
    logx=False,
    color="black",
    fmt="o",
):
    """Plot 1D histogram from direct output of np.histogram

    Args:
        hist (_type_): _description_
        bins (_type_): _description_
        err (_type_): _description_
        xlabel (_type_): _description_
        ylabel (_type_): _description_
        ylim (_type_): _description_
        canvas (_type_, optional): tuple of (fig, ax). Defaults to None. If not provided, create fig, ax
        logx (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    xvals = (bins[1:] + bins[:-1]) / 2
    xerrs = (bins[1:] - bins[:-1]) / 2

    fig, ax = plt.subplots(figsize=(8, 6)) if canvas is None else canvas
    ax.errorbar(xvals, hist, xerr=xerrs, yerr=err, fmt=fmt, color=color, label=label)
    ax.set_xlabel(xlabel, ha="right", x=0.95, fontsize=14)
    ax.set_ylabel(ylabel, ha="right", y=0.95, fontsize=14)
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(ylim)
    plt.tight_layout()

    return fig, ax


def plot_computing_time(time_data, eval_config):

    if eval_config.get("trackML_data"):
        atl.ATLAS = "TrackML Dataset"

    base_subtext = (
        (
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
            r" $t \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
        )
        if not eval_config.get("trackML_data")
        else r"$p_T > 1$GeV" + "\n"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(time_data["num_nodes"], time_data["total"], s=2, label="Total", color="black")
    ax.scatter(time_data["num_nodes"], time_data["gnn"], s=2, label="Graph attention")
    ax.scatter(time_data["num_nodes"], time_data["knn"], s=2, label="KNN")
    ax.scatter(time_data["num_nodes"], time_data["dbscan"], s=2, label="DBScan")
    ax.set_xlabel("Number of spacepoints", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Inference time per event [s]", ha="right", y=0.95, fontsize=14)
    # ax.set_xlim([76000, 156000])
    ax.set_ylim([0, 2.1])
    plt.tight_layout()

    # Save the plot
    atlasify(
        atlas=True if eval_config.get("trackML_data") else "Internal",
        subtext=base_subtext +
        f'Averaged training time per event: {(time_data["total"]).mean():.2f}s',
    )
    fig.savefig(os.path.join(eval_config["output_dir"], "inference_time.png"))

    print(
        "Finish plotting. Find the plot at"
        f' {os.path.join(eval_config["output_dir"], "inference_time.png")}'
    )

def plot_hit_parameter_prediction_accuracy(
    data, 
    config, 
    eval_config, 
    filename="hit_parameter_pred_acc.png",
    return_fig=False, # if True, return fig, ax instead of saving the figure to filename
    mask=None # boolean mask to apply to the hits before plotting
    ):
    """
    Plot the track parameter prediction accuracy 
    """
    assert mask is None or mask.int().sum() > 0, "Mask must have at least one True value"
    event = data.get(0)
    if mask is not None:
        for dk in event.keys():
            value = event[dk]
            event[dk] = value[mask] if isinstance(value, torch.Tensor) and len(value.shape) != 0 and value.shape[0] == mask.shape[0] else value 
        # Try to apply mask to all tensors in event that have the same first dimension as mask
    if eval_config.get("trackML_data"):
        atl.ATLAS = "TrackML Dataset"
    base_subtext = (
        (
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
            r" $t \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
        )
        if not eval_config.get("trackML_data")
        else r"$p_T > 1$GeV" + "\n"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    norm_constant:torch.Tensor = event.get("normalization_constant") if config.get("parameter_loss_normalize") else 1
    try:
        norm_constant = norm_constant.squeeze().item()
    except AttributeError:
        norm_constant = 1.0
    if config.get("parameter_loss_confidence"):
        c = event.hit_parameter_confidence
        norm_c = c
        ax.scatter(event.hit_parameters.cpu(), event.get("hit_charge_pt_ratio").cpu() * norm_constant, s=2, c = norm_c, cmap="gist_heat")
    else:
        ax.scatter(event.hit_parameters.cpu(), event.get("hit_charge_pt_ratio").cpu() * norm_constant, s=2, color="black")
    ax.set_xlabel("Hit parameters", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Hit charge pt ratio", ha="right", y=0.95, fontsize=14)
    x_min, x_max = event.hit_parameters.min(), event.hit_parameters.max()
    ax.plot(
        [x_min, x_max],
        list(map(lambda x: x / float(config.get("parameter_loss_scale", 1.0)), [x_min, x_max])),
        color="red",
        linestyle="--",
        label="Ideal prediction",
    )
    
    # Calculate and plot 95% confidence interval
    x = event.hit_parameters.cpu().numpy()
    y = (event.get("hit_charge_pt_ratio").cpu().numpy() * norm_constant)
    # Fit a linear model (y = ax + b)
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    n = len(x)
    residuals = y - y_pred
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    mean_x = np.mean(x)
    t_val = 1.96  # Approximate z-value for 95% CI
    conf = t_val * s_err * np.sqrt(1/n + (x - mean_x)**2 / np.sum((x - mean_x)**2))
    ax.fill_between(x, y_pred - conf, y_pred + conf, color="gray", alpha=0.3, label="95% CI")
    if config.get("parameter_loss_confidence"):
        c_min = event.hit_parameter_confidence.min().item()
        c_max = event.hit_parameter_confidence.max().item()
        fig.colorbar(
            plt.cm.ScalarMappable(cmap="gist_heat", norm = plt.Normalize(vmin=c_min, vmax=c_max)), 
            ax=ax, boundaries=[c_min, c_max], format="%.4f")
    ax.set_xlim([-.002, .002])
    ax.set_ylim([-.001, .001])
    plt.tight_layout()
    plt.legend()
    
    # Save the plot
    atlasify(
        atlas=True if eval_config.get("trackML_data") else "Internal",
        subtext=base_subtext + "Track parameter prediction loss",
    )
    if not return_fig:
        fig.savefig(os.path.join(eval_config["output_dir"], filename))
        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(eval_config["output_dir"], filename)}'
        )
    else:
        return fig, ax 

def plot_binned_hit_parameter_prediction_accuracy(
    data,
    config,
    eval_config,
    num_bins=5
):
    # Make num_bins subplots where each subplot is the hit parameter prediction accuracy for hits in that bin.
    event = data.get(0)
    bin_dir = os.path.join(eval_config["output_dir"], f"binned_hit_param_pred_acc_({num_bins})")
    print(f"INFO: {bin_dir =}")
    os.makedirs(bin_dir, exist_ok=True)
    if eval_config.get("trackML_data"):
        atl.ATLAS = "TrackML Dataset"
    confidences = [data.get(i).get("hit_parameter_confidence") for i in range(len(data))]
    sorted_confidences = torch.cat(confidences).sort().values
    bins = torch.linspace(0, len(sorted_confidences), num_bins+1)
    for i in range(len(bins)-1):
        print("DEBUG: Plotting bin", i+1, "of", num_bins)
        _bin = bins[i]
        if i == len(bins)-2:
            _bin_next = len(sorted_confidences) - 1
        else:
            _bin_next = bins[i+1]
        low = sorted_confidences[int(_bin)]
        high = sorted_confidences[int(_bin_next)]
        if low.item() == high.item():
            print(f"WARNING: Bin {i+1} of {num_bins} has no range (all values are {low.item():.4f}). Skipping this bin.")
            continue
        print(f"DEBUG: {i=}, {_bin=}, {low=}, {high=}")
        mask = torch.logical_and(event.get("hit_parameter_confidence") >= low, event.get("hit_parameter_confidence") < high)
        print(f"DEBUG: {mask.int().sum().item()=}")
        
        fig, ax = plot_hit_parameter_prediction_accuracy(
            data,
            config,
            eval_config,
            filename="ERROR.png",
            return_fig=True,
            mask=mask
        )
        ax.set_title(f"Confidence in [{low:.4f}, {high:.4f}), Bin {i+1} of {num_bins}")
        atlasify(
            atlas=True if eval_config.get("trackML_data") else "Internal",
            subtext=(
                (
                    r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                    r" $t \bar{t}$ and soft interactions) " + "\n"
                    r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
                )
                if not eval_config.get("trackML_data")
                else r"$p_T > 1$GeV" + "\n"
            ) + rf"Track parameter prediction loss for hits with confidence in [{low:.4f}, {high:.4f})"
        )
        fig.savefig(os.path.join(eval_config["output_dir"], bin_dir, f"hit_parameter_pred_acc_bin_{i+1}.png"))
        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(eval_config["output_dir"], f"hit_parameter_pred_acc_bin_{int(_bin)}.png")}'
        )
def plot_binned_std_of_residuals(
    data,
    config,
    eval_config,
    filename="binned_std_of_residuals.png",
    num_bins=50,
    bin_datakey="hit_eta",
    x_range=None,
    plot_statistics:bool=False,
    return_fig=False
):
    """
    Plot the binned variance of residuals of track parameter prediction,
    binning with respect to a specified datakey.
    """
    event = data.get(0)
    if eval_config.get("trackML_data"):
        atl.ATLAS = "TrackML Dataset"

    base_subtext = (
        (
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
            r" $t \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
        )
        if not eval_config.get("trackML_data")
        else r"$p_T > 1$GeV" + "\n"
    )

    fig, axes = plt.subplots(ncols=(2 if plot_statistics else 1), figsize=(8, 6))
    if not plot_statistics:
        ax = axes
    else:
        ax = axes[0]
    norm_constant:torch.Tensor = event.get("normalization_constant") if config.get("parameter_loss_normalize") else 1
    try:
        norm_constant = norm_constant.squeeze().item()
    except AttributeError:
        norm_constant = 1.0
    x = event.hit_parameters.cpu().numpy()
    y = (event.get("hit_charge_pt_ratio").cpu().numpy() * norm_constant)
    residuals = y - x / float(config.get("parameter_loss_scale", 1.0))

    # Bin with respect to the specified datakey
    bin_data = event[bin_datakey].cpu().numpy()
    bin_range = (bin_data.min(), bin_data.max()) if x_range is None else x_range
    bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_indices = np.digitize(bin_data, bins) - 1

    # Calculate variance in each bin
    binned_variance = np.array([
        np.std(residuals[bin_indices == i]) if np.any(bin_indices == i) else 0
        for i in range(num_bins)
    ])
    if plot_statistics:
        binned_n = np.array([
            np.sum(bin_indices == i)
            for i in range(num_bins)
        ])

    ax.plot(bin_centers, binned_variance, marker='o', linestyle='-', color='blue')
    ax.set_xlabel(bin_datakey, ha="right", x=0.95, fontsize=14)
    ax.set_ylabel(r"$\sigma$ of residuals", ha="right", y=0.95, fontsize=14)
    plt.tight_layout()
    if plot_statistics:
        axes[1].plot(bin_centers, binned_n, marker='x', linestyle='--', color='orange', label="Normalized counts")

    # Save the plot
    atlasify(
        atlas=True if eval_config.get("trackML_data") else "Internal",
        subtext=base_subtext + rf"Binned $\sigma$ of residuals w.r.t. {bin_datakey}"
    )
    if x_range is not None:
        filename = filename.replace(".png", f"_{x_range[0]} to {x_range[1]}.png")
    if not return_fig:
        fig.savefig(os.path.join(eval_config["output_dir"], filename))
        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(eval_config["output_dir"], filename)}'
        )
    else:
        return fig, axes

def plot_binned_std_of_residuals_PT(
    data,
    config,
    eval_config,
    filename="binned_std_of_residuals_PT.png",
    num_bins=50,
    x_range=None
):
    plot_binned_std_of_residuals(
        data,
        config,
        eval_config,
        filename=filename,
        num_bins=num_bins,
        bin_datakey="hit_particle_pt",
        plot_statistics=True,
        x_range=x_range
    )

def plot_binned_std_of_residuals_ETA(
    data,
    config,
    eval_config,
    filename="binned_std_of_residuals_ETA.png",
    num_bins=50
):
    plot_binned_std_of_residuals(
        data,
        config,
        eval_config,
        filename=filename,
        num_bins=num_bins,
        plot_statistics=True,
        bin_datakey="hit_eta"
    )
def plot_binned_std_of_residuals_confidence(
    data,
    config,
    eval_config,
    filename="binned_std_of_residuals_confidence.png",
    num_bins=50
):
    plot_binned_std_of_residuals(
        data,
        config,
        eval_config,
        filename=filename,
        num_bins=num_bins,
        plot_statistics=True,
        bin_datakey="hit_parameter_confidence"
    )


# TODO
        