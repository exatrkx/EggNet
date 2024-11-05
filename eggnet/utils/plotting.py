import os
from typing import List

import scipy
import numpy as np
import matplotlib.pyplot as plt
from atlasify import atlasify
import atlasify as atl


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
