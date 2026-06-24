import os
from typing import List

import scipy
import numpy as np
import matplotlib.pyplot as plt
from atlasify import atlasify
import atlasify as atl


_PT_UNIT_SCALES = {
    "ev": 1.0,
    "kev": 1e3,
    "mev": 1e6,
    "gev": 1e9,
    "tev": 1e12,
}

_PT_UNIT_LABELS = {
    "ev": "eV",
    "kev": "keV",
    "mev": "MeV",
    "gev": "GeV",
    "tev": "TeV",
}


def _format_numeric(value):
    return f"{float(value):g}"


def _format_pt_value(value, unit):
    unit_key = str(unit).lower()
    scale = _PT_UNIT_SCALES.get(unit_key)
    numeric_value = float(value)

    if scale is None:
        return _format_numeric(numeric_value), str(unit)

    value_ev = numeric_value * scale
    for candidate_unit in ("tev", "gev", "mev", "kev", "ev"):
        candidate_scale = _PT_UNIT_SCALES[candidate_unit]
        candidate_value = value_ev / candidate_scale
        if np.isclose(candidate_value, round(candidate_value), atol=1e-12) and candidate_value >= 1.0:
            return _format_numeric(candidate_value), _PT_UNIT_LABELS[candidate_unit]

    candidate_value = value_ev / scale
    return _format_numeric(candidate_value), _PT_UNIT_LABELS.get(unit_key, str(unit))


def _format_pt_selection(target_tracks, pt_unit):
    pt_cut = (target_tracks or {}).get("hit_particle_pt")
    if not isinstance(pt_cut, list) or len(pt_cut) != 2:
        return None

    lower, upper = pt_cut
    lower_is_finite = np.isfinite(lower)
    upper_is_finite = np.isfinite(upper)

    if lower_is_finite:
        lower_value, lower_unit = _format_pt_value(lower, pt_unit)
    if upper_is_finite:
        upper_value, upper_unit = _format_pt_value(upper, pt_unit)

    if lower_is_finite and upper_is_finite:
        if lower_unit == upper_unit:
            return rf"${lower_value} < p_T < {upper_value}$ {lower_unit}"
        return rf"${lower_value}$ {lower_unit} $< p_T < {upper_value}$ {upper_unit}"
    if lower_is_finite:
        return rf"$p_T > {lower_value}$ {lower_unit}"
    if upper_is_finite:
        return rf"$p_T < {upper_value}$ {upper_unit}"
    return None


def _format_eta_selection(target_tracks):
    eta_cut = (target_tracks or {}).get("hit_particle_eta")
    if not isinstance(eta_cut, list) or len(eta_cut) != 2:
        return None

    lower, upper = eta_cut
    lower_is_finite = np.isfinite(lower)
    upper_is_finite = np.isfinite(upper)

    if lower_is_finite and upper_is_finite and np.isclose(lower, -upper):
        return rf"$|\eta| < {_format_numeric(abs(upper))}$"
    if lower_is_finite and upper_is_finite:
        return rf"${_format_numeric(lower)} < \eta < {_format_numeric(upper)}$"
    if lower_is_finite:
        return rf"$\eta > {_format_numeric(lower)}$"
    if upper_is_finite:
        return rf"$\eta < {_format_numeric(upper)}$"
    return None


def _get_target_selection_subtext(eval_config):
    target_tracks = eval_config.get("target_tracks") or {}
    selections = []

    pt_selection = _format_pt_selection(target_tracks, eval_config.get("pT_unit", "MeV"))
    if pt_selection is not None:
        selections.append(pt_selection)

    eta_selection = _format_eta_selection(target_tracks)
    if eta_selection is not None:
        selections.append(eta_selection)

    if not selections:
        return ""
    return ", ".join(selections) + "\n"


def _get_base_subtext(eval_config):
    if eval_config.get("trackML_data"):
        atl.ATLAS = "TrackML Dataset"

    return (
        (
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
            r" $t \bar{t}$ and soft interactions) " + "\n"
            + _get_target_selection_subtext(eval_config)
        )
        if not eval_config.get("trackML_data")
        else _get_target_selection_subtext(eval_config)
    )


def _plot_efficiency_rate_scan(
    eps_data,
    eval_config,
    xlabel,
    output_filename,
    selection_subtext,
):
    base_subtext = _get_base_subtext(eval_config)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(eps_data.eps, eps_data.eff, color="black", marker="o", linestyle=":", label="Efficiency")
    ax.plot(
        eps_data.eps, eps_data.dup, color="red", marker="o", linestyle="-.", label="Duplication rate"
    )
    ax.plot(eps_data.eps, eps_data.fak, color="blue", marker="o", linestyle="--", label="Fake rate")
    ax.set_xlabel(xlabel, ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Efficiency (Rate)", ha="right", y=0.95, fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right", fontsize=14)
    plt.tight_layout()

    # Save the plot
    atlasify(
        atlas=True if eval_config.get("trackML_data") else "Internal",
        subtext=base_subtext + selection_subtext,
    )
    fig.savefig(os.path.join(eval_config["output_dir"], output_filename))

    print(
        "Finish plotting. Find the plot at"
        f' {os.path.join(eval_config["output_dir"], output_filename)}'
    )

    plt.clf()


def plot_eff_vs_eps(eps_data, eval_config):
    _plot_efficiency_rate_scan(
        eps_data,
        eval_config,
        xlabel=r"$\epsilon$",
        output_filename="track_eff_dbscan_vs_eps.png",
        selection_subtext="DBSCAN (min_samples = 3)",
    )


def plot_walkthrough_eff_vs_cutoff(eps_data, eval_config):
    _plot_efficiency_rate_scan(
        eps_data,
        eval_config,
        xlabel="Distance cutoff",
        output_filename="track_eff_walkthrough_vs_cutoff.png",
        selection_subtext="Walkthrough distance cutoff",
    )


def plot_eff_fixed_eps(
    matched_target_particles_hist,
    particles_hist,
    eps_data,
    eval_config,
    bins,
    xlabel,
    logx,
    filename,
    selection_subtext=None,
):

    base_subtext = _get_base_subtext(eval_config)

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
    if selection_subtext is None:
        selection_subtext = (
            r"DBSCAN ($\epsilon$"
            + f"={eval_config['eps']}, min_samples=3)"
        )
    atlasify(
        atlas=True if eval_config.get("trackML_data") else "Internal",
        subtext=base_subtext
        + selection_subtext
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
    logy=False,
    color="black",
    fmt="o",
    tightlayout=True, # for backwards compatibility
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
    if logy:
        ax.set_yscale("log")
    ax.set_ylim(ylim)
    if tightlayout:
        plt.tight_layout()

    return fig, ax


def plot_computing_time(time_data, eval_config):

    if eval_config.get("trackML_data"):
        atl.ATLAS = "TrackML Dataset"

    base_subtext = (
        (
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
            r" $t \bar{t}$ and soft interactions) " + "\n"
            + _get_target_selection_subtext(eval_config)
        )
        if not eval_config.get("trackML_data")
        else _get_target_selection_subtext(eval_config)
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
        axes=ax
    )
    fig.savefig(os.path.join(eval_config["output_dir"], "inference_time.png"))

    print(
        "Finish plotting. Find the plot at"
        f' {os.path.join(eval_config["output_dir"], "inference_time.png")}'
    )
