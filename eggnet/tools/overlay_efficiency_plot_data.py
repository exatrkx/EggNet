import json
import os
from pathlib import Path
import re

import click
import matplotlib.pyplot as plt
import numpy as np
from atlasify import atlasify
from matplotlib import font_manager

from eggnet.utils.plotting import _get_base_subtext


DEFAULT_STYLES = (
    {
        "color": "black",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "color": "red",
        "marker": "s",
        "linestyle": "--",
    },
)

PREFERRED_METRICS_FONTS = ("Helvetica", "Arial", "DejaVu Sans")
METRICS_FONT_SIZE = 9
METRICS_X = 0.03
METRICS_Y_BOTTOM = 0.03
METRICS_Y_TOP = 0.78
METRICS_BOX_PADDING = 0.015


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _default_label(path, payload):
    source = payload.get("source", {})
    checkpoint = source.get("checkpoint")
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        checkpoint_str = str(checkpoint_path)
        epoch_match = re.search(r"-epoch=(\d+)", checkpoint_path.name)
        epoch_suffix = f" ({epoch_match.group(1)} epochs)" if epoch_match else ""

        if "/experiment/dml_full/" in checkpoint_str:
            return f"DML-Full{epoch_suffix}"
        if "/experiment/double_metric_learning/" in checkpoint_str:
            return f"DML-HC{epoch_suffix}"
        if "/experiment/control/full/" in checkpoint_str:
            return f"Control-Full{epoch_suffix}"
        if "/experiment/control/hardcut/" in checkpoint_str:
            return f"Control-HC{epoch_suffix}"
        return checkpoint_path.stem
    return Path(path).stem


def _extract_plot_payload(payload, plot_key):
    plots = payload.get("plots", {})
    return plots.get(plot_key)


def _shared_plot_keys(payload_a, payload_b):
    ordered_keys = ("eff_fixed_eps_pt", "eff_fixed_eps_eta")
    return [
        key
        for key in ordered_keys
        if _extract_plot_payload(payload_a, key) is not None
        and _extract_plot_payload(payload_b, key) is not None
    ]


def _plot_one_overlay(ax, plot_a, plot_b, label_a, label_b, style_a, style_b):
    for plot_payload, label, style in (
        (plot_a, label_a, style_a),
        (plot_b, label_b, style_b),
    ):
        x = np.asarray(plot_payload["bin_centers"], dtype=np.float64)
        xerr = np.asarray(plot_payload["bin_half_widths"], dtype=np.float64)
        y = np.asarray(plot_payload["efficiency"], dtype=np.float64)
        yerr_low = np.asarray(plot_payload["efficiency_err_low"], dtype=np.float64)
        yerr_high = np.asarray(plot_payload["efficiency_err_high"], dtype=np.float64)

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=np.vstack((yerr_low, yerr_high)),
            fmt=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            markersize=4.0,
            capsize=0.0,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.0,
            label=label,
        )

    ax.set_xlabel(plot_a["xlabel"])
    ax.set_ylabel(plot_a["ylabel"])
    ax.set_ylim(plot_a.get("ylim", [0.0, 1.04]))
    ax.set_xlim(
        float(plot_a["bin_edges"][0]),
        float(plot_a["bin_edges"][-1]),
    )
    if bool(plot_a.get("logx", False)):
        ax.set_xscale("log")


def _metrics_text(plot_payload, label):
    metrics = plot_payload["fixed_eps_metrics"]
    return (
        f"{label}: "
        f"eff={metrics['eff']:.4f}, "
        f"dup={metrics['dup']:.4f}, "
        f"fake={metrics['fak']:.4f}"
    )


def _atlas_subtext(payload, plot_payload):
    source = payload.get("source", {})
    eval_config = {
        "trackML_data": bool(source.get("trackML_data", False)),
    }
    base_subtext = _get_base_subtext(eval_config)
    selection_subtext = plot_payload.get("selection_subtext", "")
    return base_subtext + selection_subtext


def _draw_metrics_text(fig, label_a, plot_a, label_b, plot_b):
    metrics_text = _metrics_text(plot_a, label_a) + "\n" + _metrics_text(plot_b, label_b)
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    metrics_font_family = next(
        (font_name for font_name in PREFERRED_METRICS_FONTS if font_name in available_fonts),
        "sans-serif",
    )

    axes = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidate_positions = (
        (METRICS_X, METRICS_Y_BOTTOM, "bottom"),
        (METRICS_X, METRICS_Y_TOP, "top"),
    )

    def _metrics_box_score(candidate):
        x_pos, y_pos, vertical_alignment = candidate
        probe_text = axes.text(
            x_pos,
            y_pos,
            metrics_text,
            ha="left",
            va=vertical_alignment,
            fontsize=METRICS_FONT_SIZE,
            fontfamily=metrics_font_family,
            linespacing=1.25,
            transform=axes.transAxes,
            alpha=0.0,
        )
        bbox_display = probe_text.get_window_extent(renderer=renderer)
        probe_text.remove()

        bbox_axes = axes.transAxes.inverted().transform(bbox_display.get_points())
        (x0, y0), (x1, y1) = bbox_axes
        x0 -= METRICS_BOX_PADDING
        y0 -= METRICS_BOX_PADDING
        x1 += METRICS_BOX_PADDING
        y1 += METRICS_BOX_PADDING

        plot_points = []
        for plot_payload in (plot_a, plot_b):
            x_vals = np.asarray(plot_payload["bin_centers"], dtype=np.float64)
            y_vals = np.asarray(plot_payload["efficiency"], dtype=np.float64)
            plot_points.append(np.column_stack((x_vals, y_vals)))
        plot_points = np.vstack(plot_points)
        plot_points_axes = axes.transAxes.inverted().transform(
            axes.transData.transform(plot_points)
        )

        x_dist = np.maximum(np.maximum(x0 - plot_points_axes[:, 0], 0.0), plot_points_axes[:, 0] - x1)
        y_dist = np.maximum(np.maximum(y0 - plot_points_axes[:, 1], 0.0), plot_points_axes[:, 1] - y1)
        distances = np.sqrt(x_dist**2 + y_dist**2)
        overlaps = int(np.count_nonzero(distances == 0.0))
        min_distance = float(distances.min())
        mean_distance = float(distances.mean())
        return (overlaps == 0, min_distance, mean_distance)

    x_pos, y_pos, vertical_alignment = max(
        candidate_positions,
        key=_metrics_box_score,
    )

    axes.text(
        x_pos,
        y_pos,
        metrics_text,
        ha="left",
        va=vertical_alignment,
        fontsize=METRICS_FONT_SIZE,
        fontfamily=metrics_font_family,
        linespacing=1.25,
        transform=axes.transAxes,
    )


def _output_path_for_plot_key(output_path, plot_key, multiple_outputs):
    if output_path is None:
        base_path = os.path.abspath("overlay_efficiency_plot.png")
    else:
        base_path = os.path.abspath(output_path)

    if not multiple_outputs:
        return base_path

    path = Path(base_path)
    suffix = "_pt" if plot_key.endswith("_pt") else "_eta"
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def overlay_efficiency_plot_data(
    json_a,
    json_b,
    output_path=None,
    label_a=None,
    label_b=None,
    title=None,
):
    payload_a = _load_json(json_a)
    payload_b = _load_json(json_b)

    plot_keys = _shared_plot_keys(payload_a, payload_b)
    if not plot_keys:
        raise ValueError(
            "The two JSON files do not share any overlayable fixed-eps plot payloads. "
            "Expected at least one of: eff_fixed_eps_pt, eff_fixed_eps_eta."
        )

    label_a = label_a or _default_label(json_a, payload_a)
    label_b = label_b or _default_label(json_b, payload_b)

    output_paths = []
    multiple_outputs = len(plot_keys) > 1

    for plot_key in plot_keys:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_a = _extract_plot_payload(payload_a, plot_key)
        plot_b = _extract_plot_payload(payload_b, plot_key)
        _plot_one_overlay(
            ax,
            plot_a,
            plot_b,
            label_a,
            label_b,
            DEFAULT_STYLES[0],
            DEFAULT_STYLES[1],
        )

        ax.legend(frameon=False, loc="upper right")
        plt.sca(ax)
        atlasify(
            atlas=True if payload_a.get("source", {}).get("trackML_data") else "Internal",
            subtext=_atlas_subtext(payload_a, plot_a),
        )

        if title:
            fig.suptitle(title, fontsize=14, y=0.99)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        else:
            fig.tight_layout()

        _draw_metrics_text(fig, label_a, plot_a, label_b, plot_b)

        plot_output_path = _output_path_for_plot_key(
            output_path,
            plot_key,
            multiple_outputs=multiple_outputs,
        )
        os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
        fig.savefig(plot_output_path, dpi=150)
        plt.close(fig)
        output_paths.append(plot_output_path)
        print(f"INFO: Saved overlay figure to {plot_output_path}")

    return output_paths[0] if len(output_paths) == 1 else output_paths


@click.command()
@click.argument("json_a")
@click.argument("json_b")
@click.option(
    "--output-path",
    default=None,
    help="Output figure path. If both pT and eta are present, _pt/_eta suffixes are added.",
)
@click.option(
    "--label-a",
    default=None,
    help="Legend label for the first JSON. Defaults to checkpoint stem or file stem.",
)
@click.option(
    "--label-b",
    default=None,
    help="Legend label for the second JSON. Defaults to checkpoint stem or file stem.",
)
@click.option(
    "--title",
    default=None,
    help="Optional overall figure title.",
)
def main(**kwargs):
    return overlay_efficiency_plot_data(**kwargs)


if __name__ == "__main__":
    main()
