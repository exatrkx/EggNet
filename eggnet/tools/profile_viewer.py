from pathlib import Path
import statistics
from time import perf_counter

import click
import torch


def _as_event_id(graph, fallback):
    """Extract a printable event id from a saved graph, with a fallback label."""
    if hasattr(graph, "event_id"):
        event_id = graph.event_id
        try:
            if len(event_id) > 0:
                event_id = event_id[0]
        except TypeError:
            pass
        if hasattr(event_id, "item"):
            try:
                return str(event_id.item())
            except (TypeError, ValueError):
                pass
        return str(event_id)
    return fallback


def _iter_graph_paths(input_path):
    """Yield one input file or all `.pyg` files found under a directory."""
    path = Path(input_path)
    if path.is_file():
        yield path
        return

    for graph_path in sorted(path.rglob("*.pyg")):
        if graph_path.is_file():
            yield graph_path


def _load_profiles(input_path, prefix=None):
    """Load profiling dictionaries from saved graphs under the requested input path."""
    records = []
    skipped = []

    for graph_path in _iter_graph_paths(input_path):
        try:
            graph = torch.load(graph_path, map_location="cpu")
        except Exception as exc:
            skipped.append((graph_path, f"load failed: {exc}"))
            continue

        profiling = getattr(graph, "profiling", None)
        if not isinstance(profiling, dict) or not profiling:
            skipped.append((graph_path, "missing profiling data"))
            continue

        if prefix is not None:
            profiling = {
                name: value
                for name, value in profiling.items()
                if name.startswith(prefix)
            }
            if not profiling:
                continue

        records.append(
            {
                "path": graph_path,
                "event_id": _as_event_id(graph, graph_path.stem),
                "profiling": profiling,
                "profile_metadata": getattr(graph, "profile_metadata", {}),
            }
        )

    return records, skipped


def _percent(value, total):
    """Format a value as a percentage of a reference total."""
    if total <= 0:
        return "-"
    return f"{100.0 * value / total:5.1f}%"


def _format_seconds(value):
    """Format a duration in seconds for fixed-width table output."""
    return f"{value:8.4f}s"


def _render_table(headers, rows):
    """Render a plain-text table with aligned columns."""
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    header_line = "  ".join(
        str(header).ljust(widths[idx]) for idx, header in enumerate(headers)
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _top_level_total(profiling):
    """Choose a reference total for percentage calculations in event views."""
    total_keys = sorted(
        name for name in profiling if name.endswith(".total") or name == "total"
    )
    if total_keys:
        return sum(profiling[name] for name in total_keys)
    return max(profiling.values(), default=0.0)


def _show_event(record, top):
    """Print the profiling breakdown for a single saved event."""
    profiling = record["profiling"]
    total = _top_level_total(profiling)
    rows = []

    for name, value in sorted(
        profiling.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top]:
        rows.append((name, _format_seconds(value), _percent(value, total)))

    click.echo(f"Event {record['event_id']}")
    click.echo(f"Source: {record['path']}")
    click.echo(f"Reference total: {_format_seconds(total)}")
    click.echo(_render_table(("stage", "seconds", "share"), rows))
    metadata = record.get("profile_metadata", {})
    if metadata:
        click.echo()
        metadata_rows = [
            (name, metadata[name])
            for name in sorted(metadata)
        ]
        click.echo(_render_table(("metadata", "value"), metadata_rows))


def _quantile(values, q):
    """Compute a linear-interpolated quantile from sorted numeric values."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(values) - 1)
    fraction = idx - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _collect_stage_timings(records):
    """Group stage timings across events for aggregate statistics and drill-downs."""
    by_stage = {}
    for record in records:
        for name, value in record["profiling"].items():
            by_stage.setdefault(name, []).append(
                {
                    "seconds": float(value),
                    "event_id": record["event_id"],
                    "path": record["path"],
                }
            )
    return by_stage


def _show_slowest_events(by_stage, stage_names, limit):
    """Print the slowest event ids for each selected stage."""
    if limit <= 0:
        return

    click.echo()
    click.echo(f"Slowest events per stage (top {limit})")
    for stage_name in stage_names:
        click.echo(stage_name)
        stage_rows = sorted(
            by_stage[stage_name],
            key=lambda row: row["seconds"],
            reverse=True,
        )[:limit]
        rows = [
            (
                row["event_id"],
                _format_seconds(row["seconds"]),
                row["path"],
            )
            for row in stage_rows
        ]
        click.echo(_render_table(("event_id", "seconds", "path"), rows))
        click.echo()


def _show_aggregate(records, top, slowest_events):
    """Print aggregate profiling statistics across multiple saved events."""
    by_stage = _collect_stage_timings(records)

    rows = []
    for name, entries in by_stage.items():
        values = [entry["seconds"] for entry in entries]
        values = sorted(values)
        mean_value = statistics.mean(values)
        rows.append(
            (
                name,
                len(values),
                mean_value,
                _format_seconds(mean_value),
                _format_seconds(statistics.median(values)),
                _format_seconds(_quantile(values, 0.95)),
                _format_seconds(max(values)),
            )
        )

    rows.sort(key=lambda row: row[2], reverse=True)
    rows = rows[:top]

    click.echo(f"Events with profiling data: {len(records)}")
    click.echo(
        _render_table(
            ("stage", "count", "mean", "median", "p95", "max"),
            [(row[0], row[1], row[3], row[4], row[5], row[6]) for row in rows],
        )
    )
    _show_slowest_events(
        by_stage,
        [row[0] for row in rows],
        limit=slowest_events,
    )


def run_profile_viewer(
    input_path,
    event_id=None,
    prefix=None,
    top=15,
    slowest_events=3,
):
    """Load profiling data from disk and print either an aggregate or event view."""
    start_time = perf_counter()
    input_path = Path(input_path)
    records, skipped = _load_profiles(input_path, prefix=prefix)

    if not records:
        raise click.ClickException("No profiled .pyg files found in the requested input.")

    if skipped:
        click.echo(f"Skipped {len(skipped)} file(s) without usable profiling data.", err=True)

    if input_path.is_file():
        _show_event(records[0], top=top)
        click.echo()
        click.echo(f"Viewer runtime: {_format_seconds(perf_counter() - start_time)}")
        return

    if event_id is not None:
        for record in records:
            if record["event_id"] == str(event_id):
                _show_event(record, top=top)
                click.echo()
                click.echo(
                    f"Viewer runtime: {_format_seconds(perf_counter() - start_time)}"
                )
                return
        raise click.ClickException(f"Could not find event_id={event_id} in {input_path}.")

    _show_aggregate(records, top=top, slowest_events=slowest_events)
    click.echo()
    click.echo(f"Viewer runtime: {_format_seconds(perf_counter() - start_time)}")


@click.command("profile")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
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
def profile(input_path, event_id, prefix, top, slowest_events):
    """View saved profiling output from one .pyg file or a directory of them."""
    return run_profile_viewer(
        input_path=input_path,
        event_id=event_id,
        prefix=prefix,
        top=top,
        slowest_events=slowest_events,
    )


if __name__ == "__main__":
    profile()
