"""CLI wrapper for running NDE merges.

Imports nde.py to register the 'nde' merge method, then delegates
to mergekit's standard run_merge pipeline.

Usage:
    python -m longcat_merge.run <config.yaml> <out_path> [options]
"""

import click
import yaml

import longcat_merge.nde  # noqa: F401  — triggers @merge_method registration

from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options


@click.command("nde-merge", cls=PrettyPrintHelp)
@click.argument("config_file")
@click.argument("out_path")
@add_merge_options
def main(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
):
    merge_options.apply_global_options()

    with open(config_file, "r", encoding="utf-8") as f:
        config_source = f.read()

    merge_config = MergeConfiguration.model_validate(yaml.safe_load(config_source))
    run_merge(
        merge_config,
        out_path,
        options=merge_options,
        config_source=config_source,
    )


if __name__ == "__main__":
    main()
