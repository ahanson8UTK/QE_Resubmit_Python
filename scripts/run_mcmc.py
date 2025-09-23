from __future__ import annotations

import typer

from driver.sweep import run_sweep

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option("config/sweep.yaml", help="Path to sweep config"),
    resume: bool = typer.Option(False, help="Resume from latest checkpoint in run dir"),
) -> None:
    run_sweep(config_path=config, resume=resume)


if __name__ == "__main__":
    app()
