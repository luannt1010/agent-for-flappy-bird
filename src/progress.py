"""Terminal progress bars shared by training, evaluation and the benchmark CLI."""

import sys

from tqdm import tqdm


def create_progress(total: int, description: str, unit: str, *, enabled: bool = False,
                    position: int = 0, leave: bool = False) -> tqdm:
    """Return a tqdm bar with completed/total, elapsed time, throughput and ETA.

    ``total`` counts runs, aggregate bird actions, episodes or races as specified
    by ``unit``. Callers update using actual completed work and close the bar in
    a context manager or finally block. ``enabled=False`` silences rendering;
    benchmark counters and results must never depend on the bar's internal n.
    Positions let an overall bar contain train and validation bars. Periodic
    redraws are throttled to twice a second to limit terminal overhead.
    """
    return tqdm(total=total, desc=description, unit=unit, disable=not enabled,
                position=position, leave=leave, dynamic_ncols=True, mininterval=0.5)


def progress_write(message: str) -> None:
    """Print one log line without overwriting active tqdm bars; return None."""
    tqdm.write(message)
    sys.stdout.flush()
