from __future__ import annotations

import pandas as pd
import psutil

import pdcast
from pdcast.util.type_hints import datetime_like, timedelta_like


def parse_memory_limit(memory_limit: int | float) -> int:
    """Allows users to specify a memory limit as a fraction of total system
    resources.

    This function is used to parse the ``'memory_limit'`` argument of automl
    model fits.
    """
    # parse memory_limit
    if isinstance(memory_limit, float):
        total_memory = psutil.virtual_memory().total // (1024**2)
        result = int(memory_limit * total_memory)
    else:
        result = memory_limit

    # ensure positive
    if memory_limit < 0:
        raise ValueError(
            f"'memory_limit' must be positive, not {memory_limit}"
        )

    return result


def parse_n_jobs(n_jobs: int) -> int:
    """Allows users to specify a memory limit as a fraction of total system
    resources.

    This function is used to parse the ``'n_jobs'`` argument of automl model
    fits.
    """
    # trivial case: default to single thread
    if n_jobs is None:
        return 1

    # parse fraction of system resources
    if isinstance(n_jobs, float):
        if not 0 < n_jobs < 1:
            raise ValueError(
                f"If 'n_jobs' is a fraction, it must be between 0 and 1, not "
                f"{n_jobs}"
            )
        return int(n_jobs * psutil.cpu_count())

    # parse integer
    if n_jobs == -1:
        return psutil.cpu_count()
    if n_jobs < 1:
        raise ValueError(f"'n_jobs' must be positive, not {n_jobs}")
    return n_jobs


def parse_time_limit(
    time_limit: int | str | datetime_like | timedelta_like
) -> int:
    """Convert an arbitrary time limit into an integer number of seconds from
    runtime.

    This function is used to parse the ``'time_limit'`` argument of automl
    model fits.
    """
    # trivial case: integer seconds
    if isinstance(time_limit, int):
        result = time_limit

    # parse datetime/timedelta
    else:
        if isinstance(time_limit, str):
            result = pdcast.cast(time_limit, "datetime", tz="local")[0]
        else:
            result = time_limit

        result = pdcast.cast(
            result,
            "int[python]",
            unit="s",
            since=pd.Timestamp.utcnow()
        )

        # ensure scalar
        if len(result) != 1:
            raise ValueError(f"'time_limit' must be scalar, not {time_limit}")
        result = result[0]

    # ensure positive
    if result < 0:
        raise ValueError(f"'time_limit' must be positive, not {time_limit}")

    return result


def shorten_labels(
    labels: list,
    max_length: int,
    sep: str
) -> str:
    """Generate a concatenated string representing the features/targets being
    used in an AutoML model fit.
    """
    if len(labels) > max_length:
        result = sep.join(str(x) for x in labels[:max_length // 2])
        result += " ... "
        result += sep.join(str(x) for x in labels[-(max_length // 2):])
    else:
        result = sep.join(str(x) for x in labels)

    return result
