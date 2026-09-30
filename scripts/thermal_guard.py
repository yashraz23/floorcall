"""Run a long GPU command under the thermal rules (DECISIONS.md D-037).

    uv run python scripts/thermal_guard.py -- uv run floorcall eval robustness --checkpoint ...

Waits until the GPU is at or below `EvalSettings.latency_start_max_temp_c`, then runs the command
and reads the GPU temperature every 2 seconds. If it reaches `latency_abort_temp_c`, the command is
stopped and this exits non-zero. Prints the peak temperature at the end. It writes nothing else: the
command's own outputs are what count.
"""

from __future__ import annotations

import subprocess
import sys
import time

from floorcall.config import get_settings
from floorcall.evaluate.latency import read_gpu_temp, wait_until_cool


def main() -> int:
    if "--" not in sys.argv:
        sys.exit("usage: thermal_guard.py -- <command ...>")
    cmd = sys.argv[sys.argv.index("--") + 1 :]
    ev = get_settings().eval
    wait_until_cool(
        ev.latency_start_max_temp_c,
        poll_s=ev.latency_cooldown_poll_s,
        max_wait_s=ev.latency_cooldown_max_s,
    )
    proc = subprocess.Popen(cmd)
    peak = 0.0
    while proc.poll() is None:
        t = read_gpu_temp()
        if t is None:
            proc.terminate()
            print("thermal guard: lost the GPU temperature reading; stopped the command")
            return 2
        peak = max(peak, t)
        if t >= ev.latency_abort_temp_c:
            proc.terminate()
            proc.wait(timeout=60)
            print(f"thermal guard: GPU at {t:.0f} C >= {ev.latency_abort_temp_c:.0f} C; stopped")
            return 3
        time.sleep(2)
    print(f"thermal guard: finished, exit {proc.returncode}, GPU peak {peak:.0f} C")
    return int(proc.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
