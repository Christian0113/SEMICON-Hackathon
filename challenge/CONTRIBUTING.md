<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->

# Contributing

## Reporting bugs

Open a [GitHub Issue](../../issues). If `setup/setup_yolov4_on_ttmetal.sh`
fails on your instance, paste the full output and say which step it stopped
at. During the hackathon, tell a mentor as well, so other teams do not lose
time to the same thing.

## Proposing changes

Bug fixes and new functionality come in as pull requests. Pull requests are
reviewed **weekly**.

During the event we are not merging, because the point is what your team did
rather than what everyone converged on. Fork it, break it, and keep your
changes in your own fork. Afterwards we will happily take pull requests,
particularly against the measurement tooling in `bench/`. If you built
something that measures this pipeline better than ours does, we would rather
use yours.

## Project-specific requirements

**Every number needs a script behind it.** If you add or change a
measurement, it must write its result to a file, record the board and the
software versions alongside, and be runnable by someone else without editing
it first. `bench/run_bench.sh` is the pattern.

**Report the tail, not only the mean.** Anything that measures latency
reports median, p95, p99 and max as well. A change that improves the average
and widens the tail has made things worse.

**Every non-empty code file carries SPDX headers**, as the two lines at the
top of any `.py` or `.sh` file in this repository show.

**Keep it dependency-free.** The demo and the tooling use only the Python
standard library plus what the Tenstorrent Console image already provides. A
change that needs a `pip install` will be refused, because it will fail on
the event machines.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Report
unacceptable behaviour to ospo@tenstorrent.com.
