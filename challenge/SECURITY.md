<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->

# Security policy

## Reporting a vulnerability

Do not open a public issue for a security problem.

Use GitHub's private vulnerability reporting on this repository, under the
Security tab. If that is unavailable to you, email **ospo@tenstorrent.com**.

Please include what you found, how to reproduce it, and what an attacker
could do with it. Proof-of-concept code is welcome and will not be shared
outside the people working on the fix.

## What happens next

| step | timing |
|---|---|
| Acknowledgement that we have received your report | within 2 business days |
| Triage, and a first assessment of severity | within 5 business days |
| Status update, and either a fix or an explanation of why not | we will keep you informed until it is closed |
| Coordinated disclosure, crediting you unless you ask otherwise | agreed with you before anything is published |

## Scope

This repository contains event material: a setup script, a webcam
demonstration server, and measurement tooling. It is not production software
and it is not part of any Tenstorrent product.

Problems in **tt-metal** itself belong at
https://github.com/tenstorrent/tt-metal rather than here.

The demonstration server in `webcam_demo/` binds to `0.0.0.0` and has no
authentication, because it is designed to be reached through the port
forwarding of a single-user cloud instance. That is a deliberate choice for
this setting rather than a defect. Do not expose it to an untrusted network.
