# Claude Code Guidelines

Welcome Claude! You are working on the **TUM Venture Labs × SEMICON Europa Hackathon (Tenstorrent Track)**.

## Core Directives
1. **Master Context:** Carefully review `AGENTS.md` in the repository root for the full challenge breakdown, target metrics (< 15 ms latency on Wormhole n150), and evaluation criteria.
2. **Challenge Docs:** Refer to the supervisor documents in the root:
   - `NextSemi Hackathon_Tenstorrent_Challenge.pdf`
   - `Tenstorrent_Technical Details_Fact Pack.pdf`
3. **Collaboration:** You are collaborating with teammates using Google Antigravity and OpenAI Codex.
   - Always commit to your feature branch (e.g., `feat/claude/<feature>`).
   - Format all Python changes with `black` or `ruff` before completing tasks.
   - Preserve existing function signatures in shared modules.
   - Document any speed vs. accuracy trade-off measurements to feed the required 1-page log.
