"""Agent-agnostic fix for a real bug found in harbor==0.23.0, not a
Claude-specific patch: this repo's Harbor tasks (mace-mof0 and
mace-mof0-smoke) never import an agent implementation directly. They stay
generic -- "submit to remote_job.py, write submission.json" -- and select an
agent purely via `harbor run -a <name-or-import-path> -m <model>`. Nothing
about the scientific task changes when the agent changes.

THE BUG (found by actually running `harbor run` twice against real images,
not by inspection -- see benchmark/harbor/README.md "Known blockers"):
several of harbor's own `BaseInstalledAgent` subclasses (confirmed for both
`ClaudeCode` and `Codex`) gate their real install step behind
`command -v <binary> >/dev/null 2>&1` run through `environment.exec()`, and
skip installing if that reports success. In this sandbox that check reported
success even when the binary was not installed -- reproduced identically
against a custom image AND Harbor's own generic `python:3.13-slim` check
image, so it is not specific to anything in this repo's Dockerfiles. The
result: the real install is skipped, and the agent invocation fails with
"<binary>: command not found" before doing any work.

THE FIX: bake each supported agent's CLI into the environment image at BUILD
TIME (see environment/Dockerfile's node/npm + claude-code + codex install
steps), and swap each agent's install() step for one that only *verifies*
the binary is already present and loudly errors if it somehow is not --
never attempting a runtime install, so there is nothing left for the buggy
check to gate. This is a mixin, not a per-agent copy-paste, specifically so
adding a third or fourth supported agent (Gemini CLI, OpenHands, ...) is one
small subclass plus one Dockerfile RUN step, never a task change.

Usage (see environment/Dockerfile for what's actually pre-installed):
    PYTHONPATH=<repo_root> harbor run -p <task> \\
        -a benchmark.harbor.agents.preinstalled:PreinstalledClaudeCode -m <model>
    PYTHONPATH=<repo_root> harbor run -p <task> \\
        -a benchmark.harbor.agents.preinstalled:PreinstalledCodex -m <model>
"""
from __future__ import annotations

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment


class PreinstalledAgentMixin:
    """Drop-in replacement for `BaseInstalledAgent.install()`.

    Verifies the CLI is already present (via the agent's own, already
    -implemented `get_version_command()`) instead of trying to install
    anything -- see this module's docstring for why. Put this FIRST in a
    subclass's bases (`class Foo(PreinstalledAgentMixin, RealAgent)`) so MRO
    resolves `install` here, not on `RealAgent`.
    """

    async def install(self, environment: BaseEnvironment) -> None:
        version_command = self.get_version_command()  # type: ignore[attr-defined]
        if version_command is None:
            # This agent class has no version-check contract at all; trust
            # the image and do nothing, same as a successful check would.
            return
        result = await environment.exec(command=version_command)
        if result.return_code != 0:
            raise RuntimeError(
                f"{type(self).__name__}: expected the CLI to already be installed in "
                f"this environment image (see environment/Dockerfile) -- this class "
                f"deliberately never attempts a runtime install (see "
                f"benchmark/harbor/agents/preinstalled.py). "
                f"{version_command!r} exited {result.return_code}: "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )


class PreinstalledClaudeCode(PreinstalledAgentMixin, ClaudeCode):
    """Claude Code, assuming environment/Dockerfile already installed it."""


class PreinstalledCodex(PreinstalledAgentMixin, Codex):
    """OpenAI Codex CLI, assuming environment/Dockerfile already installed it.

    UNVERIFIED as of this writing: environment/Dockerfile does install
    Node.js + `npm install -g @openai/codex`, and this class was written and
    reviewed against harbor.agents.installed.codex.Codex's real interface
    (get_version_command/install signatures), but no `harbor run` with this
    class has actually been executed -- see benchmark/harbor/README.md's
    verification-status table before relying on it for a real trial.
    """
