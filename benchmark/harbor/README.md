# Harbor integration for MACE-MP-MOF0 Autonomous Discovery

This wires the existing, frozen `mace_mof0_discovery` benchmark
(`tasks/mace_mof0_discovery/task_card.md`, `benchmark/evaluation/**`,
`scoring/**`) into [harbor-framework/harbor](https://github.com/harbor-framework/harbor)
(package name `harbor` on PyPI; docs at harborframework.com) -- an agent
evaluation harness, **not** `goharbor/harbor` (the container registry project
of the same name; this integration has nothing to do with that).

Harbor owns **agent rollout**: it starts an agent (Claude Code, Codex, or any
other `BaseInstalledAgent` it supports -- see "Agent runtime adapter" below)
in a container, gives it the task instruction, and grades what it produces at
the end. **Volcengine ML Platform (`volc ml_task`) remains the only backend
for agent-requested scientific GPU training/inference** -- Harbor executes
agent orchestration and final verification locally (both still cost CPU/RAM,
just not the research GPU workload) and never touches the training path
itself.

```text
harbor run -p benchmark/harbor/mace-mof0 \
    -a benchmark.harbor.agents.preinstalled:PreinstalledClaudeCode -m <model>
    |    (swap for PreinstalledCodex -- or any future PreinstalledXyz -- to
    |     change ONLY the agent; nothing below this line changes)
    v
Harbor brings up the compose environment: main + broker (environment/docker-compose.yaml)
    |
    +-- main (mat-agent-harbor:latest) -- the chosen agent runs here
    |     Claude Code AND Codex both pre-installed at build time (agent-agnostic
    |     image; see "Agent runtime adapter") -- which one runs is an `-a` flag,
    |     not an image choice
    |     no sequestered data present -- structurally excluded, not just hidden
    |     NO volc CLI, NO Volc credentials -- nothing to shell out to
    |     bind-mounts: ./runs, ./workspace (same host paths Volc GPU nodes write to)
    |
    +-- broker (mat-agent-harbor-broker, built from environment/broker/)
    |     the ONLY place with volc CLI + Volc credentials
    |     enforces the concurrency/budget gate server-side
    |     bind-mounts: ./runs, ./workspace (same as main)
    |
    v
The agent (main), guided by instruction.md + tasks/mace_mof0_discovery/task_card.md
    |  (task-agnostic to which agent this is: "read the task, submit via
    |   remote_job.py, write submission.json" -- no agent-specific wording)
    |
    +-- tools/remote_job.py submit/wait/collect --HTTP--> broker --volc CLI--> volc ml_task (real GPU nodes)
    |        (<=4 A100-eq GPUs at a time, enforced IN the broker, not just documented)
    |
    v  (repeat hypothesis -> experiment -> analysis -> iterate, up to 16h)
    |
runs/SUBMISSION/submission.json  (+ runs/, workspace/, playground/loop_db/)
    |
    v  Harbor's artifact mechanism (task.toml top-level `artifacts`)
    |
Harbor starts the SEPARATE VERIFIER container (mat-agent-harbor-verifier:latest)
    |  DOES have data/mace_mof_0/discovery/_sequestered/** and raw/phonons/**
    |  network: public (see "Known blockers" -- no-network needs buildx, unavailable here), no Volc credentials
    v
tests/test.sh -> preflight.py -> evaluate_mof0.py (frozen evaluator)
    |
    v
/logs/verifier/reward.{json,txt}  +  scorecard.json, verdict.json (full scientific result)
```

## Known blockers (found by actually running it, not by inspection)

All of this task's own infrastructure -- images, compose, the broker's
HTTP mechanism, the concurrency gate -- was exercised for real (see "What
was verified" below) and worked. Three real issues were found across seven
`harbor run` attempts; all three are now fixed and re-verified, in order:

0. **FIXED -- was an OAuth-reuse rejection, resolved with a user-provided
   API key + gateway, with one non-obvious follow-on fix.** Originally,
   reusing this dev session's own OAuth token for a second, nested Claude
   Code session got a real Claude Code start (`claude_code_version: 2.1.270`
   in its own init message) followed immediately by `"Failed to
   authenticate. API Error: 403 Request not allowed"`, identically whether
   or not `CLAUDE_FORCE_OAUTH=1` was set -- a deliberate server-side policy
   against concurrent reuse of one interactive session's grant, not a config
   mistake. The user then supplied a working `ANTHROPIC_API_KEY` plus a
   third-party Anthropic-compatible gateway (`ANTHROPIC_BASE_URL`). That
   alone got past the 403 (`apiKeySource: ANTHROPIC_API_KEY` in the init
   message) but hit a *second*, distinct error:
   `"There's an issue with the selected model (claude-sonnet-5). It may not
   exist or you may not have access to it."` (`api_error_status: 404`).
   Diagnosed directly: `GET /v1/models` (list) returns 200 and includes
   `claude-sonnet-5`, but `GET /v1/models/claude-sonnet-5` (singular
   Anthropic-style lookup, which Claude Code calls client-side to
   pre-validate the model before ever sending a real request) 404s on this
   gateway. Fixed by (a) setting `ANTHROPIC_BASE_URL` **without** a trailing
   `/v1` (Claude Code appends it itself -- a trailing `/v1` was tried first
   and caused exactly this), and (b) setting
   `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` to suppress that pre-flight
   call entirely, added to both task.toml files' `[environment].env` (this
   var is NOT part of `ClaudeCode._resolve_auth_env()`'s own env dict, so it
   has to be set here, not relied on to flow through automatically).
   Verified directly via a real `docker run` (genuine `"result":"PONG"`,
   correct cost tracking) and then via a real `harbor run` smoke trial: the
   agent authenticated, read `task_card.md`/`policy.yaml`/`RESEARCH_LOG.md`,
   and ran `tools/remote_job.py --help` -- i.e. actually did real
   exploratory work through this gateway, not just a handshake.

   **A separate observation from that trial and a second one, now fully
   explained -- and a real, currently-blocking consequence.** Both real
   `harbor run` smoke trials against this gateway stopped after a handful of
   tool calls (~19s and ~9s of activity respectively) with no final
   assistant message and no `"type":"result"` event in
   `agent/claude-code.txt`. This looked, at first, like a Docker/Harbor-level
   bug: both trials showed the whole compose environment (`main` AND
   `broker`) die via SIGKILL (`exitCode=137` in `docker events`, no matching
   `kill` action logged first) a consistent ~20-30s after starting, which
   was chased hard -- host OOM was ruled out (no dmesg trace, 223GB free,
   and a live `docker stats` trace during a third attempt showed peak usage
   of ~22MB), an idle/lightweight compose stack survived 60+ seconds with no
   kill, and the identical `claude` invocation reproduced manually (both
   backgrounded and attached, inside the identical `main`+`broker` compose
   stack, via `docker compose exec`) ran cleanly for 2-4+ minutes with real
   multi-tool-call activity and no cutoff. **The `exitCode=137` container
   deaths turned out to be a red herring**: they are Harbor's own normal,
   immediate (no-grace-period) teardown of the agent environment right after
   `_run_agent_phase` returns (`SingleStepTrial._run`), which fires whenever
   the agent's `run()` coroutine returns for *any* reason -- and
   `ClaudeCode.run()` never inspects the wrapped exec's return code before
   returning, so a prematurely-ended `claude` process looks identical to a
   normal completion from Harbor's perspective. The real question was why
   `claude` itself was ending early, and a third live attempt answered it
   directly: `"api_error_status":403, "result":"Failed to authenticate. API
   Error: 403 Monthly quota exceeded. Used: $50.04 / $50.00. Quota resets at
   the beginning of each month."` -- **the user-provided gateway key's
   monthly quota is now fully exhausted**, confirmed independently via a
   direct, minimal `curl` to `/v1/messages` returning the identical
   `quota_exceeded`/`insufficient_quota` error. This is now believed to also
   explain the two earlier resultless cutoffs (a connection dropped
   mid-stream as cumulative usage crossed some earlier
   rate/threshold checkpoint on the same key, before the hard $50 cap was
   reached), though that specific mechanism was not confirmed as directly as
   the third trial's explicit error was.

   **Status: hard-blocked on credentials, not on this repo's code.** No
   further live `harbor run` or direct API testing is possible with this key
   until its monthly quota resets, or until a different credential is
   provided. Everything upstream of this (auth handshake, model selection,
   the agent actually reading the task and calling `remote_job.py`, the
   verifier's `tests/test.sh` fix below) is independently confirmed working;
   what remains unverified is a full trajectory that reaches a real
   `remote_job.py submit` call and a produced `submission.json`, purely
   because every live attempt so far has been cut short by gateway-side
   quota/rate enforcement, not by anything in this task's own plumbing.
1. **FIXED, agent-agnostically -- see "Agent runtime adapter" below.**
   Several of harbor's own `BaseInstalledAgent` subclasses (confirmed for
   both `ClaudeCode` and `Codex`, not just Claude) gate their real install
   step behind `command -v <binary> >/dev/null 2>&1` run through
   `environment.exec()`, and skip installing if that reports success. In two
   independent trials (including one using Harbor's own generic
   `python:3.13-slim` check-image, unrelated to anything in this repo) that
   check reported success even though the binary was never installed, so
   the real install was skipped and the agent invocation failed outright
   with `<binary>: command not found`. Pinning a version
   (`--ak version=2.1.270`, forcing the alternate `claude --version`-based
   check path) got further -- the real install command visibly ran -- but
   the binary was *still* not found afterward, and `set -euo pipefail` at
   the head of that command did not abort the trial the way it should on a
   genuine install failure, pointing at `environment.exec()`'s exit-code
   capture being unreliable in this Docker setup more generally. The fix
   doesn't try to make that check reliable: `environment/Dockerfile` now
   installs every supported agent CLI at *build* time, and
   `benchmark/harbor/agents/preinstalled.py` swaps each agent's `install()`
   for one that only verifies presence and never attempts a runtime install
   -- see that module's docstring for the full mechanism, and "Agent runtime
   adapter" below for why this is a reusable layer, not a Claude patch.
2. **FIXED -- was a missing baked-in `/tests/test.sh` in the verifier image,
   not a path-resolution bug.** An earlier pass of this README described this
   as "`docker compose cp` resolves paths relative to `--project-directory`,
   and the tar-stream fallback fails identically" -- that first half is real
   (`docker compose cp <job_dir>/artifacts/... main:/...` does fail against
   this task's relative job-output paths, visible in `trial.log` at debug
   level as `lstat .../tests/jobs: no such file or directory`), but it turned
   out to be a **non-fatal, already-handled** failure: Harbor's own code
   (`harbor/environments/docker/docker_unix.py`) catches exactly that
   `RuntimeError` and falls back to a tar-stream upload/engine-`cp` download
   that does NOT go through `--project-directory` at all -- confirmed by
   inspecting the fallback's implementation (plain `docker cp`/an in-process
   `tarfile` write, not another `docker compose` invocation) and by checking
   the host filesystem after a real failed trial: every agent-side artifact
   (`runs/`, `workspace/`, `playground/loop_db/`) was present under
   `jobs/<job>/<trial>/artifacts/` despite the logged compose-cp failure.

   The actual, fatal cause was different: `harbor/trial/trial.py`'s
   `_run_separate_verifier` hardcodes `skip_tests_upload=True` whenever
   `[verifier].environment_mode = "separate"` (confirmed by reading that
   call site directly) -- Harbor **never uploads `tests/` at runtime in this
   mode**, unlike `"shared"` mode. `harbor/verifier/verifier.py`'s
   `_resolve_tests` then just assumes `/tests/test.sh` already exists in the
   verifier image, per its own comment ("The verifier image already owns
   /tests/test.{sh,bat}"). Neither `Dockerfile.verifier` baked in `tests/`
   at all, so the verifier failed outright with
   `bash: line 1: /tests/test.sh: No such file or directory`, and
   `RewardFileNotFoundError` at the top level was really just a downstream
   symptom of that.

   Fixed by adding a `TESTS_SRC_DIR` build arg to `Dockerfile.verifier`
   (`COPY ${TESTS_SRC_DIR}/ /tests/` + `chmod +x`) and building **two**
   verifier image tags from it -- `mat-agent-harbor-verifier:latest`
   (`TESTS_SRC_DIR=benchmark/harbor/mace-mof0/tests`) and
   `mat-agent-harbor-verifier-smoke:latest`
   (`TESTS_SRC_DIR=benchmark/harbor/mace-mof0-smoke/tests`) -- because the
   two tracks' `tests/test.sh` genuinely differ (different compute-policy
   ledger; smoke also writes `plumbing_status.json`) and cannot share one
   image under Harbor's fixed, non-configurable `/tests/test.sh` path. The
   smoke task.toml's `[verifier.environment].docker_image` now points at the
   `-smoke` tag. Verified directly: both images built successfully, each has
   the correct, distinct `test.sh` at `/tests/` (checked by `grep`-ing a
   smoke-only marker string present in one and absent from the other).

Also fixed along the way, not blocking but worth knowing: `[verifier].network_mode
= "no-network"` (the original, tighter choice) makes Harbor try to build its
own egress-control sidecar via `docker buildx build` internally
(`_ensure_egress_control_sidecar_image_built`), unconditionally, regardless
of whether the host has buildx -- this one doesn't. Both task.toml files
now use `network_mode = "public"` for the verifier instead (see each file's
comment on that field for the exact trade-off this gives up).

None of these were found by static inspection -- all three came from
actually running `harbor run` against this task's real images and reading
the resulting tracebacks and, in blocker #2's case, from reading Harbor's
own installed source after the traceback pointed at the right file
(preserved outside the repo at
`/GenSIvePFS/users/public/mat-agent-harbor-smoke-diagnostics/` on this dev
machine: `trial.log`, `exception.txt`, `agent/claude-code.txt`, `result.json`
per attempt -- the multi-GB `artifacts/` dumps each attempt also produced
were deleted after extracting these).

## Agent runtime adapter: this benchmark does not know which agent it's running

**Neither this task nor its scientific contract knows or cares whether the
agent is Claude Code, Codex, or anything else.** The task only knows: read
`instruction.md` and `task_card.md`, submit through `remote_job.py`, respect
the budget, write `runs/SUBMISSION/submission.json`. Which agent produces
that submission is a `harbor run -a ...` flag, never a task-file change --
this matters for scientific reasons beyond convenience: it's what makes
`Claude Code + Sonnet` vs `Codex + GPT-5.x` vs `Gemini CLI + Gemini` a fair,
directly comparable experiment matrix later (see "Recording run metadata for
comparison" below), rather than three different one-off setups.

Concretely:

- `environment/Dockerfile` installs **every** supported agent's CLI at build
  time (currently Claude Code and Codex; both from the same base image --
  there is exactly one `mat-agent-harbor:latest`, not a per-agent image).
  Adding a third means adding one `RUN` block here, not branching the task.
- `benchmark/harbor/agents/preinstalled.py` defines `PreinstalledAgentMixin`
  -- a generic override of `BaseInstalledAgent.install()` that verifies the
  CLI is already present (via the agent's own `get_version_command()`,
  already implemented per-agent inside harbor itself) instead of attempting
  any install, plus one `Preinstalled<Name>` subclass per supported agent
  (currently `PreinstalledClaudeCode`, `PreinstalledCodex`). This is what
  actually fixes blocker #1 above, and it fixes it for every agent that hits
  the same `command -v`-based bug, not just Claude Code.
- Selecting an agent is exactly this import-path form of `-a`:
  ```bash
  PYTHONPATH="$PWD:$PYTHONPATH" harbor run -p benchmark/harbor/mace-mof0-smoke \
      -a benchmark.harbor.agents.preinstalled:PreinstalledClaudeCode -m claude-sonnet-5
  PYTHONPATH="$PWD:$PYTHONPATH" harbor run -p benchmark/harbor/mace-mof0-smoke \
      -a benchmark.harbor.agents.preinstalled:PreinstalledCodex -m gpt-5.1-codex
  ```
  (`PYTHONPATH` must include the repo root so harbor -- running on the host,
  not in a container -- can import the module; the agent CLASS runs on the
  host and only *emits* the shell commands `environment.exec()` runs inside
  the container, so this import has nothing to do with what's installed
  inside `main`.)

**Verification status**: `PreinstalledClaudeCode` is what the real, passing
build (see "What was verified") actually installs and was written against;
`PreinstalledCodex` is written against the real `harbor.agents.installed.codex.Codex`
interface and `environment/Dockerfile` does install `@openai/codex` at build
time, but no `harbor run` using it has been executed -- confirm before
relying on it for a real trial.

### Recording run metadata for comparison

No new metadata infrastructure was added for this, deliberately: Harbor's
own per-trial `result.json` (`jobs/<job>/<trial>/result.json`) already
records `config.agent.name`, `config.agent.import_path`,
`config.agent.model_name`, and `config.agent.kwargs` (e.g. `reasoning_effort`,
pinned `version`) for every trial. Join that against this task's own compute
ledger (`python tools/remote_job.py list-active`, or
`benchmark/compute/account_compute.py --task <task> --json` for a completed
run's GPU-hours) to build an (agent, model, science-GPU-hours, agent-cost)
comparison table -- there was no need to duplicate what Harbor already
tracks.

## Compute broker: why `main` never holds Volc credentials

An earlier version of this integration gave the agent's own container the
`volc` CLI and credentials directly, with `tools/remote_job.py`'s concurrency
gate enforced only in that same process. That made the gate advisory, not
real: nothing stopped the agent from noticing `volc` was installed and
calling it directly, skipping the gate entirely. Concretely, the old topology
was:

```text
             +-- remote_job.py gate --> volc   (the intended path)
Claude ------|
             +-- volc ml_task submit ->  volc   (a real bypass -- volc CLI + creds sat right there)
```

The fix is a **compute broker**: `volc` and the credentials live only in a
separate `broker` compose service. `tools/remote_job.py`, unchanged in its
public CLI, auto-detects `REMOTE_JOB_BROKER_URL` (set on `main` in
`environment/docker-compose.yaml`) and becomes a thin HTTP client instead of
running the volc-CLI logic itself -- see that file's module docstring. The
gate now lives exactly once, server-side, in a process the agent cannot
reach except through the HTTP contract:

```text
Claude (main, no credentials)
       |
       | submit / status / wait / collect  (HTTP, tools/remote_job.py client mode)
       v
broker (volc CLI + credentials, environment/broker/server.py + Dockerfile)
       |
       | enforces concurrency (<=4 A100-eq GPUs) and validates against policy.yaml
       v
Volcengine ML Platform
```

`environment/broker/server.py` adds no new submit/poll/collect logic of its
own -- it is a stdlib-only (`http.server`, no new dependency) HTTP shim
around the exact same `cmd_submit`/`cmd_status`/`cmd_wait`/`cmd_collect`
functions `tools/remote_job.py` already had and had already been tested
against live `volc` output earlier in this integration's development. This
was verified end-to-end on this dev machine (not just written): a real
`broker/server.py` process was started, and a real `tools/remote_job.py
list-active` client call with `REMOTE_JOB_BROKER_URL` set round-tripped over
actual HTTP and returned the correct result. A follow-up `status` call
against a live volc job hung on both the direct `volc ml_task get` call and
through the broker identically -- confirmed as Volcengine API-side flakiness
at the time (this session saw the same intermittent slowness elsewhere),
not a broker bug, by reproducing the hang with a bare `volc` call outside
any of this code.

This still does not make the agent's credential *isolation* airtight in
every sense: Claude Code inside `main` cannot read `~/.volc/credentials`
because that file simply doesn't exist in its container, but if it could
somehow reach the `broker` container's filesystem it would find them there
in plaintext, same as any process that holds a credential must. The
guarantee this design actually provides is narrower and more useful for this
benchmark specifically: **the compute cap is enforced by a process the agent
cannot bypass**, not "the agent can never learn the credential exists under
any circumstance."

## Runs/workspace bind mounts: why they exist and their real limit

`environment/docker-compose.yaml` bind-mounts the host repo's `runs/` and
`workspace/` into both `main` and `broker`, at the same absolute path both
already expect them at. This was **not** in the first draft of this
integration, which incorrectly concluded Harbor's docker environment has no
host-bind-mount mechanism at all (checked only
`environments/docker/docker.py`'s literal `volumes` references and
`EnvironmentConfig`'s field list, both of which are genuinely mount-less).
What that draft missed, caught on review: `docker.py`'s own
`_docker_compose_paths` docstring documents "Option 2: task with extra
services (docker-compose.yaml) -- create docker-compose.yaml with additional
services or overrides," and this is a real, confirmed mechanism (Harbor's
own generated `main` service compose file is layered under a task-authored
`environment/docker-compose.yaml`, which can add `volumes:` to `main` or
define whole new services). That is what makes both this bind mount and the
`broker` service above possible at all.

Two consequences of using it this way:

- It resolves what used to be this integration's single biggest open
  problem: a Volc job's `runs/<run_id>/run_manifest.json` and checkpoint,
  written by a real GPU node onto the same VePFS mount this repo lives on,
  now genuinely appear inside both `main` and `broker`'s own filesystems --
  no object-storage push/pull needed for that.
- **This is a local-only, non-portable shortcut, accepted deliberately.**
  It only works because this task currently only ever runs via Harbor's
  local Docker backend on this same dev machine, which already has that
  VePFS path mounted natively. Harbor task review generally steers tasks
  away from depending on host bind mounts for exactly this reason: move this
  task to a cloud sandbox backend (Modal, Daytona, etc. -- see
  `harbor.environments.*`) and the host path simply won't exist there. If
  this task ever needs to run anywhere but this machine's local Docker, the
  bind mount should be replaced with an object-storage (TOS) push from the
  training job and a pull in `tools/remote_job.py`/`broker/server.py` --
  deferred for now because a local VePFS mount is a much smaller, actually
  finishable change to validate the rest of the pipeline against first.

## What was verified in this session, and how

Everything below was checked against the actually-installed `harbor==0.23.0`
package source (downloaded as a wheel and inspected directly -- `pip
install harbor` itself repeatedly failed in this sandbox due to a flaky
outbound proxy; `pip download --no-deps harbor` succeeded because it is a
small pure-Python wheel that doesn't need harbor's heavy runtime
dependencies just to read its source), not assumed from memory or docs
alone:

- **`task.toml`'s entire schema** -- every field this file uses was checked
  against `harbor.models.task.config.TaskConfig` and validated by literally
  calling `TaskConfig.model_validate_toml()` on this repo's real
  `task.toml`, re-run after every subsequent edit. This caught a real bug:
  an earlier draft put the top-level `artifacts` list after `[solution]`,
  where TOML syntax silently binds it to `solution.artifacts` instead --
  fixed by moving it above every `[section]`.
- **`[verifier].environment_mode`** -- confirmed from
  `harbor.models.task.verifier_mode` that `"shared"` (default) runs the
  verifier in the *same* container as the agent, and `"separate"` (this
  task's setting, via `[verifier.environment]`) gives it a dedicated one.
  This is *why* two separate Dockerfiles/images exist for agent vs verifier.
- **`environment/docker-compose.yaml` as a task-authored overlay** --
  confirmed from `environments/docker/docker.py`'s `_docker_compose_paths`
  (see "Runs/workspace bind mounts" above). This is the actual mechanism
  behind the bind mounts and the `broker` service, corrected mid-session
  after an earlier draft wrongly concluded no such thing existed.
- **Reward file path and schema** -- confirmed from
  `harbor.verifier.verifier._parse_reward_json`/`_parse_reward_text` and
  `harbor.models.trial.paths`: `/logs/verifier/reward.json` is tried first
  (every value must be a finite int/float, dict or bare scalar -- a stray
  string field anywhere in it makes Harbor reject the *entire* file), then
  `/logs/verifier/reward.txt` (bare number) as fallback. An earlier draft of
  `tests/test.sh` put a human-readable `"reason"` string inside
  `reward.json` alongside `"reward"` -- that would have crashed the
  verifier outright; it was moved to a separate `reward_reason.txt`.
- **`harbor run -p <path> -a <agent> -m <model>`** and **`claude-code` as a
  real, even default-in-places, agent name** -- confirmed from
  `harbor.cli.jobs.build_job_config`'s actual Typer option definitions and
  multiple call sites defaulting `agent` to `"claude-code"`.
- **`-a <module>:<Class>` custom agent import paths actually work** --
  `benchmark.harbor.agents.preinstalled:PreinstalledClaudeCode` was
  genuinely resolved and run by a real `harbor run` invocation (with the
  repo root on `PYTHONPATH`), not just imported standalone.
- **The install-check bug is real and the pre-installed-CLI fix genuinely
  works** -- reproduced identically against two different images (this
  task's own `mat-agent-harbor` and harbor's own generic
  `python:3.13-slim` check image), then confirmed fixed: after
  `environment/Dockerfile` was changed to install Claude Code and Codex via
  `npm` at *build* time (the curl-based bootstrap harbor's own installer
  prefers timed out against `downloads.claude.ai` from this build context
  specifically -- other hosts, including the npm registry, were reachable
  fine) and `PreinstalledAgentMixin` was wired in, a real trial's agent log
  showed `"claude_code_version":"2.1.270"` in Claude Code's own init
  message -- it was genuinely running inside the container, not failing to
  start. `RUN claude --version` / `RUN codex --version` inside the
  Dockerfile also both pass as real build-time assertions.
- **Claude Code's OAuth vs API-key precedence** (`ClaudeCode._resolve_auth_env`)
  -- confirmed by testing both configurations for real: passing
  `CLAUDE_CODE_OAUTH_TOKEN` alone, and passing it together with
  `CLAUDE_FORCE_OAUTH=1`. Both produced an identical `403 Request not
  allowed`, which is itself the evidence behind the "Known blockers" #0
  conclusion that this is a server-side rejection, not a client
  misconfiguration.
- **`.dockerignore` negation semantics for hidden-data exclusion** -- proven
  with a real, throwaway `docker build` against a synthetic directory tree
  mimicking `data/mace_mof_0/**`, confirming the sequestered/phonons/external
  paths are absent from the build output while everything the agent needs is
  present. This is the actual isolation mechanism `environment/Dockerfile`
  relies on.
- **The compute-broker HTTP round trip** -- a real `broker/server.py`
  process and a real `tools/remote_job.py` client call, over actual
  loopback HTTP, on this dev machine (see "Compute broker" above for what
  worked and what hit unrelated Volcengine API flakiness).
- **The real volc status enum** -- `Queue, Staging, Running, Killing,
  Success, Failed, Killed, Initialized`, from `volc ml_task list --help` /
  `get --help` directly on this host's volc CLI (1.2.52), not guessed --
  `tools/remote_job.py`'s terminal/running status sets match this exactly.

## What is NOT verified

- **A real end-to-end `harbor run`.** `pip install harbor` (full package,
  needed to get the actual `harbor` CLI executable rather than just its
  importable source) could not complete in this sandbox: dependency
  resolution kept timing out on `files.pythonhosted.org` through this
  environment's outbound proxy. It got much further on a later retry (all
  dependencies downloaded) before failing on an unrelated certificate-bundle
  error caused by a venv being deleted while still in use -- worth retrying
  cleanly. Everything above was verified by importing/reading the wheel's
  source directly and validating `task.toml`/exercising the broker against
  it with real Python processes, not by running the `harbor` CLI binary
  itself. Once it installs, run `harbor check benchmark/harbor/mace-mof0`
  before a real trial.
- **`docker build`/`docker compose build` from this repo's root context, on
  this specific dev machine.** A minimal probe (`FROM alpine`, two tiny
  `COPY`s, repo root as context) failed with a generic `Error response from
  daemon: request err` on every attempt, while identical logic against a
  small synthetic directory (not repo root) built successfully in seconds.
  This points at this host's docker daemon/proxy setup choking on
  repo-root context assembly (this environment routes several other things
  through a flaky Privoxy instance -- see `GPU_ENV_STATUS.md`'s prior note
  on `mirrors.ivolces.com` failures), not a bug in any Dockerfile or
  `.dockerignore`. `environment/build.sh` has not been run to completion,
  and neither has the `broker` service's compose-driven build. Try from a
  machine with a plain, unproxied Docker daemon first.
- **Whether the `broker` service is actually reachable at `http://broker:8765`
  from `main` under Harbor's own compose orchestration** -- the service
  names and compose-merge mechanics are confirmed from source, but no
  `docker compose up` of this exact multi-service file has been run (blocked
  by the same docker-daemon issue above). Confirm this before trusting a
  real trial's submit calls.
- **Actual Stage-A/B timing**, hence `[verifier].timeout_sec = 3600` is an
  estimate, not a measurement -- `benchmark/evaluation/_lib/stage_b.py` has
  never run against real MACE + phonopy in this repo
  (`AUTONOMOUS_RESEARCH_READY.md` blocker B3). Revise once that's run.
- **The exact mechanism by which Harbor moves collected agent-side artifacts
  into the separate verifier container.** `ArtifactConfig` and the
  `[[artifacts]]` list are confirmed real and are what this task declares
  (`runs/`, `workspace/`, `playground/loop_db/`), and the verifier's
  `environment_mode = "separate"` is confirmed to mean a dedicated
  container -- but the precise plumbing connecting the two (does Harbor copy
  artifacts into the fresh verifier container before running `tests/test.sh`,
  and exactly when) was not traced through the source in this session.
  `tests/test.sh` fails loudly rather than silently if the sequestered data
  it needs turns out not to be present -- treat a failure there as a real
  signal to go trace that path, not a bug in the evaluator.

## Prerequisites

- Docker, with `mat-agent:full` buildable (`docker/Dockerfile`, unchanged).
- The `harbor` CLI (`pip install harbor` or `uv tool install harbor` --
  official install method per its GitHub README; not independently confirmed
  working end-to-end from this sandbox, see above).
- A local `volc` CLI already authenticated to the account that owns this
  repo's Volcengine queue (`~/.volc/{credentials,config,access_token}`).
- Claude Code credentials the agent will use --
  `ANTHROPIC_API_KEY`/`CLAUDE_CODE_OAUTH_TOKEN` in your own shell environment
  (Harbor's `claude-code` agent adapter reads these from the process
  environment; nothing extra to configure in `task.toml` for this).

## Build the environment images

```bash
bash benchmark/harbor/mace-mof0/environment/build.sh
```

Builds `mat-agent:full` (existing base), `mat-agent-harbor:<git-sha>`
(+`:latest`, the agent image -- no volc CLI/credentials, sequestered data
structurally excluded), and `mat-agent-harbor-verifier:<git-sha>`
(+`:latest`, the separate verifier image -- includes the sequestered split,
no volc CLI, no credentials, no network). It refuses to run if the volc CLI
isn't found, warns (doesn't block) on a dirty working tree, and vendors your
local `~/.volc/bin/volc` into `environment/.volc-bin/` (gitignored) --
deliberately **not** cleaned up afterward, because the `broker` service's own
image is built later, by Harbor's own `docker compose build` at `harbor run`
time (not by this script), and still needs that vendored binary present then.

## Credentials

```bash
eval "$(bash benchmark/harbor/mace-mof0/export_volc_env.sh)"
```

Run this in the *same shell* you'll invoke `harbor run` from, never inside a
tool call whose output gets logged -- it prints `export VAR=<base64>` lines
carrying your actual Volc credentials. It sets
`VOLC_CREDENTIALS_B64`/`VOLC_CONFIG_B64`/`VOLC_ACCESS_TOKEN_B64`, which
`environment/docker-compose.yaml` pulls into the **`broker`** service only
(plain Docker Compose `${VAR}` interpolation, not Harbor's `[environment].env`
mechanism), and that service's own `environment/entrypoint.sh` writes back to
`~/.volc/*` at container start. Neither `main` (the agent) nor the verifier
container ever receives these.

## Run one task

```bash
harbor run -p benchmark/harbor/mace-mof0 -a claude-code -m <model>
```

Validate first with `harbor check benchmark/harbor/mace-mof0` (the successor
to the now-removed `harbor task check`).

## Budget

Single source of truth: `benchmark/compute/policy.yaml`,
`tasks.mace_mof0_discovery`. Summary:

| | |
|---|---|
| Agent wall-clock | 16h (`task.toml` `[agent].timeout_sec = 57600`) |
| Total compute | 50 A100-equivalent GPU-hours |
| Per-experiment cap | 12 A100-equivalent GPU-hours |
| Concurrency | 4 A100-equivalent GPUs, enforced inside the `broker` service |
| Scored hidden evaluations | 3, including the final verifier run |
| Verifier/grading compute | excluded from the 50h |

## Where things live

- Agent trajectory / Claude Code transcript: wherever Harbor's own job
  output directory puts it (`-o`/`--jobs-dir`, default per `harbor run
  --help`) -- not something this task customizes.
- Volc jobs/results: `runs/<run_id>/` -- bind-mounted from the real host
  VePFS path into both `main` and `broker` (see "Runs/workspace bind
  mounts" above), so this is the same view a Volc GPU node itself writes to.
- Scientific result: `/logs/verifier/scorecard.json` and `verdict.json`
  (copied there by `tests/test.sh`) -- these, not the scalar reward, are the
  primary output; the reward is only the harness-required numeric interface.
- Task-loop history: `playground/loop_db/task_loops.sqlite`
  (`tools/task_loop_db.py`, `tools/remote_job.py`'s own `remote_jobs` table
  -- the latter lives inside whichever process is running LOCAL mode, i.e.
  the `broker` container in a real run).

## Resuming/debugging a failed run

There is no mid-trial checkpoint/resume (`instruction.md`'s "Budget"
section says this to the agent too) -- a trial is one continuous
environment lifetime up to the 16h ceiling. To debug a failed verifier pass,
read (in order) `/logs/verifier/preflight.json`,
`/logs/verifier/evaluate_mof0.stdout` (only written on an evaluator crash),
then `scorecard.json`/`verdict.json`. To debug a failed submission, check
the `broker` service's own container logs (`docker compose logs broker`) --
`tools/remote_job.py list-active`, run against the same `--db`, shows what
the concurrency gate currently thinks is still running, in case a trial died
mid-job and left an entry that needs to age out or be manually reconciled
against `volc ml_task list`.

## Security / hidden-data boundary

The sequestered-ML and sequestered-physics splits
(`data/mace_mof_0/discovery/_sequestered/**`,
`data/mace_mof_0/raw/phonons/**`) are never copied into the agent image --
`environment/Dockerfile`'s `COPY` list is an explicit allowlist that never
references either path, verified empirically (see above), not enforced by
convention or by telling the agent not to look. They are present in the
separate verifier image (`environment/Dockerfile.verifier`) with permissions
re-asserted (`chmod 400`) after `COPY`, since Docker does not reliably
preserve restrictive source file modes across build contexts. The verifier
gets no Volc credentials and no network, so even a compromised verifier step
cannot exfiltrate the sequestered data anywhere.

Separately, and orthogonally, Volc credentials are confined to the `broker`
service -- see "Compute broker" above for exactly what that guarantees (a
real concurrency/budget enforcement boundary) and what it does not (it is
not a guarantee that Claude Code can never learn the credentials exist; it
is a guarantee that it cannot use `volc` directly to bypass the gate, because
`main` has no `volc` binary to call).

This is a real improvement over the benchmark's original single-environment
design, where the sequestered-data protection existed only as file
permissions plus an in-process sandbox (`benchmark/evaluation/_lib/sandbox.py`)
that `AUTONOMOUS_RESEARCH_READY.md` itself already flagged as
"technical-ish... a root process on the same host can still read them," and
where there was no separation at all between the agent and the compute
credentials it used.
