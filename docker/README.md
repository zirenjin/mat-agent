# Docker

`docker/` contains the unified MatterTune-centered runtime.

```bash
docker build --build-arg MAT_AGENT_INSTALL_MODE=full -f docker/Dockerfile -t mat-agent:full .
```

Build modes:

- `smoke`: validate source layout without installing Python dependencies.
- `core`: install MatterTune core dependencies.
- `full`: install MatterTune extras for training, data, logging, and neighbor lists.

The default PyPI mirror is `https://mirrors.ivolces.com/pypi/simple/`, which is fastest on the current Beijing/Volcano remote host. Override `PIP_INDEX_URL` and `PIP_TRUSTED_HOST` if running from another network.

See [../docs/mattertune_bash_api.md](../docs/mattertune_bash_api.md) for the supported shell API contract.

Run the Bash-first lifecycle API with commands such as `docker run --rm mat-agent:full train.sh mace ... --dry-run`. Public lifecycle commands are `train.sh`, `inference.sh`, and `evaluate.sh`; `score.sh` is not the MatterTune lifecycle API. The image uses `CMD ["/bin/bash"]` rather than a fixed entrypoint so subcommands have normal command semantics.
