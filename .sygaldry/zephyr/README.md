# Zephyr Container Infra Portable Kit

This directory is the vendorable runtime kit copied into downstream repos as a hidden Zephyr runtime directory.

Canonical user manual:

- `docs/ZEPHYR_VENDORING_GUIDE.md`

Canonical release and rollout doc:

- `docs/ZEPHYR_CONTAINER_ADAPT_RELEASE_LAUNCH.md`

## What This Kit Contains

- `bin/repoctl` and `bin/jobctl` launchers
- `lib/infra_config.sh` config and image-mode resolver
- `container/` runtime launcher, entrypoints, and verifiers
- `Dockerfile.zephyr` and `build_repo_image.sh` for repo-derived images
- `infra.yaml` defaults and runtime policy

## Core Policy

- Heavy dependencies come from digest-pinned Spack snapshot images.
- Consumer repos do not run Spack rebuild/install.
- uv installs app dependencies on top without overriding Spack-owned packages.

## Modes

`infra.yaml` supports:

- `image_mode: standard`
- `image_mode: auto`
- `image_mode: derived`

Use `repoctl config show` in vendored repos to view `effective_image` resolution.
