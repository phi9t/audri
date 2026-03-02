#!/usr/bin/env python3
"""
parse-env-yaml.py — YAML -> shell/JSON bridge for zephyr MLSys envs.

Reads a zephyr-mlsys/v1 Environment YAML and emits either shell-sourceable
variables or JSON (for safe consumption without eval).

Usage:
  python3 parse-env-yaml.py <env.yaml>                  # shell output (default)
  python3 parse-env-yaml.py --output shell <env.yaml>   # shell output
  python3 parse-env-yaml.py --output json <env.yaml>    # JSON output (safe)

Requires PyYAML (available in Spack view).
"""

import json
import os
import sys

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required but not installed.\n"
        "HINT:  Install it for your Python runtime, for example:\n"
        "  python3 -m pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_yaml(path):
    """Parse YAML file using PyYAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def shell_quote(s):
    """Quote a string for shell."""
    if not s:
        return '""'
    if "\n" in str(s):
        escaped = str(s).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        return f"$'{escaped}'"
    return f'"{s}"'


def extract_data(data):
    """Extract structured data from parsed YAML into a flat dict."""
    meta = data.get("metadata", {}) or {}
    spec = data.get("spec", {}) or {}
    validation = spec.get("validation", data.get("validation", {})) or {}

    result = {
        "name": meta.get("name", ""),
        "description": meta.get("description", ""),
        "image": spec.get("image", "${SYGALDRY_SNAPSHOT_IMAGE:-sygaldry/zephyr:spack}"),
        "venvs": [],
        "validation": {
            "spack_provenance": [],
            "uv_provenance": [],
            "no_nvidia_pip": False,
            "gpu_functional": [],
            "hard_fail_on": [],
            "soft_fail_on": [],
        },
    }

    # Venvs
    venvs = spec.get("venvs", []) or []
    for i, venv in enumerate(venvs):
        if not isinstance(venv, dict):
            continue
        packages = venv.get("packages", []) or []
        if isinstance(packages, str):
            packages = [packages]
        overrides = venv.get("overrides", []) or []
        if isinstance(overrides, str):
            overrides = [overrides]
        result["venvs"].append(
            {
                "name": venv.get("name", f"venv-{i}"),
                "packages": [str(p) for p in packages],
                "overrides": [str(o) for o in overrides],
            }
        )

    # Validation
    prov = validation.get("provenance", {}) or {}
    spack_prov = prov.get("spack", []) or []
    if isinstance(spack_prov, str):
        spack_prov = [spack_prov]
    result["validation"]["spack_provenance"] = [str(p) for p in spack_prov]

    uv_prov = prov.get("uv", []) or []
    if isinstance(uv_prov, str):
        uv_prov = [uv_prov]
    result["validation"]["uv_provenance"] = [str(p) for p in uv_prov]

    no_nvidia = validation.get("no_nvidia_pip", False)
    if isinstance(no_nvidia, str):
        no_nvidia = no_nvidia.lower() == "true"
    result["validation"]["no_nvidia_pip"] = bool(no_nvidia)

    gpu_scripts = validation.get("gpu_functional", []) or []
    for i, script in enumerate(gpu_scripts):
        if not isinstance(script, dict):
            continue
        patterns = script.get("hard_fail_patterns", []) or []
        if isinstance(patterns, str):
            patterns = [patterns]
        result["validation"]["gpu_functional"].append(
            {
                "name": script.get("name", f"gpu-test-{i}"),
                "script": script.get("script", ""),
                "hard_fail_patterns": [str(p) for p in patterns],
            }
        )

    hard_fail = validation.get("hard_fail_on", []) or []
    if isinstance(hard_fail, str):
        hard_fail = [hard_fail]
    result["validation"]["hard_fail_on"] = [str(h) for h in hard_fail]

    soft_fail = validation.get("soft_fail_on", []) or []
    if isinstance(soft_fail, str):
        soft_fail = [soft_fail]
    result["validation"]["soft_fail_on"] = [str(s) for s in soft_fail]

    return result


def emit_json(data):
    """Emit JSON output (safe for consumption without eval)."""
    print(json.dumps(data, indent=2))


def emit_shell(data):
    """Emit shell-sourceable variables."""
    print(f'ENV_NAME={shell_quote(data["name"])}')
    print(f'ENV_DESCRIPTION={shell_quote(data["description"])}')
    print(f'ENV_IMAGE={shell_quote(data["image"])}')

    venvs = data["venvs"]
    print(f"VENV_COUNT={len(venvs)}")
    for i, venv in enumerate(venvs):
        print(f'VENV_{i}_NAME={shell_quote(venv["name"])}')
        print(f'VENV_{i}_PACKAGES={shell_quote(" ".join(venv["packages"]))}')
        print(f'VENV_{i}_OVERRIDES={shell_quote(" ".join(venv["overrides"]))}')

    v = data["validation"]
    print(f'VALIDATION_SPACK_PROVENANCE={shell_quote(" ".join(v["spack_provenance"]))}')
    print(f'VALIDATION_UV_PROVENANCE={shell_quote(" ".join(v["uv_provenance"]))}')
    print(f'VALIDATION_NO_NVIDIA_PIP={"true" if v["no_nvidia_pip"] else "false"}')

    gpu = v["gpu_functional"]
    print(f"VALIDATION_GPU_SCRIPT_COUNT={len(gpu)}")
    for i, script in enumerate(gpu):
        print(f'VALIDATION_GPU_SCRIPT_{i}_NAME={shell_quote(script["name"])}')
        print(f'VALIDATION_GPU_SCRIPT_{i}={shell_quote(script["script"] or "")}')
        print(
            f'VALIDATION_GPU_SCRIPT_{i}_HARD_FAIL_PATTERNS={shell_quote(" ".join(script["hard_fail_patterns"]))}'
        )

    print(f'VALIDATION_HARD_FAIL_ON={shell_quote(" ".join(v["hard_fail_on"]))}')
    print(f'VALIDATION_SOFT_FAIL_ON={shell_quote(" ".join(v["soft_fail_on"]))}')


def main():
    output_mode = "shell"
    args = sys.argv[1:]

    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_mode = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
        else:
            print("ERROR: --output requires a value (shell or json)", file=sys.stderr)
            sys.exit(2)

    if not args:
        print(
            "Usage: parse-env-yaml.py [--output shell|json] <env.yaml>", file=sys.stderr
        )
        sys.exit(2)

    path = args[0]
    if not os.path.isfile(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw = parse_yaml(path)

    api_ver = raw.get("apiVersion", "")
    if api_ver != "zephyr-mlsys/v1":
        print(
            f"WARNING: Expected apiVersion zephyr-mlsys/v1, got '{api_ver}'",
            file=sys.stderr,
        )

    data = extract_data(raw)

    if output_mode == "json":
        emit_json(data)
    elif output_mode == "shell":
        emit_shell(data)
    else:
        print(
            f"ERROR: Unknown output mode: {output_mode} (expected shell or json)",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
