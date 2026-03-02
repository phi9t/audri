#!/usr/bin/env python3
"""Verify that Spack-owned packages resolve from /opt/spack_store/view
and that no nvidia-* pip packages are installed."""

import importlib.metadata as md
import sys

SPACK_VIEW = "/opt/spack_store/view"

# Full list of pip package names managed by Spack.
# Must stay in sync with container/spack_owned_packages.conf.
SPACK_OWNED = [
    "torch",
    "torchvision",
    "torchaudio",
    "jax",
    "jaxlib",
    "triton",
    "numpy",
    "scipy",
    "scikit-learn",
    "numba",
    "llvmlite",
    "matplotlib",
    "pandas",
    "soundfile",
    "jupyterlab",
]


def is_spack(path: str) -> bool:
    return SPACK_VIEW in (path or "")


def check_spack_provenance() -> list[str]:
    """Verify all importable Spack-owned packages come from the Spack view."""
    errors = []
    for pkg_name in SPACK_OWNED:
        mod_name = pkg_name.replace("-", "_")
        try:
            mod = __import__(mod_name)
        except ImportError:
            # Package not installed at all — acceptable if the env doesn't
            # include it, but worth noting.
            print(f"  SKIP: {pkg_name} (not importable)")
            continue

        mod_file = getattr(mod, "__file__", "") or ""
        if is_spack(mod_file):
            print(f"  OK:   {pkg_name} -> {mod_file}")
        else:
            errors.append(f"{pkg_name} is NOT from Spack view: {mod_file}")
            print(f"  FAIL: {pkg_name} -> {mod_file}")
    return errors


def check_no_nvidia_pip() -> list[str]:
    """Verify no nvidia-* pip packages are installed."""
    errors = []
    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        if name.startswith("nvidia-"):
            loc = str(dist._path) if hasattr(dist, "_path") else "unknown"
            # Allow nvidia packages that are inside the Spack view
            if SPACK_VIEW in loc:
                print(f"  OK:   {name} (from Spack view)")
                continue
            errors.append(f"nvidia pip package installed: {name} at {loc}")
            print(f"  FAIL: {name} at {loc}")
    if not errors:
        print("  OK:   No nvidia-* pip packages outside Spack view")
    return errors


def main() -> int:
    print("=== UV-Spack Provenance Check ===")
    print()
    print(f"sys.executable:  {sys.executable}")
    print(f"sys.base_prefix: {sys.base_prefix}")
    print()

    # Check Spack base interpreter
    base_ok = is_spack(sys.base_prefix) or is_spack(
        getattr(sys, "_base_executable", "")
    )
    if base_ok:
        print("OK: Spack base interpreter detected")
    else:
        print("FAIL: Spack base interpreter not detected")
    print()

    print("--- Spack-owned package provenance ---")
    prov_errors = check_spack_provenance()
    print()

    print("--- NVIDIA pip package check ---")
    nvidia_errors = check_no_nvidia_pip()
    print()

    all_errors = []
    if not base_ok:
        all_errors.append("Spack base interpreter not detected")
    all_errors.extend(prov_errors)
    all_errors.extend(nvidia_errors)

    if all_errors:
        print(f"FAILED ({len(all_errors)} error(s)):")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print("OK: All UV-Spack provenance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
