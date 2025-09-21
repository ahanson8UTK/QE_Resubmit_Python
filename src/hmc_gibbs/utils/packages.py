"""Helpers for ensuring optional runtime dependencies are available."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from collections.abc import Iterable, Mapping
from typing import Dict


class PackageInstallationError(RuntimeError):
    """Raised when an automatic dependency installation fails."""


PackageSpec = Dict[str, str]


def _normalise_packages(packages: Iterable[str] | Mapping[str, str] | str) -> PackageSpec:
    """Return a mapping of import names to pip install specifications."""

    if isinstance(packages, str):
        return {packages: packages}

    if isinstance(packages, Mapping):
        return dict(packages)

    normalised: Dict[str, str] = {}
    for name in packages:
        normalised[name] = name
    return normalised


def _module_available(module_name: str) -> bool:
    """Return ``True`` if ``module_name`` can be imported."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _install_package(spec: str) -> None:
    """Install ``spec`` using ``pip``."""

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise PackageInstallationError(f"Failed to install required package '{spec}'.") from exc


def ensure_packages_installed(packages: Iterable[str] | Mapping[str, str] | str) -> None:
    """Ensure that each requested package can be imported.

    Parameters
    ----------
    packages:
        Either a string naming a package, an iterable of strings, or a mapping from
        module import names to pip install specifications. When a package is
        missing it will be installed using ``pip``.
    """

    package_specs = _normalise_packages(packages)
    missing = [(name, spec) for name, spec in package_specs.items() if not _module_available(name)]

    for import_name, install_spec in missing:
        _install_package(install_spec)
        importlib.invalidate_caches()
        if not _module_available(import_name):
            raise PackageInstallationError(
                f"Package '{install_spec}' was installed but '{import_name}' could still not be imported."
            )


__all__ = ["ensure_packages_installed", "PackageInstallationError"]

