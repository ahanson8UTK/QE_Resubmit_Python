from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from cw2017.utils.packages import (
    PackageInstallationError,
    ensure_packages_installed,
)


def test_ensure_packages_installed_noop_when_present(monkeypatch):
    def fake_find_spec(name: str):
        return object()

    def fail_check_call(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("pip should not be invoked when package is available")

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(subprocess, "check_call", fail_check_call)

    ensure_packages_installed("already_there")


def test_ensure_packages_installed_installs_missing(monkeypatch):
    installed = {"needs_install": False}

    def fake_find_spec(name: str):
        if installed["needs_install"]:
            return object()
        return None

    def fake_check_call(cmd, **kwargs):
        assert cmd[:4] == [sys.executable, "-m", "pip", "install"]
        assert cmd[4] == "needs_install"
        installed["needs_install"] = True

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    ensure_packages_installed("needs_install")
    assert installed["needs_install"]


def test_ensure_packages_installed_raises_on_failure(monkeypatch):
    def fake_find_spec(name: str):
        return None

    def fake_check_call(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    with pytest.raises(PackageInstallationError):
        ensure_packages_installed("never_installs")
