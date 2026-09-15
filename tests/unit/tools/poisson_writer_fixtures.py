"""Observe real file handles at publication boundaries; no platform emulation."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def writer_lifecycle(tool, monkeypatch):
    state = SimpleNamespace(stream=None, events=[], fail_at=None, error=OSError("injected writer failure"))
    real_temporary = tool.tempfile.NamedTemporaryFile
    real_fsync, real_link, real_unlink = tool.os.fsync, tool.os.link, Path.unlink

    def step(name, operation, *args, **kwargs):
        state.events.append(name)
        if state.fail_at == name:
            raise state.error
        return operation(*args, **kwargs)

    def temporary(*args, **kwargs):
        stream = real_temporary(*args, **kwargs)
        state.stream = stream
        monkeypatch.setattr(stream, "write", partial(step, "write", stream.write))
        monkeypatch.setattr(stream, "flush", partial(step, "flush", stream.flush))
        return stream

    def sync(descriptor):
        assert not state.stream.closed
        assert state.events == ["write", "flush"]
        return step("fsync", real_fsync, descriptor)

    def link(source, destination):
        assert state.stream.closed, "temporary stream must be closed before link"
        assert state.events == ["write", "flush", "fsync"]
        return step("link", real_link, source, destination)

    def unlink(path, *args, **kwargs):
        if state.stream is not None and path == Path(state.stream.name):
            assert state.stream.closed, "temporary stream must be closed before unlink"
            return step("unlink", real_unlink, path, *args, **kwargs)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tool.tempfile, "NamedTemporaryFile", temporary)
    monkeypatch.setattr(tool.os, "fsync", sync)
    monkeypatch.setattr(tool.os, "link", link)
    monkeypatch.setattr(Path, "unlink", unlink)
    return state
