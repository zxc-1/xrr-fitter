"""Artifact manifest, archive metadata, and smoke validation contracts."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

COMMIT = "1" * 40
TREE = "2" * 40


def test_distribution_archive_identity_reads_dynamic_package_version(
    tmp_path: Path,
    load_tool_module,
) -> None:
    load_tool_module("verify_distribution")
    module = sys.modules["distribution_archive"]
    package = tmp_path / "src" / "xrr_fitter"
    package.mkdir(parents=True)
    (package / "version.py").write_text('__version__ = "0.2.2"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "xrr-fitter"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "xrr_fitter.version.__version__"}
""",
        encoding="utf-8",
    )

    assert module._project_identity(tmp_path) == ("xrr_fitter", "0.2.2")


def test_distribution_archive_identity_rejects_static_and_dynamic_versions(
    tmp_path: Path,
    load_tool_module,
) -> None:
    load_tool_module("verify_distribution")
    module = sys.modules["distribution_archive"]
    package = tmp_path / "src" / "xrr_fitter"
    package.mkdir(parents=True)
    (package / "version.py").write_text('__version__ = "0.2.2"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "xrr-fitter"
version = "1.0.0"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "xrr_fitter.version.__version__"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both statically and dynamically"):
        module._project_identity(tmp_path)


def test_distribution_archive_accepts_cli_only_entry_points_metadata(
    tmp_path: Path,
    load_tool_module,
) -> None:
    load_tool_module("verify_distribution")
    module = sys.modules["distribution_archive"]
    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "xrr-fitter"
version = "0.2.2"

[project.scripts]
xrr-fitter-cli = "xrr_fitter.cli.main:main"
""",
        encoding="utf-8",
    )

    assert "xrr_fitter-0.2.2.dist-info/entry_points.txt" in module._wheel_metadata(tmp_path)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _artifact_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "bundle" / "artifacts"
    directory.mkdir(parents=True)
    (directory / "xrr_fitter-0.2.0-py3-none-any.whl").write_bytes(b"wheel-bytes")
    (directory / "xrr_fitter-0.2.0.tar.gz").write_bytes(b"sdist-bytes")
    return directory


def _manifest(module, tmp_path: Path):
    directory = _artifact_dir(tmp_path)
    return directory, module.calculate_artifact_manifest(
        directory,
        head_commit=COMMIT,
        head_tree=TREE,
    )


def _value(manifest) -> dict[str, object]:
    return {
        "schema": manifest.schema,
        "status": manifest.status,
        "head_commit": manifest.head_commit,
        "head_tree": manifest.head_tree,
        "artifacts": [
            {
                "kind": record.kind,
                "path": record.path,
                "filename": record.filename,
                "size": record.size,
                "sha256": record.sha256,
            }
            for record in manifest.artifacts
        ],
    }


def test_calculator_hashes_exact_wheel_and_sdist_in_kind_order(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify_distribution")
    directory, manifest = _manifest(module, tmp_path)

    assert (manifest.schema, manifest.status) == (
        "xrr-r23-artifact-manifest-v1",
        "PASS",
    )
    assert (manifest.head_commit, manifest.head_tree) == (COMMIT, TREE)
    assert tuple(record.kind for record in manifest.artifacts) == ("sdist", "wheel")
    for record in manifest.artifacts:
        artifact = directory / record.filename
        assert record.path == f"artifacts/{record.filename}"
        assert record.size == artifact.stat().st_size
        assert record.sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ("missing", "extra", "wheel-extra", "wrong-dir"))
def test_calculator_rejects_missing_extra_or_relocated_artifacts(
    tmp_path: Path,
    load_tool_module,
    mutation: str,
) -> None:
    module = load_tool_module("verify_distribution")
    directory = _artifact_dir(tmp_path)
    if mutation == "missing":
        next(directory.glob("*.whl")).unlink()
    elif mutation == "extra":
        (directory / "notes.txt").write_text("unexpected", encoding="utf-8")
    elif mutation == "wheel-extra":
        (directory / "other-0.2.0-py3-none-any.whl").write_bytes(b"other")
    else:
        relocated = directory.with_name("renamed")
        directory.rename(relocated)
        directory = relocated

    with pytest.raises(ValueError, match="artifact|wheel|sdist|directory"):
        module.calculate_artifact_manifest(
            directory,
            head_commit=COMMIT,
            head_tree=TREE,
        )


def test_manifest_parser_round_trips_only_canonical_bytes(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify_distribution")
    _directory, manifest = _manifest(module, tmp_path)
    content = module.canonical_manifest_bytes(manifest)

    assert content == _canonical(_value(manifest))
    assert module.parse_artifact_manifest(content) == manifest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("status"),
        lambda value: value.update(extra=True),
        lambda value: value.update(schema="wrong"),
        lambda value: value.update(status="FAIL"),
        lambda value: value.update(head_commit="A" * 40),
        lambda value: value.update(head_tree="2" * 39),
        lambda value: value.update(artifacts=value["artifacts"][:1]),
        lambda value: value.update(artifacts=value["artifacts"] + [value["artifacts"][0]]),
        lambda value: value.update(artifacts=list(reversed(value["artifacts"]))),
        lambda value: value["artifacts"][0].pop("size"),
        lambda value: value["artifacts"][0].update(extra=True),
        lambda value: value["artifacts"][0].update(kind="archive"),
        lambda value: value["artifacts"][0].update(path="../archive.tar.gz"),
        lambda value: value["artifacts"][0].update(path="bundle/archive.tar.gz"),
        lambda value: value["artifacts"][0].update(filename="other.tar.gz"),
        lambda value: value["artifacts"][0].update(size=True),
        lambda value: value["artifacts"][0].update(size=0),
        lambda value: value["artifacts"][0].update(sha256="F" * 64),
    ],
    ids=(
        "missing-top",
        "extra-top",
        "schema",
        "status",
        "commit",
        "tree",
        "artifact-missing",
        "artifact-extra",
        "artifact-order",
        "record-missing",
        "record-extra",
        "kind",
        "traversal",
        "directory",
        "filename",
        "boolean-size",
        "zero-size",
        "sha",
    ),
)
def test_manifest_parser_rejects_field_path_order_and_value_drift(
    tmp_path: Path,
    load_tool_module,
    mutate,
) -> None:
    module = load_tool_module("verify_distribution")
    _directory, manifest = _manifest(module, tmp_path)
    value = _value(manifest)
    mutate(value)

    with pytest.raises(ValueError, match="manifest|artifact|commit|tree|field|path|size|SHA"):
        module.parse_artifact_manifest(_canonical(value))


@pytest.mark.parametrize(
    "content",
    (
        b'{"schema":"xrr-r23-artifact-manifest-v1","schema":"duplicate"}\n',
        b"{}",
        b"{}\r\n",
        b"{ }\n",
        b"[]\n",
        b"\xff",
    ),
)
def test_manifest_parser_rejects_duplicate_or_noncanonical_json(
    load_tool_module,
    content: bytes,
) -> None:
    module = load_tool_module("verify_distribution")

    with pytest.raises(ValueError, match="JSON|canonical|manifest|duplicate"):
        module.parse_artifact_manifest(content)


def test_manifest_validation_recomputes_artifacts_and_git_identity(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify_distribution")
    directory, manifest = _manifest(module, tmp_path)
    module.validate_artifact_manifest(
        manifest,
        directory,
        head_commit=COMMIT,
        head_tree=TREE,
    )

    next(directory.glob("*.whl")).write_bytes(b"drift")
    with pytest.raises(ValueError, match="drift"):
        module.validate_artifact_manifest(
            manifest,
            directory,
            head_commit=COMMIT,
            head_tree=TREE,
        )
    with pytest.raises(ValueError, match="drift"):
        module.validate_artifact_manifest(
            manifest,
            directory,
            head_commit="3" * 40,
            head_tree=TREE,
        )
    with pytest.raises(ValueError, match="drift"):
        module.validate_artifact_manifest(
            manifest,
            directory,
            head_commit=COMMIT,
            head_tree="3" * 40,
        )


def _wheel(path: Path, dependencies: tuple[str, ...]) -> None:
    metadata = "Metadata-Version: 2.4\nName: xrr-fitter\nVersion: 0.2.0\n"
    metadata += "".join(f"Requires-Dist: {value}\n" for value in dependencies)
    metadata += "\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xrr_fitter-0.2.0.dist-info/METADATA", metadata)


def test_wheel_requires_dist_matches_pyproject_exactly(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify_distribution")
    observed = (
        "numpy<3,>=2.0",
        "PySide6<7,>=6.8",
        'pytest<9,>=8.3; extra == "test"',
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        "name='xrr-fitter'\n"
        "version='0.2.0'\n"
        "dependencies=['numpy>=2.0,<3','PySide6>=6.8,<7']\n"
        "[project.optional-dependencies]\n"
        "test=['pytest>=8.3,<9']\n",
        encoding="utf-8",
    )
    wheel = tmp_path / "xrr_fitter-0.2.0-py3-none-any.whl"
    _wheel(wheel, observed)

    module.verify_wheel_dependencies(wheel, pyproject)

    _wheel(wheel, observed[:-1])
    with pytest.raises(ValueError, match="Requires-Dist"):
        module.verify_wheel_dependencies(wheel, pyproject)


def test_reproducibility_compares_names_and_every_artifact_byte(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify_distribution")
    first = _artifact_dir(tmp_path / "first")
    second = _artifact_dir(tmp_path / "second")

    module.verify_reproducible_artifacts(first, second)

    next(second.glob("*.tar.gz")).write_bytes(b"drift")
    with pytest.raises(ValueError, match="reproducible"):
        module.verify_reproducible_artifacts(first, second)


def test_installed_smoke_uses_only_absolute_venv_commands_and_isolated_environment(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    environment = tmp_path / "smoke-venv"
    bin_dir = environment / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    entrypoint = bin_dir / "xrr-fitter"
    cli_entrypoint = bin_dir / "xrr-fitter-cli"
    python.write_bytes(b"python")
    entrypoint.write_bytes(b"entrypoint")
    cli_entrypoint.write_bytes(b"cli-entrypoint")
    monkeypatch.setenv("PYTHONPATH", "/caller/source")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(args, **kwargs):
        calls.append((tuple(str(value) for value in args), kwargs))

    module.smoke_installed(environment, runner=runner)

    resolved = environment.resolve()
    assert [args for args, _kwargs in calls] == [
        (str(resolved / "bin/python"), "-c", "import xrr_fitter.api"),
        (str(resolved / "bin/python"), "-m", "xrr_fitter", "--help"),
        (str(resolved / "bin/xrr-fitter"), "--help"),
        (str(resolved / "bin/xrr-fitter-cli"), "--help"),
    ]
    for args, kwargs in calls:
        assert Path(args[0]).is_absolute()
        assert kwargs["check"] is True
        assert kwargs["cwd"] == resolved
        child = kwargs["env"]
        assert child["PATH"] == ""
        assert "PYTHONPATH" not in child
        assert all(Path(child[name]).is_relative_to(resolved) for name in ("HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"))


def test_wheel_entry_points_must_exactly_match_declared_scripts(
    tmp_path: Path,
    load_tool_module,
) -> None:
    load_tool_module("verify_distribution")
    module = sys.modules["distribution_archive"]
    pyproject = tmp_path / "pyproject.toml"
    project_text = """[project]
name = "xrr-fitter"
version = "0.2.0"

[project.gui-scripts]
xrr-fitter = "xrr_fitter.__main__:main"

[project.scripts]
xrr-fitter-cli = "xrr_fitter.cli.main:main"
"""
    pyproject.write_text(project_text, encoding="utf-8")
    wheel = tmp_path / "xrr_fitter-0.2.0-py3-none-any.whl"
    root = "xrr_fitter-0.2.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("xrr_fitter/__init__.py", "")
        for name in ("METADATA", "RECORD", "WHEEL", "top_level.txt"):
            archive.writestr(f"{root}/{name}", "")
        archive.writestr(
            f"{root}/entry_points.txt",
            """[console_scripts]
xrr-fitter-cli = xrr_fitter.cli.main:main

[gui_scripts]
xrr-fitter = xrr_fitter.__main__:main
""",
        )

    inputs = (PurePosixPath("src/xrr_fitter/__init__.py"),)
    spec = {"wheel_content_policy": {"forbidden_roots": []}}
    module._verify_wheel_members(tmp_path, wheel, inputs, spec)

    pyproject.write_text(project_text.replace("cli.main", "cli.wrong"), encoding="utf-8")
    with pytest.raises(ValueError, match="entry point"):
        module._verify_wheel_members(tmp_path, wheel, inputs, spec)


def _write_sdist(path: Path, members: tuple[tuple[str, bytes, str], ...]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            elif kind == "directory":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.type = tarfile.SYMTYPE
                info.linkname = "pyproject.toml"
                archive.addfile(info)


def test_sdist_smoke_extracts_one_safe_source_root_before_building(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_tool_module("verify_distribution")
    module = __import__("distribution_smoke")
    sdist = tmp_path / "xrr_fitter-0.2.0.tar.gz"
    _write_sdist(
        sdist,
        (
            ("xrr_fitter-0.2.0", b"", "directory"),
            ("xrr_fitter-0.2.0/pyproject.toml", b"[build-system]\n", "file"),
        ),
    )
    environment = tmp_path / "venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").write_bytes(b"python")
    observed_sources: list[Path] = []

    monkeypatch.setattr(module, "_install_environment", lambda *_args: environment)
    monkeypatch.setattr(module, "_install_wheel", lambda *_args: None)
    monkeypatch.setattr(module, "smoke_installed", lambda *_args: None)
    monkeypatch.setattr(module, "verify_wheel_dependencies", lambda *_args: None)

    def run(args, **_kwargs):
        source = Path(args[-1])
        observed_sources.append(source)
        assert source.is_dir()
        assert (source / "pyproject.toml").read_bytes() == b"[build-system]\n"
        output = Path(args[args.index("--outdir") + 1])
        (output / "rebuilt.whl").write_bytes(b"wheel")

    monkeypatch.setattr(module.subprocess, "run", run)

    module.smoke_sdist(tmp_path, sdist, 123)

    assert len(observed_sources) == 1
    assert observed_sources[0].name == "xrr_fitter-0.2.0"
    assert observed_sources[0] != sdist


@pytest.mark.parametrize(
    "member",
    (
        ("../escape.txt", b"escape", "file"),
        ("/absolute.txt", b"escape", "file"),
        ("xrr_fitter-0.2.0\\escape.txt", b"escape", "file"),
        ("xrr_fitter-0.2.0/link", b"", "symlink"),
        ("xrr_fitter-0.2.0/pyproject.toml", b"duplicate", "file"),
        ("other-root/file.txt", b"other", "file"),
    ),
)
def test_sdist_extraction_rejects_unsafe_or_multiple_roots(
    tmp_path: Path,
    load_tool_module,
    member: tuple[str, bytes, str],
) -> None:
    load_tool_module("verify_distribution")
    module = __import__("distribution_smoke")
    sdist = tmp_path / "xrr_fitter-0.2.0.tar.gz"
    baseline = ("xrr_fitter-0.2.0/pyproject.toml", b"project", "file")
    _write_sdist(sdist, (baseline, member))
    destination = tmp_path / "source"
    destination.mkdir()

    with pytest.raises(ValueError, match="sdist|archive|root|member"):
        module.extract_sdist(sdist, destination)

    assert tuple(destination.iterdir()) == ()
