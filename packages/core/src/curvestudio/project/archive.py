"""Save and load ``.cstudio`` project archives.

A project is a ZIP holding ``manifest.json``, one CSV pair (``original.csv``,
``cleaned.csv``) plus ``operations.json`` per dataset, one JSON file per analysis and a
``plot_settings.json``. Loading treats the file as untrusted: entry names must be safe
relative POSIX paths, the declared uncompressed sizes and compression ratio are bounded,
every payload must match its SHA-256 in the manifest, and the cleaned table must be
exactly what replaying the operation log on the rebuilt original produces. Nothing is
written to disk and nothing is unpickled or evaluated.
"""

import hashlib
import json
import os
import platform
import re
import stat
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import IO, Any, BinaryIO, cast

from pydantic import ValidationError

from curvestudio.data import (
    DataError,
    Dataset,
    ExcludeRows,
    Operation,
    RenameColumn,
    TransformColumn,
)
from curvestudio.data.operations import TransformFunc
from curvestudio.fitting import FitResult
from curvestudio.project.csvio import ColumnSpec, column_specs, decode_frame, encode_frame
from curvestudio.project.errors import ProjectError, ProjectErrorCode
from curvestudio.project.model import (
    Analysis,
    Manifest,
    ManifestEntry,
    Project,
    ProjectDataset,
)

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
"""The versions this build reads; a file of either is read, and saving writes the latest."""
MANIFEST_NAME = "manifest.json"
PLOT_SETTINGS_NAME = "plot_settings.json"

MAX_ENTRIES = 2000
MAX_ENTRY_BYTES = 200 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
# Entries at most this large skip the compression-ratio check. A legitimate save can
# compress enormously: a dataset with a constant column of a few thousand rows is all
# repetitions, and deflate turns it into a handful of bytes. The ratio check exists to
# stop a small archive from expanding into gigabytes of RAM; the per-entry and total
# caps already bound the memory a small entry can claim, so the ratio only matters for
# large declared sizes.
MIN_RATIO_CHECK_BYTES = 16 * 1024 * 1024

_DATASET_FILE = re.compile(
    r"^datasets/(?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<kind>original\.csv|cleaned\.csv|operations\.json)$"
)
_ANALYSIS_FILE = re.compile(r"^analyses/(?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*)\.json$")
_TRANSFORMS = frozenset({"log", "log10", "sqrt", "reciprocal", "square", "affine"})

_APP_DISTRIBUTION = "curvestudio-core"
_DISTRIBUTIONS = {
    "python": None,  # filled from platform.python_version()
    "numpy": "numpy",
    "scipy": "scipy",
    "sympy": "sympy",
    "curvestudio": _APP_DISTRIBUTION,
}


def save_project(project: Project, target: Path | BinaryIO) -> None:
    """Write ``project`` to ``target`` (a path or a binary file object) as a ZIP.

    The archive is deterministic for a given project: entry timestamps are fixed, so
    saving an unchanged project twice produces identical bytes. A path target is
    written atomically: the previous file survives any failure untouched.
    """
    payloads = _payloads(project)
    _check_casefold_names(payloads)
    if len(payloads) + 1 > MAX_ENTRIES:
        raise ProjectError(
            ProjectErrorCode.UNSUPPORTED_DATA,
            f"the project needs {len(payloads) + 1:,} archive entries; the limit is "
            f"{MAX_ENTRIES:,}",
        )
    manifest = Manifest(
        app_version=_version_of(_APP_DISTRIBUTION),
        versions=_versions(),
        project_name=project.name,
        created_at=project.created_at,
        entries=tuple(
            ManifestEntry(path=name, sha256=_sha256(data)) for name, data in payloads.items()
        ),
    )
    document = manifest.model_dump(mode="json")
    entries = {MANIFEST_NAME: _json_bytes(document, "the manifest"), **payloads}
    if isinstance(target, (str, os.PathLike)):
        _write_path_atomically(Path(target), entries)
    else:
        try:
            _write_zip(target, entries)
        except OSError as exc:
            raise ProjectError(
                ProjectErrorCode.SAVE_FAILED, f"cannot save the project: {exc}"
            ) from exc


def _check_casefold_names(payloads: Mapping[str, bytes]) -> None:
    """Refuse names that collide on a case-insensitive filesystem."""
    seen: dict[str, str] = {}
    for name in payloads:
        folded = name.casefold()
        if folded in seen:
            raise ProjectError(
                ProjectErrorCode.UNSUPPORTED_DATA,
                f"{name!r} and {seen[folded]!r} collide case-insensitively",
            )
        seen[folded] = name


def load_project(source: Path | BinaryIO) -> Project:
    """Read a ``.cstudio`` file (a path or a binary file object).

    Every failure is a :class:`ProjectError`; the original exception, if any, is chained
    as its cause. Datasets are rebuilt through the public ``Dataset`` API and checked
    against the stored fingerprint and the stored cleaned table.
    """
    archive = _open_archive(source)
    try:
        return _load(archive)
    except ProjectError:
        raise
    except zipfile.BadZipFile as exc:
        raise ProjectError(
            ProjectErrorCode.CORRUPT, f"the project archive is damaged: {exc}"
        ) from exc
    except Exception as exc:  # boundary: untrusted input never leaks another type
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"the project cannot be read: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        archive.close()


# --- saving ----------------------------------------------------------------------------


def _payloads(project: Project) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for entry in project.datasets:
        frame = entry.dataset.original
        specs = column_specs(frame)
        files = (
            (f"datasets/{entry.id}/original.csv", encode_frame(frame)),
            (f"datasets/{entry.id}/cleaned.csv", encode_frame(entry.dataset.table())),
            (f"datasets/{entry.id}/operations.json", _dataset_bytes(entry, specs)),
        )
        for name, data in files:
            _check_payload(name, data)
            payloads[name] = data
    for analysis in project.analyses:
        name = f"analyses/{analysis.id}.json"
        data = _analysis_bytes(analysis)
        _check_payload(name, data)
        payloads[name] = data
    settings = _json_bytes(project.plot_settings, "plot_settings")
    _check_payload(PLOT_SETTINGS_NAME, settings)
    payloads[PLOT_SETTINGS_NAME] = settings
    return payloads


def _check_payload(name: str, data: bytes) -> None:
    if len(data) > MAX_ENTRY_BYTES:
        raise ProjectError(
            ProjectErrorCode.UNSUPPORTED_DATA,
            f"{name} is {len(data):,} bytes; the per-file limit is {MAX_ENTRY_BYTES:,} bytes",
        )


def _dataset_bytes(entry: ProjectDataset, specs: Sequence[ColumnSpec]) -> bytes:
    document = {
        "schema_version": SCHEMA_VERSION,
        "name": entry.name,
        "dataset_name": entry.dataset.name,
        "import_notes": list(entry.dataset.import_notes),
        "columns": [{"name": name, "dtype": dtype} for name, dtype in specs],
        "operations": [_operation_json(op) for op in entry.dataset.ops],
        "fingerprint": entry.dataset.fingerprint(),
    }
    return _json_bytes(document, f"dataset {entry.id!r}")


def _operation_json(op: Operation) -> dict[str, Any]:
    if isinstance(op, ExcludeRows):
        return {"op": "exclude_rows", "rows": list(op.rows)}
    if isinstance(op, TransformColumn):
        return {
            "op": "transform_column",
            "source": op.source,
            "target": op.target,
            "func": op.func,
            "scale": op.scale,
            "offset": op.offset,
        }
    if isinstance(op, RenameColumn):
        return {"op": "rename_column", "old": op.old, "new": op.new}
    raise ProjectError(ProjectErrorCode.UNSUPPORTED_DATA, f"unknown operation {op!r}")


def _analysis_bytes(analysis: Analysis) -> bytes:
    result: Any = analysis.result
    if isinstance(result, FitResult):
        result = json.loads(result.to_json())
    document = {
        "schema_version": SCHEMA_VERSION,
        "id": analysis.id,
        "dataset_id": analysis.dataset_id,
        "kind": analysis.kind,
        "config": analysis.config,
        "note": analysis.note,
        "result": result,
        "created_at": analysis.created_at,
    }
    return _json_bytes(document, f"analysis {analysis.id!r}")


def _json_bytes(value: Any, what: str) -> bytes:
    try:
        text = json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:
        raise ProjectError(
            ProjectErrorCode.UNSUPPORTED_DATA, f"{what} is not plain JSON data: {exc}"
        ) from exc
    return text.encode("utf-8")


def _write_zip(target: Path | IO[bytes], payloads: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in payloads.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix, so external_attr carries a real file mode
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)


def _write_path_atomically(target: Path, payloads: Mapping[str, bytes]) -> None:
    """Write the archive beside ``target``, then atomically replace it.

    Only a complete temporary file is renamed over the target, so a save that fails
    partway (disk full, permissions) leaves the previous file byte-identical. The
    temporary file is removed on any failure.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            _write_zip(handle.file, payloads)
            handle.flush()
            os.fsync(handle.fileno())
        assert temp_path is not None  # assigned as soon as the temporary file exists
        # mkstemp-style files are 0600; a project file should be an ordinary user file.
        if target.exists():
            os.chmod(temp_path, target.stat().st_mode & 0o777)
        else:
            os.chmod(temp_path, 0o644)
        os.replace(temp_path, target)
        temp_path = None
    except OSError as exc:
        raise ProjectError(ProjectErrorCode.SAVE_FAILED, f"cannot save {target!r}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name, distribution in _DISTRIBUTIONS.items():
        versions[name] = (
            platform.python_version() if distribution is None else _version_of(distribution)
        )
    return versions


def _version_of(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- loading ---------------------------------------------------------------------------


def _open_archive(source: Path | BinaryIO) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(source, "r")
    except FileNotFoundError as exc:
        raise ProjectError(ProjectErrorCode.NOT_A_PROJECT, "file not found") from exc
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        raise ProjectError(
            ProjectErrorCode.NOT_A_PROJECT, f"not a CurveStudio project archive: {exc}"
        ) from exc
    except (NotImplementedError, RuntimeError, zlib.error) as exc:
        # A well-formed ZIP that needs a version, compression method or feature this
        # reader does not implement (e.g. "version needed to extract" above 6.3).
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"the project archive uses an unsupported ZIP feature: {exc}",
        ) from exc


def _load(archive: zipfile.ZipFile) -> Project:
    entries = _inspect(archive)
    manifest = _read_manifest(archive, entries)
    payloads = _verified_payloads(archive, entries, manifest)
    return _build_project(manifest, payloads)


def _inspect(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Check every entry's name and declared metadata before reading any content."""
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise ProjectError(
            ProjectErrorCode.UNSAFE_ARCHIVE,
            f"the archive holds {len(infos):,} entries; the limit is {MAX_ENTRIES:,}",
        )
    entries: dict[str, zipfile.ZipInfo] = {}
    folded: dict[str, str] = {}
    total = 0
    for info in infos:
        _check_entry_name(info.filename, ProjectErrorCode.UNSAFE_ARCHIVE)
        if info.filename in entries:
            raise ProjectError(
                ProjectErrorCode.UNSAFE_ARCHIVE,
                f"duplicate archive entry {info.filename!r}",
            )
        key = info.filename.casefold()
        if key in folded:
            raise ProjectError(
                ProjectErrorCode.UNSAFE_ARCHIVE,
                f"archive entry {info.filename!r} collides case-insensitively with {folded[key]!r}",
            )
        folded[key] = info.filename
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ProjectError(
                ProjectErrorCode.UNSAFE_ARCHIVE,
                f"symlink entry {info.filename!r} is not allowed",
            )
        if info.flag_bits & 0x1:
            raise ProjectError(
                ProjectErrorCode.UNSAFE_ARCHIVE,
                f"encrypted entry {info.filename!r} is not supported",
            )
        if info.file_size > MAX_ENTRY_BYTES:
            raise ProjectError(
                ProjectErrorCode.UNSAFE_ARCHIVE,
                f"entry {info.filename!r} expands to {info.file_size:,} bytes; the limit is "
                f"{MAX_ENTRY_BYTES:,}",
            )
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ProjectError(
                ProjectErrorCode.UNSAFE_ARCHIVE,
                f"the archive expands to more than {MAX_TOTAL_BYTES:,} bytes",
            )
        if info.file_size > MIN_RATIO_CHECK_BYTES and (
            info.compress_size == 0 or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
        ):
            raise ProjectError(
                ProjectErrorCode.UNSAFE_ARCHIVE,
                f"entry {info.filename!r} has an unsafe compression ratio",
            )
        entries[info.filename] = info
    return entries


def _check_entry_name(name: str, code: ProjectErrorCode) -> None:
    problem = _name_problem(name)
    if problem is not None:
        raise ProjectError(code, f"unsafe archive entry name {name!r}: {problem}")


def _name_problem(name: str) -> str | None:
    if not name:
        return "empty name"
    if name.endswith("/"):
        return "directory entries are not allowed"
    if "\\" in name:
        return "backslashes are not allowed"
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in name):
        return "control characters are not allowed"
    path = PurePosixPath(name)
    if path.is_absolute():
        return "absolute paths are not allowed"
    parts = path.parts
    if not parts:
        return "empty path"
    if any(part == ".." for part in parts):
        return "parent directory references are not allowed"
    if ":" in parts[0]:
        return "drive letters are not allowed"
    return None


def _read_manifest(archive: zipfile.ZipFile, entries: Mapping[str, zipfile.ZipInfo]) -> Manifest:
    info = entries.get(MANIFEST_NAME)
    if info is None:
        raise ProjectError(ProjectErrorCode.NOT_A_PROJECT, "the archive has no manifest.json")
    if info.file_size > MAX_MANIFEST_BYTES:
        raise ProjectError(
            ProjectErrorCode.UNSAFE_ARCHIVE,
            f"manifest.json declares {info.file_size:,} bytes; the limit is {MAX_MANIFEST_BYTES:,}",
        )
    document = _json_object(_entry_bytes(archive, info), MANIFEST_NAME)
    listed = document.get("entries")
    if isinstance(listed, list) and len(listed) > MAX_ENTRIES:
        raise ProjectError(
            ProjectErrorCode.UNSAFE_ARCHIVE,
            f"manifest.json lists {len(listed):,} entries; the limit is {MAX_ENTRIES:,}",
        )
    version = document.get("schema_version")
    if version is None:
        raise ProjectError(ProjectErrorCode.CORRUPT, "manifest.json has no schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"manifest.json schema_version must be an integer, not {version!r}",
        )
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ProjectError(
            ProjectErrorCode.UNSUPPORTED_VERSION,
            f"project schema version {version} is not supported; this CurveStudio reads "
            f"versions {', '.join(str(v) for v in sorted(SUPPORTED_SCHEMA_VERSIONS))}",
        )
    try:
        return Manifest.model_validate(document)
    except ValidationError as exc:
        raise ProjectError(ProjectErrorCode.CORRUPT, f"manifest.json is invalid: {exc}") from exc


def _verified_payloads(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    manifest: Manifest,
) -> dict[str, bytes]:
    listed: dict[str, ManifestEntry] = {}
    for entry in manifest.entries:
        _check_entry_name(entry.path, ProjectErrorCode.UNSAFE_ARCHIVE)
        if entry.path == MANIFEST_NAME:
            raise ProjectError(ProjectErrorCode.CORRUPT, "manifest.json cannot list itself")
        if entry.path in listed:
            raise ProjectError(
                ProjectErrorCode.CORRUPT, f"manifest.json lists {entry.path!r} twice"
            )
        listed[entry.path] = entry
    unexpected = sorted(set(entries) - set(listed) - {MANIFEST_NAME})
    if unexpected:
        raise ProjectError(
            ProjectErrorCode.UNSAFE_ARCHIVE,
            f"archive entry {unexpected[0]!r} is not listed in manifest.json",
        )
    missing = sorted(set(listed) - set(entries))
    if missing:
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"manifest.json lists {missing[0]!r}, which is not in the archive",
        )
    payloads: dict[str, bytes] = {}
    for path, info in entries.items():
        if path == MANIFEST_NAME:
            continue
        data = _entry_bytes(archive, info)
        if _sha256(data) != listed[path].sha256:
            raise ProjectError(
                ProjectErrorCode.CHECKSUM_MISMATCH,
                f"{path} does not match its SHA-256 in manifest.json",
            )
        payloads[path] = data
    return payloads


def _entry_bytes(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    try:
        return archive.read(info)
    except (
        zipfile.BadZipFile,
        NotImplementedError,
        RuntimeError,
        OSError,
        EOFError,
        zlib.error,
    ) as exc:
        raise ProjectError(
            ProjectErrorCode.CORRUPT, f"cannot read archive entry {info.filename!r}: {exc}"
        ) from exc


def _json_object(data: bytes, path: str) -> dict[str, Any]:
    try:
        document = json.loads(data.decode("utf-8"), parse_constant=_reject_constant(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectError(
            ProjectErrorCode.CORRUPT, f"{path} is not valid UTF-8 JSON: {exc}"
        ) from exc
    except ValueError as exc:
        # int() inside json.loads refuses numbers longer than the interpreter's
        # digit limit; report that as a format failure, not a raw conversion error.
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"{path} is not valid UTF-8 JSON: {_number_error_detail(exc)}",
        ) from exc
    if not isinstance(document, dict):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} must hold a JSON object")
    return document


def _number_error_detail(exc: ValueError) -> str:
    message = str(exc)
    if "limit" in message.lower() and "digit" in message.lower():
        return "number too large"
    return message


def _reject_constant(path: str) -> Callable[[str], Any]:
    def reject(token: str) -> Any:
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"{path} contains the non-finite JSON constant {token!r}",
        )

    return reject


def _require_supported_version(document: Mapping[str, Any], path: str) -> None:
    """A payload's ``schema_version`` must be an integer this build reads."""
    version = document.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} has an unexpected schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ProjectError(
            ProjectErrorCode.UNSUPPORTED_VERSION,
            f"{path} has schema version {version}, which is not supported; this CurveStudio "
            f"reads versions {', '.join(str(v) for v in sorted(SUPPORTED_SCHEMA_VERSIONS))}",
        )


def _build_project(manifest: Manifest, payloads: Mapping[str, bytes]) -> Project:
    dataset_files: dict[str, dict[str, bytes]] = {}
    analyses: list[Analysis] = []
    settings: dict[str, Any] = {}
    for path, data in payloads.items():
        if path == PLOT_SETTINGS_NAME:
            settings = _json_object(data, path)
            continue
        dataset_match = _DATASET_FILE.fullmatch(path)
        if dataset_match is not None:
            kind = dataset_match.group("kind")
            dataset_files.setdefault(dataset_match.group("id"), {})[kind] = data
            continue
        analysis_match = _ANALYSIS_FILE.fullmatch(path)
        if analysis_match is not None:
            analyses.append(_load_analysis(analysis_match.group("id"), data))
            continue
        raise ProjectError(ProjectErrorCode.CORRUPT, f"unexpected file in the project: {path!r}")
    datasets = [_load_dataset(dataset_id, files) for dataset_id, files in dataset_files.items()]
    try:
        return Project(
            name=manifest.project_name,
            created_at=manifest.created_at,
            datasets=tuple(datasets),
            analyses=tuple(analyses),
            plot_settings=settings,
        )
    except ValidationError as exc:
        raise ProjectError(
            ProjectErrorCode.CORRUPT, f"the project model is invalid: {exc}"
        ) from exc


def _load_dataset(dataset_id: str, files: Mapping[str, bytes]) -> ProjectDataset:
    path = f"datasets/{dataset_id}/operations.json"
    missing = sorted({"original.csv", "cleaned.csv", "operations.json"} - set(files))
    if missing:
        raise ProjectError(
            ProjectErrorCode.CORRUPT, f"dataset {dataset_id!r} is missing {missing[0]!r}"
        )
    document = _json_object(files["operations.json"], path)
    _require_supported_version(document, path)
    name = _string(document, "name", path)
    dataset_name = _string(document, "dataset_name", path)
    import_notes = _string_list(document, "import_notes", path)
    specs = _column_specs(document, path)
    operations = tuple(
        _operation_from_json(item, path) for item in _list(document, "operations", path)
    )
    try:
        dataset = Dataset(
            name=dataset_name,
            original=decode_frame(files["original.csv"], specs),
            ops=operations,
            import_notes=tuple(import_notes),
        )
    except DataError as exc:
        raise ProjectError(
            ProjectErrorCode.CORRUPT, f"dataset {dataset_id!r} cannot be rebuilt: {exc}"
        ) from exc
    if dataset.fingerprint() != _string(document, "fingerprint", path):
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"dataset {dataset_id!r} does not match its stored fingerprint",
        )
    if encode_frame(dataset.table()) != files["cleaned.csv"]:
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"dataset {dataset_id!r}: replaying the operation log does not reproduce cleaned.csv",
        )
    return ProjectDataset(id=dataset_id, name=name, dataset=dataset)


def _load_analysis(file_id: str, data: bytes) -> Analysis:
    path = f"analyses/{file_id}.json"
    document = _json_object(data, path)
    _require_supported_version(document, path)
    analysis_id = _string(document, "id", path)
    if analysis_id != file_id:
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"{path} names analysis {analysis_id!r}, not {file_id!r}",
        )
    kind = document.get("kind")
    if kind not in ("fit", "autofit", "stat"):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} has an unknown kind {kind!r}")
    result_document = document.get("result")
    if not isinstance(result_document, dict):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} has no JSON object result")
    result: FitResult | dict[str, Any]
    if kind == "fit":
        try:
            result = FitResult.model_validate(result_document)
        except ValidationError as exc:
            raise ProjectError(
                ProjectErrorCode.CORRUPT, f"{path} has an invalid fit result: {exc}"
            ) from exc
    else:
        # an "autofit" or "stat" result: plain JSON, checked by the analysis model
        result = result_document
    try:
        return Analysis(
            id=analysis_id,
            dataset_id=_string(document, "dataset_id", path),
            kind=kind,
            config=_object(document, "config", path),
            result=result,
            note=_string(document, "note", path, default=""),
            created_at=_optional_iso(document, "created_at", path),
        )
    except ValidationError as exc:
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} is invalid: {exc}") from exc


# --- small typed readers ---------------------------------------------------------------


def _string(document: Mapping[str, Any], key: str, path: str, *, default: str | None = None) -> str:
    value = document.get(key, default)
    if not isinstance(value, str):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} field {key!r} must be a string")
    return value


def _optional_iso(document: Mapping[str, Any], key: str, path: str) -> str | None:
    """An optional timestamp field: absent or null reads as ``None``, never corrupt.

    An archive written before the field existed simply does not carry the key.
    """
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} field {key!r} must be a string")
    return value


def _string_list(document: Mapping[str, Any], key: str, path: str) -> list[str]:
    value = document.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProjectError(
            ProjectErrorCode.CORRUPT, f"{path} field {key!r} must be a list of strings"
        )
    return list(value)


def _list(document: Mapping[str, Any], key: str, path: str) -> list[Any]:
    value = document.get(key, [])
    if not isinstance(value, list):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} field {key!r} must be a list")
    return list(value)


def _object(document: Mapping[str, Any], key: str, path: str) -> dict[str, Any]:
    value = document.get(key, {})
    if not isinstance(value, dict):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} field {key!r} must be an object")
    return value


def _number(document: Mapping[str, Any], key: str, path: str, default: float) -> float:
    value = document.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} field {key!r} must be a number")
    return float(value)


def _column_specs(document: Mapping[str, Any], path: str) -> tuple[ColumnSpec, ...]:
    raw = document.get("columns")
    if not isinstance(raw, list) or not raw:
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} must declare at least one column")
    specs: list[ColumnSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} column entries must be objects")
        name, dtype = item.get("name"), item.get("dtype")
        if not isinstance(name, str) or not isinstance(dtype, str):
            raise ProjectError(
                ProjectErrorCode.CORRUPT,
                f"{path} column entries need string name and dtype",
            )
        specs.append((name, dtype))
    return tuple(specs)


def _operation_from_json(item: Any, path: str) -> Operation:
    if not isinstance(item, dict):
        raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} operation entries must be objects")
    tag = item.get("op")
    if tag == "exclude_rows":
        rows = item.get("rows")
        if not isinstance(rows, list) or not all(
            isinstance(row, int) and not isinstance(row, bool) for row in rows
        ):
            raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} exclude_rows needs integer rows")
        return ExcludeRows(tuple(rows))
    if tag == "transform_column":
        func = item.get("func")
        if func not in _TRANSFORMS:
            raise ProjectError(
                ProjectErrorCode.CORRUPT, f"{path} has an unknown transform {func!r}"
            )
        return TransformColumn(
            source=_string(item, "source", path),
            target=_string(item, "target", path),
            func=cast(TransformFunc, func),
            scale=_number(item, "scale", path, 1.0),
            offset=_number(item, "offset", path, 0.0),
        )
    if tag == "rename_column":
        return RenameColumn(old=_string(item, "old", path), new=_string(item, "new", path))
    raise ProjectError(ProjectErrorCode.CORRUPT, f"{path} holds an unknown operation {tag!r}")
