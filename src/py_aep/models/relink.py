"""Batch relinking of missing footage sources.

Implements `Project.relink_footage`: after a project moves machines or the
media folders are reorganised, a batch of stored source paths stop
resolving. Instead of calling `FootageItem.replace` one source at a time,
the whole batch is pre-flighted against one or more search roots and only
sources with exactly one candidate are ever modified.

Candidate discovery matches the *tail* of the stored path: a stored path
ending ``media/footage/a.png`` matches ``<root>/media/footage/a.png`` with
a longer suffix than a bare ``<root>/other/a.png``. The longest suffix
wins; several files tied at the best suffix length are ambiguous and the
source is skipped rather than guessed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Union, cast

from ..data.file_formats import get_file_format
from .import_options import CURRENT_VALUE, _CurrentValue
from .preferences import default_sequence_fps
from .sources.file import FileSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .items.av_item import AVItem
    from .items.footage import FootageItem
    from .project import Project

#: Entry statuses exposed on `RelinkItem.status`.
RELINK_PLANNED = "planned"
"""A unique candidate was found during a dry run; nothing was modified."""

RELINK_REPLACED = "replaced"
"""The source was replaced with its unique candidate."""

RELINK_NOT_FOUND = "not_found"
"""No file under the search roots matches the stored path tail."""

RELINK_AMBIGUOUS = "ambiguous"
"""Several files tie as the best suffix match; the source was not touched."""

RELINK_UNSUPPORTED_FORMAT = "unsupported_format"
"""The unique candidate has an extension After Effects / py_aep cannot import."""

RELINK_FAILED = "failed"
"""Building or swapping in the replacement source raised while executing."""

PathLike = Union[str, "os.PathLike[str]"]


class RelinkItem(NamedTuple):
    """One source (main footage or proxy) considered by a relink run.

    Attributes:
        item: The [AVItem][] the source belongs to (a [FootageItem][] for
            `"main"`, a [FootageItem][] or [CompItem][] for `"proxy"`).
        role: `"main"` for a footage main source, `"proxy"` for a proxy.
        old_path: The stored path that no longer resolves.
        new_path: The chosen candidate path, or `None` when no unique
            importable candidate was found (or the replacement failed).
        status: One of the `RELINK_*` status constants.
        candidates: Every file tied as the best match, sorted; more than
            one for an `ambiguous` entry, the single candidate otherwise.
        error: A human-readable explanation for an
            `unsupported_format` / `failed` entry, else `None`.
    """

    item: AVItem
    role: str
    old_path: str
    new_path: str | None
    status: str
    candidates: tuple[str, ...] = ()
    error: str | None = None


class RelinkResult:
    """Outcome of a `Project.relink_footage` run.

    Attributes:
        dry_run: `True` for a run that only planned replacements.
        completed: Sources with a unique importable candidate - replaced
            (`status="replaced"`) in execute mode, merely planned
            (`status="planned"`) in a dry run.
        skipped: Sources left untouched after the pre-flight: `not_found`,
            `ambiguous`, or `unsupported_format`.
        failed: Sources whose replacement raised an unexpected error after
            writing had started (execute mode only). The exception text is
            in `RelinkItem.error`; earlier completed replacements stand.
    """

    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.completed: list[RelinkItem] = []
        self.skipped: list[RelinkItem] = []
        self.failed: list[RelinkItem] = []

    @property
    def incomplete(self) -> list[RelinkItem]:
        """Every source the run could not resolve: pre-flight skips plus
        execution failures."""
        return [*self.skipped, *self.failed]

    @property
    def planned(self) -> list[RelinkItem]:
        """The unique-candidate entries of a dry run (same list as
        `completed`)."""
        return self.completed if self.dry_run else []

    def __len__(self) -> int:
        return len(self.completed) + len(self.skipped) + len(self.failed)

    def __bool__(self) -> bool:
        """`True` when every considered source was resolved."""
        return not self.skipped and not self.failed

    def __repr__(self) -> str:
        return (
            f"RelinkResult(dry_run={self.dry_run!r}, completed="
            f"{len(self.completed)}, skipped={len(self.skipped)}, "
            f"failed={len(self.failed)})"
        )


def _stored_path_exists(file: str) -> bool:
    """Whether a stored source path resolves on this machine.

    A project migrated from the other platform keeps that platform's
    separators (a Windows ``C:\\...`` path opened on macOS), so a
    backslash-style path is also tried with native separators.
    """
    if os.path.exists(file):
        return True
    if "\\" in file:
        return os.path.exists(file.replace("\\", os.sep))
    return False


_DRIVE_RE = re.compile(r"^[A-Za-z]:$")


def _stored_path_parts(file: str) -> list[str]:
    """Split a stored path (POSIX, Windows, or mixed separators) into the
    comparable sequence of path parts, dropping the drive/root noise."""
    parts = re.split(r"[\\/]+", file)
    if parts and _DRIVE_RE.match(parts[0]):
        parts = parts[1:]
    return [part for part in parts if part and part != "."]


def _index_root(root: Path) -> dict[tuple[str, ...], list[Path]]:
    """Map every file under `root` to the lower-cased tuple of its parts
    relative to the root. Unreadable folders are skipped rather than
    aborting the whole search."""
    index: dict[tuple[str, ...], list[Path]] = {}
    for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            key = tuple(part.lower() for part in rel.parts)
            index.setdefault(key, []).append(path)
    return index


def _best_candidates(
    orig_parts: Sequence[str],
    index: dict[tuple[str, ...], list[Path]],
) -> tuple[list[Path], int]:
    """Find the files under the indexed roots whose relative path best
    matches the tail of `orig_parts`.

    A match length of N means the last N parts of the stored path (its
    folder tail plus filename) equal the file's last N relative parts.
    Every file sharing the maximum match length is returned - one file
    means a unique candidate, several means an ambiguous name clash.
    """
    wanted = [part.lower() for part in orig_parts]
    best_len = 0
    best: list[Path] = []
    for key, paths in index.items():
        limit = min(len(key), len(wanted))
        match_len = 0
        for length in range(limit, 0, -1):
            if key[-length:] == tuple(wanted[-length:]):
                match_len = length
                break
        if match_len == 0 or match_len < best_len:
            continue
        if match_len > best_len:
            best_len = match_len
            best = []
        best.extend(paths)
    # The same physical file may be reachable through several roots.
    unique: dict[str, Path] = {}
    for path in best:
        try:
            resolved = os.path.realpath(path)
        except OSError:
            resolved = str(path)
        unique.setdefault(resolved, path)
    return list(unique.values()), best_len


def _importable(path: Path) -> tuple[bool, str | None]:
    """Static pre-flight gate: is the candidate's format one py_aep can
    build a source for? Media-header failures (an undecodable track) are
    not predictable here and surface as execution failures instead."""
    try:
        fmt = get_file_format(path.suffix)
    except ValueError:
        return False, f"Unsupported footage format: {path.suffix!r}"
    if fmt.opti == "unsupported":
        return False, (
            f"footage import does not yet support {path.suffix}: After "
            "Effects requires a format-specific opti header that has not "
            "been reverse-engineered yet."
        )
    return True, None


def _normalize_roots(
    search_roots: PathLike | Sequence[PathLike],
) -> list[Path]:
    roots_in: Sequence[PathLike]
    if isinstance(search_roots, (str, os.PathLike)):
        roots_in = [search_roots]
    else:
        roots_in = search_roots
    roots: list[Path] = []
    seen: set[str] = set()
    for root_in in roots_in:
        root = Path(root_in)
        if not root.is_dir():
            raise ValueError(f"search root is not an existing directory: {root}")
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _iter_targets(
    project: Project, include_proxies: bool
) -> list[tuple[AVItem, str, FileSource]]:
    """All missing file sources to consider, stably ordered: each
    footage's main source first, then footage proxies, then comp proxies
    (a composition has no main source)."""
    targets: list[tuple[AVItem, str, FileSource]] = []
    proxies: list[tuple[AVItem, FileSource]] = []
    for footage in project.footages:
        main = footage.main_source
        if isinstance(main, FileSource) and not _stored_path_exists(main.file):
            targets.append((footage, "main", main))
        if include_proxies:
            proxy = footage.proxy_source
            if isinstance(proxy, FileSource):
                proxies.append((footage, proxy))
    if include_proxies:
        for comp in project.compositions:
            proxy = comp.proxy_source
            if isinstance(proxy, FileSource):
                proxies.append((comp, proxy))
    for item, proxy in proxies:
        if not _stored_path_exists(proxy.file):
            targets.append((item, "proxy", proxy))
    return targets


def _replacement_source(
    old: FileSource, candidate: Path, project: Project
) -> FileSource:
    """Probe the candidate the same way the old source was imported: an
    image sequence stays a sequence (alphabetical ordering included)."""
    if old.target_is_folder:
        return FileSource._from_file(
            candidate,
            sequence=True,
            force_alphabetical=old._is_alphabetical,
            windows=project._windows,
            default_sequence_fps=default_sequence_fps(project._preferences),
        )
    return FileSource._from_file(candidate, windows=project._windows)


def _apply_replacement(
    item: AVItem,
    role: str,
    old: FileSource,
    candidate: Path,
) -> None:
    """Swap one missing source for its candidate.

    Main sources go through `FootageItem.replace` (single-layer bindings
    preserved with `CURRENT_VALUE`), which carries the Interpret Footage
    settings and leaves a user-assigned item name in place. Proxies are
    rebuilt the same way and additionally inherit the old proxy's
    interpretation - this is a path repair, not the "set new proxy"
    command, which would reset them to the import preferences - while the
    proxy assignment and its in-use flag are preserved as they were.
    """
    if role == "main":
        # Only footage items carry a main source (see `_iter_targets`).
        footage = cast("FootageItem", item)
        if old.target_is_folder:
            footage.replace_with_sequence(
                candidate, force_alphabetical=old._is_alphabetical
            )
            return
        layer_index: int | _CurrentValue | None = (
            CURRENT_VALUE if old.layer_name else None
        )
        footage.replace(candidate, layer_index)
        return

    new_source = _replacement_source(old, candidate, item._project)
    old._carry_interpretation_to(new_source)
    was_in_use = item.use_proxy
    item._set_proxy(new_source)
    if not was_in_use:
        # _set_proxy forces use_proxy on; an assigned-but-disabled proxy
        # only had its path repaired, so restore its disabled state.
        item.use_proxy = False


def relink_footage(
    project: Project,
    search_roots: PathLike | Sequence[PathLike],
    *,
    dry_run: bool = False,
    include_proxies: bool = True,
) -> RelinkResult:
    """Batch-relink missing footage sources (see
    [Project.relink_footage][py_aep.models.project.Project.relink_footage])."""
    roots = _normalize_roots(search_roots)
    index: dict[tuple[str, ...], list[Path]] = {}
    for root in roots:
        for key, paths in _index_root(root).items():
            index.setdefault(key, []).extend(paths)

    result = RelinkResult(dry_run=dry_run)
    # Phase 1: pre-flight the entire batch. No source is touched until
    # every stored path has either one importable candidate or a recorded
    # reason to skip it.
    plan: list[tuple[AVItem, str, FileSource, Path]] = []
    for item, role, old in _iter_targets(project, include_proxies):
        old_path = old.file
        candidates, _match_len = _best_candidates(
            _stored_path_parts(old_path), index
        )
        if not candidates:
            result.skipped.append(
                RelinkItem(item, role, old_path, None, RELINK_NOT_FOUND)
            )
            continue
        candidate_paths = tuple(sorted(str(path) for path in candidates))
        if len(candidates) > 1:
            result.skipped.append(
                RelinkItem(
                    item,
                    role,
                    old_path,
                    None,
                    RELINK_AMBIGUOUS,
                    candidates=candidate_paths,
                )
            )
            continue
        candidate = candidates[0]
        importable, error = _importable(candidate)
        if not importable:
            result.skipped.append(
                RelinkItem(
                    item,
                    role,
                    old_path,
                    str(candidate),
                    RELINK_UNSUPPORTED_FORMAT,
                    candidates=candidate_paths,
                    error=error,
                )
            )
            continue
        plan.append((item, role, old, candidate))

    if dry_run:
        for item, role, old, candidate in plan:
            result.completed.append(
                RelinkItem(
                    item,
                    role,
                    old.file,
                    str(candidate),
                    RELINK_PLANNED,
                    candidates=(str(candidate),),
                )
            )
        return result

    # Phase 2: apply the confirmed plan. A failure on one source is
    # recorded with the original exception text and never prevents the
    # remaining confirmed replacements from being written.
    for item, role, old, candidate in plan:
        try:
            _apply_replacement(item, role, old, candidate)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            result.failed.append(
                RelinkItem(
                    item,
                    role,
                    old.file,
                    str(candidate),
                    RELINK_FAILED,
                    candidates=(str(candidate),),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            result.completed.append(
                RelinkItem(
                    item,
                    role,
                    old.file,
                    str(candidate),
                    RELINK_REPLACED,
                    candidates=(str(candidate),),
                )
            )
    return result
