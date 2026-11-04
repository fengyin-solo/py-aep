"""Read-only dependency audit for a [Project][py_aep.models.project.Project].

The audit walks the project tree without parsing layer bodies, reading media,
or mutating any project object, and answers the questions a pre-delivery
check asks:

- does every footage's main source / proxy file exist on disk,
- which stored paths are missing (main source vs proxy kept separate),
- which source file is referenced by more than one
  [FootageItem][py_aep.models.items.footage.FootageItem],
- which footage and compositions no composition uses,
- whether the nested-composition graph contains a cycle.

File existence is checked the way After Effects locates footage: the stored
absolute path first (only when it is written in the auditing machine's path
style), then the project-relative fallback reconstructed from the `alas`
record's `ascendcount_*` counts, resolved against an explicit working
directory or, by default, the parsed project file's folder. A path that
cannot be parsed or stated is reported as missing - never raised.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from ..binary.utils import parse_alas_data
from ..resolvers.platform_paths import path_flavor, relative_source_path
from .items.composition import CompItem
from .items.folder import FolderItem
from .items.footage import FootageItem
from .sources.file import FileSource
from .sources.placeholder import PlaceholderSource
from .sources.solid import SolidSource

if TYPE_CHECKING:
    from .sources.footage import FootageSource

#: The main source slot of a footage item.
ROLE_MAIN = "main"

#: The proxy source slot of a footage item.
ROLE_PROXY = "proxy"


class SourceAudit(NamedTuple):
    """Existence information for one source slot of a footage item.

    Solid and placeholder sources have no backing file: `path` is empty,
    `exists` is `None` (not applicable) and the source can never appear in
    [DependencyAuditReport.missing_paths][].
    """

    role: str
    """`"main"` or `"proxy"`."""

    kind: str
    """`"file"`, `"solid"` or `"placeholder"`."""

    path: str
    """The stored source path in the project's platform style, or `""` for a
    solid/placeholder source."""

    exists: bool | None
    """Whether the file was found on disk (`True`/`False`), or `None` when the
    source has no backing file."""

    missing_at_save: bool
    """Whether After Effects marked the footage missing when the project was
    last saved (the `sspc` flag), independent of the live existence check."""

    relative_path: str
    """The project-relative fallback location reconstructed from the record's
    `ascendcount_*` counts (POSIX-style), or `""` when the record defines no
    relative path."""


class FootageAuditRecord(NamedTuple):
    """A stably-ordered view of one footage item."""

    id: int
    """The footage item id."""

    name: str
    """The footage item name."""

    folder_id: int
    """The containing folder's id (0 for the root folder)."""

    main: SourceAudit
    """The main source."""

    proxy: SourceAudit | None
    """The proxy source, or `None` when the footage has no proxy assigned."""

    used_by_compositions: tuple[int, ...]
    """Sorted ids of the compositions that use this footage."""


class CompositionAuditRecord(NamedTuple):
    """A stably-ordered view of one composition."""

    id: int
    """The composition item id."""

    name: str
    """The composition name."""

    folder_id: int
    """The containing folder's id (0 for the root folder)."""

    uses_footage: tuple[int, ...]
    """Sorted ids of the footage items the composition's layers reference."""

    uses_compositions: tuple[int, ...]
    """Sorted ids of the nested compositions the composition's layers
    reference."""

    used_by_compositions: tuple[int, ...]
    """Sorted ids of the compositions that nest this composition."""


class FolderAuditRecord(NamedTuple):
    """A stably-ordered view of one project-panel folder."""

    id: int
    """The folder item id (0 for the root folder)."""

    name: str
    """The folder name (`"root"` for the virtual root folder)."""

    parent_id: int | None
    """The parent folder's id, or `None` for the root folder."""

    is_root: bool
    """`True` for the virtual root folder."""

    item_ids: tuple[int, ...]
    """Sorted ids of the items directly contained in the folder."""


class MissingPathRecord(NamedTuple):
    """A file-backed source slot whose file could not be located on disk."""

    footage_id: int
    """The footage item id."""

    footage_name: str
    """The footage item name."""

    role: str
    """`"main"` or `"proxy"`."""

    path: str
    """The stored path that is missing."""


class DuplicateSourceRecord(NamedTuple):
    """A source file referenced by more than one footage item."""

    path: str
    """The shared stored path (as the first involved footage records it)."""

    footage_ids: tuple[int, ...]
    """Sorted ids of the footage items referencing `path` as their main
    source."""


class DependencyAuditReport(NamedTuple):
    """The result of [Project.audit_dependencies][].

    Every collection is a tuple sorted by stable ids or paths, so running the
    audit twice on the same project yields equal reports.
    """

    footage: tuple[FootageAuditRecord, ...]
    """All footage items, sorted by id."""

    compositions: tuple[CompositionAuditRecord, ...]
    """All compositions, sorted by id."""

    folders: tuple[FolderAuditRecord, ...]
    """All folders (including the root folder), sorted by id."""

    missing_paths: tuple[MissingPathRecord, ...]
    """Missing main/proxy files, sorted by footage id then role."""

    duplicate_sources: tuple[DuplicateSourceRecord, ...]
    """Main-source paths referenced by several footage items, sorted by path
    then footage ids."""

    unused_footage: tuple[int, ...]
    """Sorted ids of footage not used by any composition."""

    unused_compositions: tuple[int, ...]
    """Sorted ids of compositions not nested in any composition."""

    cycles: tuple[tuple[int, ...], ...]
    """Cycles in the nested-composition graph: each entry is one strongly
    connected component (a self-nesting comp reports a one-id cycle), sorted
    by its sorted id tuple."""


def _host_path_flavor() -> str:
    return "windows" if os.name == "nt" else "macos"


def _safe_exists(path: Path) -> bool:
    """`os.path.exists` that turns every lookup failure (bad path syntax,
    broken encoding, permissions) into "missing" - a missing file must never
    surface as a parsing or operating-system error."""
    try:
        return path.exists()
    except (OSError, ValueError):
        return False


def _audit_base_dir(
    project: Any, workdir: str | os.PathLike[str] | None
) -> Path | None:
    """The folder relative paths are interpreted from.

    An explicit `workdir` wins; otherwise the parsed project file's folder is
    used, mirroring After Effects. A project with no file location (e.g. one
    built via `py_aep.new()`) has no base directory.
    """
    if workdir is not None:
        return Path(os.fspath(workdir))
    project_file = getattr(project, "file", "")
    return Path(project_file).parent if project_file else None


def _audit_file_source(
    source: FileSource,
    role: str,
    base_dir: Path | None,
    host_flavor: str,
) -> SourceAudit:
    """Build the [SourceAudit][] for a [FileSource][], locating the file the
    way After Effects does: absolute path first, relative fallback second."""
    relative = ""
    exists = False
    try:
        alas = parse_alas_data(source._pin.chunks)
        fullpath = str(alas.get("fullpath", "")) or source._file
        target_is_folder = bool(alas.get("target_is_folder", False))
        try:
            ascend_base = int(alas.get("ascendcount_base", 0))
            ascend_target = int(alas.get("ascendcount_target", 0))
        except (TypeError, ValueError):
            ascend_base = ascend_target = 0

        stored_path = source._file
        flavor = path_flavor(stored_path)
        absolute_exists = (
            flavor == host_flavor and _safe_exists(Path(stored_path))
        )

        if ascend_base > 0 and ascend_target > 0:
            relative = relative_source_path(
                fullpath,
                ascendcount_base=ascend_base,
                ascendcount_target=ascend_target,
                target_is_folder=target_is_folder,
            )
            # A sequence's record points at the folder; the first StVc frame
            # is the actual file After Effects opens.
            if relative and target_is_folder and source._file_names:
                relative = f"{relative}/{source._file_names[0]}"

        relative_exists = False
        if relative and base_dir is not None:
            relative_exists = _safe_exists(base_dir / relative)

        exists = absolute_exists or relative_exists
        path = stored_path
    except Exception:  # noqa: BLE001 - any record problem is "missing"
        path = getattr(source, "_file", "")
    return SourceAudit(
        role=role,
        kind="file",
        path=path,
        exists=exists,
        missing_at_save=bool(source._footage_missing),
        relative_path=relative,
    )


def _audit_inline_source(source: FootageSource, role: str) -> SourceAudit:
    """A solid or placeholder has no backing file and is never missing."""
    if isinstance(source, SolidSource):
        kind = "solid"
    elif isinstance(source, PlaceholderSource):
        kind = "placeholder"
    else:  # pragma: no cover - defensive, FootageSource is abstract in practice
        kind = type(source).__name__
    return SourceAudit(
        role=role,
        kind=kind,
        path="",
        exists=None,
        missing_at_save=bool(getattr(source, "_footage_missing", False)),
        relative_path="",
    )


def _audit_source(
    source: FootageSource,
    role: str,
    base_dir: Path | None,
    host_flavor: str,
) -> SourceAudit:
    if isinstance(source, FileSource):
        return _audit_file_source(source, role, base_dir, host_flavor)
    return _audit_inline_source(source, role)


def _duplicate_key(source: FileSource) -> tuple[str, str] | None:
    """Identity of a source file for duplicate grouping.

    Windows-style paths compare case-insensitively (as After Effects does);
    a path in no recognized absolute style is grouped verbatim. Empty paths
    do not participate.
    """
    path = source._file
    if not path:
        return None
    if path_flavor(path) == "windows":
        return ("windows", path.replace("\\", "/").lower())
    if path_flavor(path) == "macos":
        return ("macos", path)
    return ("other", path)


def _cyclic_components(
    comp_ids: set[int], edges: dict[int, set[int]]
) -> list[tuple[int, ...]]:
    """Strongly connected components of the comp-to-comp graph that contain a
    cycle (an SCC with more than one member, or a self-nesting comp).

    Iterative Tarjan so a deeply nested project cannot hit Python's recursion
    limit. Neighbours are visited in id order, keeping the SCC contents
    order-independent.
    """
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    counter = 0
    cycles: list[tuple[int, ...]] = []

    for root in sorted(comp_ids):
        if root in index:
            continue
        index[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work: list[tuple[int, Any]] = [
            (root, iter(sorted(edges.get(root, ()))))
        ]
        while work:
            vertex, neighbours = work[-1]
            descended = False
            for neighbour in neighbours:
                if neighbour not in comp_ids:
                    continue
                if neighbour not in index:
                    index[neighbour] = lowlink[neighbour] = counter
                    counter += 1
                    stack.append(neighbour)
                    on_stack.add(neighbour)
                    work.append(
                        (neighbour, iter(sorted(edges.get(neighbour, ()))))
                    )
                    descended = True
                    break
                if neighbour in on_stack:
                    lowlink[vertex] = min(lowlink[vertex], index[neighbour])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[vertex])
            if lowlink[vertex] == index[vertex]:
                component: list[int] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == vertex:
                        break
                if len(component) > 1 or vertex in edges.get(vertex, set()):
                    cycles.append(tuple(sorted(component)))
    return sorted(cycles)


def audit_dependencies(
    project: Any,
    workdir: str | os.PathLike[str] | None = None,
) -> DependencyAuditReport:
    """Audit a project's footage/composition dependencies without changing it.

    See [Project.audit_dependencies][py_aep.models.project.Project.audit_dependencies]
    for the public entry point and argument semantics.
    """
    base_dir = _audit_base_dir(project, workdir)
    host_flavor = _host_path_flavor()

    all_items = project.items
    footages: list[FootageItem] = [
        item for item in all_items.values() if isinstance(item, FootageItem)
    ]
    comps: list[CompItem] = [
        item for item in all_items.values() if isinstance(item, CompItem)
    ]
    folders: list[FolderItem] = [
        item for item in all_items.values() if isinstance(item, FolderItem)
    ]
    footages.sort(key=lambda item: item.id)
    comps.sort(key=lambda item: item.id)
    folders.sort(key=lambda item: item.id)

    footage_ids = {item.id for item in footages}
    comp_ids = {item.id for item in comps}

    # Layer-reference graph straight from source ids, without forcing the
    # deferred layer parse or touching the mutable used_in back-references.
    edges: dict[int, set[int]] = {}
    used_by: dict[int, set[int]] = {}
    for comp in comps:
        referenced: set[int] = set()
        for source_id in comp._source_ids_for_linking():
            if source_id in footage_ids or source_id in comp_ids:
                referenced.add(source_id)
                used_by.setdefault(source_id, set()).add(comp.id)
        edges[comp.id] = referenced

    footage_records: list[FootageAuditRecord] = []
    missing: list[MissingPathRecord] = []
    duplicate_members: dict[tuple[str, str], list[FootageItem]] = {}
    duplicate_paths: dict[tuple[str, str], str] = {}

    for footage in footages:
        main_audit = _audit_source(
            footage.main_source, ROLE_MAIN, base_dir, host_flavor
        )
        proxy_source = footage.proxy_source
        proxy_audit: SourceAudit | None = None
        if proxy_source is not None:
            proxy_audit = _audit_source(
                proxy_source, ROLE_PROXY, base_dir, host_flavor
            )

        footage_records.append(
            FootageAuditRecord(
                id=footage.id,
                name=footage.name,
                folder_id=(
                    footage.parent_folder.id
                    if footage.parent_folder is not None
                    else 0
                ),
                main=main_audit,
                proxy=proxy_audit,
                used_by_compositions=tuple(
                    sorted(used_by.get(footage.id, set()))
                ),
            )
        )

        if main_audit.kind == "file" and main_audit.exists is False:
            missing.append(
                MissingPathRecord(
                    footage_id=footage.id,
                    footage_name=footage.name,
                    role=ROLE_MAIN,
                    path=main_audit.path,
                )
            )
        if proxy_audit is not None and proxy_audit.exists is False:
            missing.append(
                MissingPathRecord(
                    footage_id=footage.id,
                    footage_name=footage.name,
                    role=ROLE_PROXY,
                    path=proxy_audit.path,
                )
            )

        source = footage.main_source
        if isinstance(source, FileSource):
            key = _duplicate_key(source)
            if key is not None:
                duplicate_members.setdefault(key, []).append(footage)
                duplicate_paths.setdefault(key, source._file)

    comp_records: list[CompositionAuditRecord] = []
    for comp in comps:
        referenced = edges.get(comp.id, set())
        comp_records.append(
            CompositionAuditRecord(
                id=comp.id,
                name=comp.name,
                folder_id=(
                    comp.parent_folder.id if comp.parent_folder is not None else 0
                ),
                uses_footage=tuple(sorted(referenced & footage_ids)),
                uses_compositions=tuple(sorted(referenced & comp_ids)),
                used_by_compositions=tuple(sorted(used_by.get(comp.id, set()))),
            )
        )

    folder_records: list[FolderAuditRecord] = []
    for folder in folders:
        parent = folder.parent_folder
        folder_records.append(
            FolderAuditRecord(
                id=folder.id,
                name=folder.name,
                parent_id=parent.id if parent is not None else None,
                is_root=parent is None,
                item_ids=tuple(sorted(item.id for item in folder.items)),
            )
        )

    duplicate_records = [
        DuplicateSourceRecord(
            path=duplicate_paths[key],
            footage_ids=tuple(sorted(item.id for item in members)),
        )
        for key, members in duplicate_members.items()
        if len(members) > 1
    ]
    duplicate_records.sort(key=lambda record: (record.path, record.footage_ids))

    missing.sort(
        key=lambda record: (
            record.footage_id,
            0 if record.role == ROLE_MAIN else 1,
            record.path,
        )
    )

    return DependencyAuditReport(
        footage=tuple(footage_records),
        compositions=tuple(comp_records),
        folders=tuple(folder_records),
        missing_paths=tuple(missing),
        duplicate_sources=tuple(duplicate_records),
        unused_footage=tuple(
            item.id for item in footages if item.id not in used_by
        ),
        unused_compositions=tuple(
            item.id for item in comps if item.id not in used_by
        ),
        cycles=tuple(_cyclic_components(comp_ids, edges)),
    )
