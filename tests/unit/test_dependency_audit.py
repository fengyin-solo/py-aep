"""Tests for Project.audit_dependencies(): the read-only dependency audit.

Covers main/proxy existence, missing paths, duplicate source references,
unused items, nested compositions, project-relative path resolution, empty /
folder-only projects, nested-graph cycles and repeat-run stability.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

from helpers import project_bytes

import py_aep
from py_aep import parse as parse_aep
from py_aep.binary.utils import find_by_list_type, find_by_type
from py_aep.models.items.footage import FootageItem
from py_aep.models.sources.file import FileSource

SAMPLES = Path(__file__).parent.parent.parent / "samples"
ITEM_SAMPLE = SAMPLES / "models" / "item" / "proxy.aep"
MISSING_SAMPLE = SAMPLES / "models" / "footage" / "footage_missing.aep"
PRESENT_SAMPLE = SAMPLES / "models" / "footage" / "footage_not_missing.aep"
COMPLETE_SAMPLE = SAMPLES / "versions" / "ae2025" / "complete.aep"

_HOST_WINDOWS = os.name == "nt"


def _add_file_footage(
    project: py_aep.Project,
    path: str | Path,
    *,
    windows: bool | None = None,
) -> py_aep.FootageItem:
    """Register a file footage backed by an unresearched FileSource.

    Mirrors Project.import_placeholder's registration without touching the
    file on disk (the audit must work without reading media either).
    """
    source = FileSource._new(
        os.fspath(path),
        source_format="png!",
        width=64,
        height=32,
        duration=0,
        frame_rate=0,
        windows=_HOST_WINDOWS if windows is None else windows,
    )
    item = FootageItem._new(
        source,
        project=project,
        parent_folder=project.root_folder,
    )
    container = project.root_folder._children_container
    container.append(item._item_list)
    container.extend(item._view_data)
    project.items[item.id] = item
    project.root_folder.items.append(item)
    return item


def _footage_record(report: py_aep.DependencyAuditReport, item_id: int):
    records = [record for record in report.footage if record.id == item_id]
    assert len(records) == 1
    return records[0]


def _comp_record(report: py_aep.DependencyAuditReport, item_id: int):
    records = [record for record in report.compositions if record.id == item_id]
    assert len(records) == 1
    return records[0]


# ---- Empty and folder-only projects ------------------------------------------------


class TestEmptyProjects:
    def test_empty_project_returns_valid_report(self) -> None:
        report = py_aep.new().project.audit_dependencies()

        assert report.footage == ()
        assert report.compositions == ()
        assert report.missing_paths == ()
        assert report.duplicate_sources == ()
        assert report.unused_footage == ()
        assert report.unused_compositions == ()
        assert report.cycles == ()
        # The virtual root folder is always reported.
        assert len(report.folders) == 1
        root = report.folders[0]
        assert root.is_root is True
        assert root.parent_id is None
        assert root.item_ids == ()

    def test_folder_only_project_returns_valid_report(self) -> None:
        project = py_aep.new().project
        deliver = project.root_folder.add_folder("Deliver")
        deliver.add_folder("Nested")

        report = project.audit_dependencies()

        assert report.footage == ()
        assert report.compositions == ()
        assert report.missing_paths == ()
        assert report.duplicate_sources == ()
        assert report.cycles == ()
        assert [folder.name for folder in report.folders] == [
            "root",
            "Deliver",
            "Nested",
        ]
        root, deliver_record, nested_record = report.folders
        assert root.parent_id is None
        assert deliver_record.parent_id == root.id
        assert deliver_record.item_ids == (nested_record.id,)
        assert nested_record.parent_id == deliver_record.id
        assert nested_record.item_ids == ()


# ---- Main / proxy existence and missing paths ---------------------------------------


class TestMissingPaths:
    def test_main_footage_missing(self) -> None:
        project = parse_aep(MISSING_SAMPLE).project
        footage = project.footages[0]

        report = project.audit_dependencies()
        record = _footage_record(report, footage.id)

        assert record.main.kind == "file"
        assert record.main.exists is False
        assert record.main.missing_at_save is True
        assert record.proxy is None
        assert report.missing_paths == (
            py_aep.MissingPathRecord(
                footage_id=footage.id,
                footage_name=footage.name,
                role="main",
                path=record.main.path,
            ),
        )

    def test_live_missing_independent_of_missing_at_save_flag(self) -> None:
        """A source AE did not flag at save time is still reported missing
        when it cannot be located now ("missing" is a live check, not a parse
        error)."""
        project = parse_aep(PRESENT_SAMPLE).project

        report = project.audit_dependencies()
        record = report.footage[0]

        assert record.main.missing_at_save is False
        assert record.main.exists is False
        assert len(report.missing_paths) == 1
        assert report.missing_paths[0].role == "main"

    def test_main_and_proxy_existing_are_separated(self) -> None:
        project = parse_aep(ITEM_SAMPLE).project

        report = project.audit_dependencies()
        by_name = {record.name: record for record in report.footage}

        enabled = by_name["proxy_enabled"]
        assert enabled.main.exists is True
        assert enabled.proxy is not None
        assert enabled.proxy.role == "proxy"
        assert enabled.proxy.kind == "file"
        assert enabled.proxy.exists is True

        disabled = by_name["proxy_disabled"]
        assert disabled.main.exists is True
        assert disabled.proxy is not None
        assert disabled.proxy.exists is True

        no_proxy = by_name["no_proxy"]
        assert no_proxy.main.exists is True
        assert no_proxy.proxy is None

        # Everything resolves through the in-repo relative fallback.
        assert report.missing_paths == ()

    def test_proxy_missing_but_main_present(self) -> None:
        project = parse_aep(ITEM_SAMPLE).project
        footage = next(item for item in project.footages if item.name == "proxy_enabled")

        missing_proxy = FileSource._new(
            "C:/audit_dependency_nonexistent/proxy.png",
            source_format="png!",
            width=8,
            height=8,
            duration=0,
            frame_rate=0,
            windows=True,
        )
        with warnings.catch_warnings():
            # The sample is an OCIO project whose config is not on this
            # machine; _join_project warns, irrelevant to the audit.
            warnings.simplefilter("ignore")
            footage._set_proxy(missing_proxy)

        report = project.audit_dependencies()
        record = _footage_record(report, footage.id)

        assert record.main.exists is True
        assert record.proxy is not None
        assert record.proxy.exists is False
        proxy_missing = [
            entry for entry in report.missing_paths if entry.role == "proxy"
        ]
        assert len(proxy_missing) == 1
        assert proxy_missing[0].footage_id == footage.id
        assert proxy_missing[0].path.endswith("proxy.png")
        # The main source is not on the missing list.
        assert all(entry.role != "main" for entry in report.missing_paths)

    def test_missing_path_does_not_raise_on_unparseable_record(self) -> None:
        """A malformed path record is reported missing, never raised."""
        project = py_aep.new().project
        footage = _add_file_footage(project, str(Path.cwd() / "never_created.png"))
        source = footage.main_source
        assert isinstance(source, FileSource)

        # Corrupt the alas record: valid JSON envelope, unstatable fullpath
        # and relative counts pointing at it.
        als2 = find_by_list_type(source._pin.chunks, list_type="Als2")
        alas = find_by_type(als2.chunks, chunk_type="alas")
        data = json.loads(alas.value)
        data["fullpath"] = "C:/bad/\x00/folder/x.png"
        data["ascendcount_base"] = 3
        data["ascendcount_target"] = 3
        alas.value = json.dumps(data)
        source._file = data["fullpath"]

        report = project.audit_dependencies(workdir=Path.cwd())
        record = _footage_record(report, footage.id)
        assert record.main.exists is False
        assert [entry.footage_id for entry in report.missing_paths] == [footage.id]

    def test_zero_byte_file_is_present_without_content_read(self, tmp_path: Path) -> None:
        project = py_aep.new().project
        empty_media = tmp_path / "not_really_a_png.png"
        empty_media.write_bytes(b"")
        footage = _add_file_footage(project, str(empty_media))

        report = project.audit_dependencies()
        record = _footage_record(report, footage.id)
        assert record.main.exists is True
        assert report.missing_paths == ()


# ---- Relative path resolution --------------------------------------------------------


class TestRelativePaths:
    def test_default_base_is_project_file_folder(self) -> None:
        # The sample was saved on Windows; the absolute path cannot exist on
        # this machine, but its ascend counts locate the assets two folders
        # up from the parsed project file - where they live in this repo.
        project = parse_aep(ITEM_SAMPLE).project
        report = project.audit_dependencies()

        enabled = next(
            record for record in report.footage if record.name == "proxy_enabled"
        )
        assert enabled.main.relative_path == "../../assets/mov_23_976.mov"
        assert enabled.main.exists is True
        assert enabled.proxy is not None
        assert enabled.proxy.relative_path == "../../assets/image_with_alpha.png"
        assert enabled.proxy.exists is True

    def test_explicit_workdir_replaces_project_folder(self, tmp_path: Path) -> None:
        project = parse_aep(ITEM_SAMPLE).project

        # Empty stand-in: nothing exists there -> everything is missing
        # through the relative fallback.
        empty_workdir = tmp_path / "models" / "item"
        empty_workdir.mkdir(parents=True)
        missing_report = project.audit_dependencies(workdir=empty_workdir)
        assert len(missing_report.missing_paths) == 5  # 3 main + 2 proxy slots

        # Populate the tree the ascend counts describe (up 2, then assets/):
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "mov_23_976.mov").write_bytes(b"")
        (assets / "image_with_alpha.png").write_bytes(b"")
        workdir = tmp_path / "models" / "item"
        report = project.audit_dependencies(workdir=workdir)

        assert report.missing_paths == ()
        assert all(
            record.main.exists for record in report.footage if record.main.kind == "file"
        )
        assert all(
            record.proxy.exists
            for record in report.footage
            if record.proxy is not None
        )


# ---- Duplicate source references -----------------------------------------------------


class TestDuplicateSources:
    def test_same_file_referenced_by_two_footage_items(self, tmp_path: Path) -> None:
        project = py_aep.new().project
        shared = tmp_path / "shared.png"
        shared.write_bytes(b"")
        first = _add_file_footage(project, str(shared))
        second = _add_file_footage(project, str(shared))
        other = _add_file_footage(project, str(tmp_path / "other.png"))

        report = project.audit_dependencies()

        assert report.duplicate_sources == (
            py_aep.DuplicateSourceRecord(
                path=str(shared),
                footage_ids=tuple(sorted((first.id, second.id))),
            ),
        )
        assert other.id not in report.duplicate_sources[0].footage_ids

    def test_sample_duplicate_group(self) -> None:
        project = parse_aep(ITEM_SAMPLE).project
        report = project.audit_dependencies()

        assert len(report.duplicate_sources) == 1
        group = report.duplicate_sources[0]
        assert group.path.endswith("mov_23_976.mov")
        assert len(group.footage_ids) == 3

    def test_windows_paths_compare_case_insensitively(self) -> None:
        project = py_aep.new().project
        upper = _add_file_footage(
            project, "C:/Audit Dependency/Media/clip.png", windows=True
        )
        lower = _add_file_footage(
            project, "c:/audit dependency/media/clip.png",
            windows=True,
        )

        report = project.audit_dependencies()

        assert len(report.duplicate_sources) == 1
        assert report.duplicate_sources[0].footage_ids == tuple(
            sorted((upper.id, lower.id))
        )


# ---- Unused items --------------------------------------------------------------------


class TestUnusedItems:
    def test_unused_comp_and_footage_in_built_project(self, tmp_path: Path) -> None:
        project = py_aep.new().project
        comp = project.root_folder.add_comp("Used Comp", 1920, 1080, 1.0, 10.0, 24.0)
        comp.add_solid([1, 0, 0], "Red")
        project.root_folder.add_comp("Orphan Comp", 1280, 720, 1.0, 5.0, 24.0)
        unused = _add_file_footage(project, str(tmp_path / "unused.png"))

        report = project.audit_dependencies()

        assert unused.id in report.unused_footage
        orphan = next(item for item in project.compositions if item.name == "Orphan Comp")
        assert orphan.id in report.unused_compositions
        used_comp = next(
            item for item in project.compositions if item.name == "Used Comp"
        )
        assert used_comp.id in report.unused_compositions  # top-level comp
        # The solid footage backing the red solid layer is used.
        solid = next(
            item for item in project.footages if item.name == "Red"
        )
        assert solid.id not in report.unused_footage

    def test_sample_usage_lists(self) -> None:
        project = parse_aep(COMPLETE_SAMPLE).project
        report = project.audit_dependencies()
        names = {item.id: item.name for item in project.footages}

        unused_names = {names[item_id] for item_id in report.unused_footage}
        # The background used inside Pre_Comp is not unused; the two stray
        # footages nothing references are.
        precomp_bg = next(item for item in project.footages if item.id == 76)
        assert precomp_bg.name not in unused_names
        assert sorted(unused_names) == sorted(names[i] for i in (6, 10))

        comp_names = {item.id: item.name for item in project.compositions}
        unused_comp_names = {
            comp_names[item_id] for item_id in report.unused_compositions
        }
        assert "Pre_Comp" not in unused_comp_names
        assert "Main_Comp" in unused_comp_names


# ---- Nested compositions and cycles --------------------------------------------------


class TestNesting:
    def test_nested_composition_dependencies(self) -> None:
        project = parse_aep(COMPLETE_SAMPLE).project
        report = project.audit_dependencies()

        main_id = next(item.id for item in project.compositions if item.name == "Main_Comp")
        pre_id = next(item.id for item in project.compositions if item.name == "Pre_Comp")

        main_record = _comp_record(report, main_id)
        pre_record = _comp_record(report, pre_id)
        assert pre_id in main_record.uses_compositions
        assert main_record.used_by_compositions == ()
        assert pre_record.uses_compositions == ()
        assert pre_record.used_by_compositions == (main_id,)
        # Pre_Comp references its background footage.
        footage_ids = {record.id for record in report.footage}
        assert {76} <= footage_ids
        assert 76 in pre_record.uses_footage

    def test_self_nesting_is_a_cycle(self) -> None:
        project = parse_aep(COMPLETE_SAMPLE).project
        main = next(item for item in project.compositions if item.name == "Main_Comp")
        layer = next(layer for layer in main.layers if layer.name == "Nested_Precomp")
        layer._source_id = main.id
        if hasattr(layer, "_source"):
            delattr(layer, "_source")

        report = project.audit_dependencies()

        assert report.cycles == ((main.id,),)

    def test_two_composition_cycle(self) -> None:
        project = parse_aep(COMPLETE_SAMPLE).project
        main = next(item for item in project.compositions if item.name == "Main_Comp")
        pre = next(item for item in project.compositions if item.name == "Pre_Comp")
        layer = next(layer for layer in pre.layers if layer.name == "Precomp_BG")
        layer._source_id = main.id
        if hasattr(layer, "_source"):
            delattr(layer, "_source")

        report = project.audit_dependencies()

        assert report.cycles == (tuple(sorted((main.id, pre.id))),)

    def test_acyclic_sample_has_no_cycles(self) -> None:
        project = parse_aep(COMPLETE_SAMPLE).project
        assert project.audit_dependencies().cycles == ()


# ---- Solids and placeholders are never "missing files" -------------------------------


class TestInlineSources:
    def test_solid_and_placeholder_have_no_file_status(self, tmp_path: Path) -> None:
        project = py_aep.new().project
        comp = project.root_folder.add_comp("C", 64, 64, 1.0, 1.0, 24.0)
        comp.add_solid([0, 0.5, 1], "Blue")
        placeholder = project.import_placeholder(None, 64, 64, 24.0, 1.0)

        report = project.audit_dependencies()

        by_id = {record.id: record for record in report.footage}
        solid = next(item for item in project.footages if item.name == "Blue")
        assert by_id[solid.id].main.kind == "solid"
        assert by_id[solid.id].main.exists is None
        assert by_id[placeholder.id].main.kind == "placeholder"
        assert by_id[placeholder.id].main.exists is None
        assert report.missing_paths == ()
        assert report.duplicate_sources == ()
        assert placeholder.id in report.unused_footage
        assert solid.id not in report.unused_footage


# ---- Stability and read-only behavior ------------------------------------------------


class TestStabilityAndReadOnly:
    def test_repeated_audits_are_identical(self) -> None:
        project = parse_aep(ITEM_SAMPLE).project
        first = project.audit_dependencies()
        second = project.audit_dependencies()
        assert first == second

        project2 = parse_aep(COMPLETE_SAMPLE).project
        assert project2.audit_dependencies() == project2.audit_dependencies()

    def test_collections_are_sorted(self) -> None:
        report = parse_aep(COMPLETE_SAMPLE).project.audit_dependencies()

        assert [record.id for record in report.footage] == sorted(
            record.id for record in report.footage
        )
        assert [record.id for record in report.compositions] == sorted(
            record.id for record in report.compositions
        )
        assert [record.id for record in report.folders] == sorted(
            record.id for record in report.folders
        )
        assert list(report.unused_footage) == sorted(report.unused_footage)
        assert list(report.unused_compositions) == sorted(report.unused_compositions)
        for record in report.footage:
            assert list(record.used_by_compositions) == sorted(
                record.used_by_compositions
            )

    def test_audit_does_not_modify_the_project(self) -> None:
        project = parse_aep(ITEM_SAMPLE).project
        before = project_bytes(project)
        assert project._used_in_linked is False

        project.audit_dependencies()
        project.audit_dependencies(workdir=Path.cwd())

        assert project._used_in_linked is False
        assert project_bytes(project) == before

    def test_audit_unchanged_after_used_in_access(self) -> None:
        project = parse_aep(ITEM_SAMPLE).project
        first = project.audit_dependencies()
        # used_in populates the mutable back-reference cache; the audit must
        # neither depend on it nor alter its result because of it.
        for footage in project.footages:
            footage.used_in  # noqa: B018
        assert project._used_in_linked is True
        assert project.audit_dependencies() == first
