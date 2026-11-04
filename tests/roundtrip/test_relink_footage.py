"""Tests for `Project.relink_footage`: batch relinking of footage whose
stored source path was lost in a project/machine migration.

Every test builds a fresh project from sample assets copied into an "old"
media tree, migrates the media into search roots laid out for the
scenario, and asserts what the batch run did (and did not) do.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from helpers import project_bytes

from py_aep import (
    RELINK_AMBIGUOUS,
    RELINK_FAILED,
    RELINK_NOT_FOUND,
    RELINK_PLANNED,
    RELINK_REPLACED,
    RELINK_UNSUPPORTED_FORMAT,
    FieldSeparationType,
    FileSource,
    FootageItem,
    ImportOptions,
)
from py_aep import new as new_app
from py_aep import parse as parse_aep

ASSETS = Path(__file__).parent.parent.parent / "samples" / "assets"
FOOTAGE_MISSING_AEP = (
    ASSETS.parent / "models" / "footage" / "footage_missing.aep"
)


# ---- workspace builders ----------------------------------------------------


def _new_project():
    return new_app(platform="macos").project


def _place(base: Path, rel: str, asset: str) -> Path:
    """Copy a sample asset to `<base>/<rel>` and return its new path."""
    target = base / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ASSETS / asset, target)
    return target


def _import(project, path: Path) -> FootageItem:
    return project.import_file(ImportOptions(path))


def _unknown_footage(project, path: Path) -> FootageItem:
    """Attach a FileSource footage for an extension py_aep cannot import,
    as a project parsed from another machine might hold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a media file")
    source = FileSource._new(
        path,
        source_format="ZZZZ",
        width=12,
        height=34,
        duration=0,
        frame_rate=0,
        windows=project._windows,
    )
    item = FootageItem._new(
        source, project=project, parent_folder=project.root_folder
    )
    container = project.root_folder._children_container
    container.append(item._item_list)
    container.extend(item._view_data)
    project.items[item.id] = item
    project.root_folder.items.append(item)
    return item


def _statuses(result) -> list[tuple[str, str]]:
    return [(entry.role, entry.status) for entry in [*result.completed, *result.skipped]]


# ---- candidate matching ----------------------------------------------------


class TestCandidateMatching:
    def test_unique_match_relinks(self, tmp_path: Path) -> None:
        old_base = tmp_path / "old" / "media"
        old_file = _place(old_base, "alpha.png", "image_with_alpha.png")
        project = _new_project()
        item = _import(project, old_file)

        root = tmp_path / "root"
        new_file = root / "alpha.png"
        new_file.parent.mkdir(parents=True)
        shutil.copy(old_file, new_file)
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)

        assert result.completed and not result.incomplete
        entry = result.completed[0]
        assert (entry.role, entry.status) == ("main", RELINK_REPLACED)
        assert entry.new_path == str(new_file)
        assert item.main_source.file == str(new_file)
        assert item.footage_missing is False
        assert (item.width, item.height) == (640, 346)

    def test_relative_subdir_match_prefers_folder_tail(
        self, tmp_path: Path
    ) -> None:
        old_file = _place(tmp_path / "old" / "media", "alpha.png",
                          "image_with_alpha.png")
        project = _new_project()
        _import(project, old_file)

        root = tmp_path / "root"
        in_subdir = _place(root / "keep" / "media", "alpha.png",
                           "image_with_alpha.png")
        _place(root / "other", "alpha.png", "image_with_alpha.png")
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)

        assert result.completed and not result.skipped
        assert Path(result.completed[0].new_path) == in_subdir

    def test_same_name_conflict_is_skipped(self, tmp_path: Path) -> None:
        old_file = _place(tmp_path / "old", "alpha.png",
                          "image_with_alpha.png")
        project = _new_project()
        item = _import(project, old_file)

        root = tmp_path / "root"
        first = _place(root / "d1", "alpha.png", "image_with_alpha.png")
        second = _place(root / "d2", "alpha.png", "image_with_alpha.png")
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)

        assert not result.completed
        entry = result.skipped[0]
        assert entry.status == RELINK_AMBIGUOUS
        assert set(entry.candidates) == {str(first), str(second)}
        # Nothing was guessed: the stored, missing path is untouched.
        assert item.main_source.file == str(old_file)

    def test_no_candidate_recorded_as_not_found(self, tmp_path: Path) -> None:
        old_file = _place(tmp_path / "old" / "media", "alpha.png",
                          "image_with_alpha.png")
        project = _new_project()
        _import(project, old_file)
        shutil.rmtree(tmp_path / "old")

        root = tmp_path / "root"
        root.mkdir()
        _place(root, "unrelated.png", "8bits_compressed.png")

        result = project.relink_footage(root)

        assert not result.completed
        assert result.skipped[0].status == RELINK_NOT_FOUND

    def test_searches_multiple_roots(self, tmp_path: Path) -> None:
        old_file = _place(tmp_path / "old" / "media", "alpha.png",
                          "image_with_alpha.png")
        project = _new_project()
        _import(project, old_file)
        shutil.rmtree(tmp_path / "old")

        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        full_root = tmp_path / "media_root"
        new_file = _place(full_root / "media", "alpha.png",
                          "image_with_alpha.png")

        result = project.relink_footage([empty_root, full_root])

        assert result.completed[0].new_path == str(new_file)

    def test_missing_search_root_raises(self, tmp_path: Path) -> None:
        project = _new_project()
        with pytest.raises(ValueError, match="search root"):
            project.relink_footage(tmp_path / "does_not_exist")

    def test_windows_style_path_from_sample_project(self, tmp_path: Path) -> None:
        # The corpus' footage_missing.aep stores a Windows path
        # (...\Downloads\sample_motionblur_transparency.exr) opened on
        # macOS: the separators must be split and the Downloads/ tail
        # matched.
        project = parse_aep(FOOTAGE_MISSING_AEP).project
        item = project.footages[0]
        assert item.footage_missing is True

        root = tmp_path / "root"
        candidate = root / "Downloads" / "sample_motionblur_transparency.exr"
        candidate.parent.mkdir(parents=True)
        shutil.copy(ASSETS / "new_exr.0002.exr", candidate)

        result = project.relink_footage(root)

        assert not result.incomplete
        assert item.main_source.file == str(candidate)
        assert item.footage_missing is False

    def test_resolved_path_is_not_a_target(self, tmp_path: Path) -> None:
        # The media never moved: its stored path still resolves, so the run
        # has nothing to do with it.
        old_file = _place(tmp_path / "media", "alpha.png",
                          "image_with_alpha.png")
        project = _new_project()
        _import(project, old_file)

        result = project.relink_footage(tmp_path)

        assert len(result) == 0
        assert bool(result) is True


# ---- main footage and proxies ----------------------------------------------


class TestMainAndProxy:
    def _project_with_proxy(self, tmp_path: Path, *, in_use: bool):
        old_main = _place(tmp_path / "old" / "media", "alpha.png",
                          "image_with_alpha.png")
        old_proxy = _place(tmp_path / "old" / "proxies", "alpha_proxy.png",
                           "8bits_compressed.png")
        project = _new_project()
        item = _import(project, old_main)
        item.set_proxy(old_proxy)
        item.use_proxy = in_use
        return project, item, old_main, old_proxy

    def test_main_and_proxy_both_relinked(self, tmp_path: Path) -> None:
        project, item, old_main, old_proxy = self._project_with_proxy(
            tmp_path, in_use=True
        )
        root = tmp_path / "root"
        new_main = _place(root / "media", "alpha.png",
                          "image_with_alpha.png")
        new_proxy = _place(root / "proxies", "alpha_proxy.png",
                           "8bits_compressed.png")
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)

        assert not result.incomplete
        assert _statuses(result) == [
            ("main", RELINK_REPLACED),
            ("proxy", RELINK_REPLACED),
        ]
        assert item.main_source.file == str(new_main)
        assert item.proxy_source is not None
        assert item.proxy_source.file == str(new_proxy)
        # The proxy assignment and its enabled state survive.
        assert item.use_proxy is True
        assert item._idta._proxy_active == 1

    def test_disabled_proxy_keeps_disabled_after_relink(
        self, tmp_path: Path
    ) -> None:
        project, item, _old_main, _old_proxy = self._project_with_proxy(
            tmp_path, in_use=False
        )
        root = tmp_path / "root"
        _place(root / "media", "alpha.png", "image_with_alpha.png")
        _place(root / "proxies", "alpha_proxy.png", "8bits_compressed.png")
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)

        assert not result.incomplete
        assert item.use_proxy is False
        assert item._idta._proxy_active == 1
        assert item.proxy_source is not None
        assert Path(item.proxy_source.file) == root / "proxies/alpha_proxy.png"

    def test_include_proxies_false_skips_proxy(self, tmp_path: Path) -> None:
        project, item, _old_main, _old_proxy = self._project_with_proxy(
            tmp_path, in_use=True
        )
        root = tmp_path / "root"
        new_main = _place(root / "media", "alpha.png",
                          "image_with_alpha.png")
        _place(root / "proxies", "alpha_proxy.png", "8bits_compressed.png")
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root, include_proxies=False)

        assert len(result) == 1
        assert result.completed[0].role == "main"
        assert item.main_source.file == str(new_main)
        assert "alpha_proxy" in item.proxy_source.file
        # The proxy was not considered: its old, missing path survives.
        assert not Path(item.proxy_source.file).exists()


# ---- dry run ----------------------------------------------------------------


class TestDryRun:
    def test_dry_run_plans_without_modifying(self, tmp_path: Path) -> None:
        old_file = _place(tmp_path / "old" / "media", "alpha.png",
                          "image_with_alpha.png")
        project = _new_project()
        item = _import(project, old_file)
        root = tmp_path / "root"
        new_file = _place(root / "media", "alpha.png",
                          "image_with_alpha.png")
        # A second footage with no candidate, to prove pre-flight reporting
        # is shown but not acted upon either.
        orphan = _place(tmp_path / "old", "orphan.png",
                        "8bits_compressed.png")
        _import(project, orphan)
        shutil.rmtree(tmp_path / "old")

        before = project_bytes(project)
        result = project.relink_footage(root, dry_run=True)
        after = project_bytes(project)

        assert after == before
        assert result.dry_run is True
        planned, skipped = result.completed[0], result.skipped[0]
        assert planned.status == RELINK_PLANNED
        assert planned.new_path == str(new_file)
        assert skipped.status == RELINK_NOT_FOUND
        # The model was not touched.
        assert item.main_source.file == str(old_file)

        # The same roots in execute mode then perform the planned change.
        execute = project.relink_footage(root)
        assert execute.completed[0].status == RELINK_REPLACED
        assert item.main_source.file == str(new_file)


# ---- pre-flight rejection and execution errors ------------------------------


class TestSkipAndFailure:
    def test_unknown_format_candidate_is_skipped(
        self, tmp_path: Path
    ) -> None:
        good_old = _place(tmp_path / "old" / "media", "alpha.png",
                          "image_with_alpha.png")
        project = _new_project()
        good_item = _import(project, good_old)
        weird_old = tmp_path / "old" / "media" / "mystery.xyz"
        weird_item = _unknown_footage(project, weird_old)

        root = tmp_path / "root"
        good_new = _place(root / "media", "alpha.png",
                          "image_with_alpha.png")
        weird_new = root / "media" / "mystery.xyz"
        weird_new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(weird_old, weird_new)
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)

        by_name = {
            Path(entry.old_path).name: entry for entry in result.incomplete
        }
        skipped = by_name["mystery.xyz"]
        assert skipped.status == RELINK_UNSUPPORTED_FORMAT
        assert "Unsupported footage format" in skipped.error
        assert weird_item.main_source.file == str(weird_old)
        # A rejected candidate does not block a confirmed sibling.
        assert good_item.main_source.file == str(good_new)

    def test_single_failure_does_not_block_other_confirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alpha_old = _place(tmp_path / "old" / "media", "alpha.png",
                           "image_with_alpha.png")
        bmp_old = _place(tmp_path / "old" / "media", "dup.bmp", "bmp.bmp")
        project = _new_project()
        alpha_item = _import(project, alpha_old)
        bmp_item = _import(project, bmp_old)

        root = tmp_path / "root"
        alpha_new = _place(root / "media", "alpha.png",
                           "image_with_alpha.png")
        bmp_new = _place(root / "media", "dup.bmp", "bmp.bmp")
        shutil.rmtree(tmp_path / "old")

        original = FileSource._from_file.__func__

        def failing_from_file(cls, file, **kwargs):
            if Path(str(file)).name == "alpha.png":
                raise RuntimeError("simulated media load failure")
            return original(cls, file, **kwargs)

        monkeypatch.setattr(
            FileSource, "_from_file", classmethod(failing_from_file)
        )

        result = project.relink_footage(root)

        assert len(result.failed) == 1
        failed = result.failed[0]
        assert failed.status == RELINK_FAILED
        assert failed.new_path == str(alpha_new)
        assert "RuntimeError" in failed.error
        assert "simulated media load failure" in failed.error
        # The error was reported, not swallowed by raising, and the other
        # confirmed replacement still went through.
        assert bmp_item.main_source.file == str(bmp_new)
        assert alpha_item.main_source.file == str(alpha_old)


# ---- preserved state and round trip ----------------------------------------


class TestPreservationAndRoundTrip:
    def test_interpretation_and_custom_name_preserved(
        self, tmp_path: Path
    ) -> None:
        old_file = _place(tmp_path / "old" / "media", "alpha.png",
                          "image_with_alpha.png")
        project = _new_project()
        item = _import(project, old_file)
        item.name = "My Custom Name"
        # Interpret Footage choices AE (and the relink) carry over. The
        # alpha mode is deliberately not among them: AE re-estimates it per
        # file, mirroring FootageItem.replace.
        item.main_source.field_separation_type = (
            FieldSeparationType.UPPER_FIELD_FIRST
        )
        item.main_source.high_quality_field_separation = True
        item.main_source.loop = 3
        item.pixel_aspect = 1.5

        root = tmp_path / "root"
        new_file = _place(root / "media", "alpha.png",
                          "image_with_alpha.png")
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)
        assert not result.incomplete

        source = item.main_source
        assert item.name == "My Custom Name"
        assert item._name_utf8.value == "My Custom Name"
        assert source.field_separation_type == (
            FieldSeparationType.UPPER_FIELD_FIRST
        )
        assert source.high_quality_field_separation is True
        assert source.loop == 3
        assert item.pixel_aspect == pytest.approx(1.5)
        assert source.file == str(new_file)

    def test_save_and_reparse(self, tmp_path: Path) -> None:
        old_main = _place(tmp_path / "old" / "media", "alpha.png",
                          "image_with_alpha.png")
        old_proxy = _place(tmp_path / "old" / "proxies", "alpha_proxy.png",
                           "8bits_compressed.png")
        project = _new_project()
        item = _import(project, old_main)
        item.set_proxy(old_proxy)
        item.name = "Relinked Me"

        root = tmp_path / "root"
        new_main = _place(root / "media", "alpha.png",
                          "image_with_alpha.png")
        new_proxy = _place(root / "proxies", "alpha_proxy.png",
                           "8bits_compressed.png")
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)
        assert not result.incomplete

        out = tmp_path / "relinked.aep"
        project.save(out)
        reparsed = parse_aep(out).project
        footage = next(f for f in reparsed.footages if f.name == "Relinked Me")

        assert footage.main_source.file == str(new_main)
        assert footage.proxy_source.file == str(new_proxy)
        assert footage.footage_missing is False
        assert footage.use_proxy is True

    def test_sequence_source_relinks_to_sequence_folder(
        self, tmp_path: Path
    ) -> None:
        old_dir = tmp_path / "old" / "seq"
        frames = [
            _place(old_dir, "dpx_seq.0001.dpx", "dpx_seq.0001.dpx"),
            _place(old_dir, "dpx_seq.0002.dpx", "dpx_seq.0002.dpx"),
            _place(old_dir, "dpx_seq.0003.dpx", "dpx_seq.0003.dpx"),
        ]
        project = _new_project()
        options = ImportOptions(frames[0])
        options.sequence = True
        item = project.import_file(options)
        assert item.main_source.target_is_folder is True

        root = tmp_path / "root"
        new_dir = root / "seq"
        new_dir.mkdir(parents=True)
        for frame in frames:
            shutil.copy(frame, new_dir / frame.name)
        shutil.rmtree(tmp_path / "old")

        result = project.relink_footage(root)

        assert not result.incomplete
        source = item.main_source
        assert source.target_is_folder is True
        # A numbered sequence stores its frame range rather than a StVc
        # name list; the folder and representative frame were repaired.
        assert (source._start_frame, source._end_frame) == (1, 3)
        assert Path(source.file).name == "dpx_seq.0001.dpx"
        assert Path(source.file).parent == new_dir
