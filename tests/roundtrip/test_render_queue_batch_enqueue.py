"""Project.enqueue_compositions(): transactional batch render-queue enqueue.

The batch entry point must validate every input before touching the queue
and either add the whole batch or leave no trace. These tests run without
an After Effects preferences directory (templates are optional); the few
template-specific cases are gated on AE_PREFS_DIR like the rest of the
render-queue suite.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from py_aep import parse as parse_aep
from py_aep.binary.render_chunks import (
    ROUT_ITEMS_PER_RQ_ITEM,
    OutputModuleSettingsItem,
    RenderSettingsItem,
    RouuChunk,
    TiffRoptChunk,
)
from py_aep.enums import RQItemStatus
from py_aep.models.items.composition import CompItem

if TYPE_CHECKING:
    from py_aep.models.project import Project

RQ_DIR = Path(__file__).parent.parent.parent / "samples" / "models" / "renderqueue"
EMPTY_SAMPLE = RQ_DIR / "empty.aep"
TWO_COMPS_SAMPLE = RQ_DIR / "numItems_2.aep"
AE_PREFS_DIR = os.getenv("AE_PREFS_DIR")


def _project(sample: Path = EMPTY_SAMPLE, **kwargs: Any) -> Project:
    return parse_aep(sample, **kwargs).project


def _serialized(project: Project) -> bytes:
    """The bytes `save()` would write for `project` right now."""
    import io

    from py_aep.binary.chunk import write_aep

    buf = io.BytesIO()
    write_aep(buf, project._rifx, project._xmp)
    return buf.getvalue()


def _assert_rout_consistent(project: Project) -> None:
    rq = project.render_queue
    assert rq._rout.count == len(rq.items) * ROUT_ITEMS_PER_RQ_ITEM
    assert rq._rs_lhd3.count == len(rq.items)
    assert rq._arsi.queue_nonempty == (1 if rq.items else 0)


# ---------------------------------------------------------------------------
# Empty / single / multiple
# ---------------------------------------------------------------------------


class TestEmptyBatch:
    def test_empty_batch_returns_empty_list(self) -> None:
        project = _project()
        assert project.enqueue_compositions([], []) == []

    def test_empty_batch_does_not_touch_an_empty_queue(self) -> None:
        project = _project()
        rq = project.render_queue
        assert rq.num_items == 0

        project.enqueue_compositions([], [])

        assert rq.num_items == 0
        assert rq._arsi.queue_nonempty == 0

    def test_empty_batch_leaves_existing_queue_intact(self) -> None:
        project = _project(TWO_COMPS_SAMPLE)
        before = project.render_queue.items

        result = project.enqueue_compositions([], [])

        assert result == []
        after = project.render_queue.items
        assert after == before
        assert [item.comp.name for item in after] == ["Comp1", "Comp2"]

    def test_empty_batch_is_byte_identical(self, tmp_path: Path) -> None:
        project = _project()
        before = _serialized(project)

        project.enqueue_compositions([], [])

        assert _serialized(project) == before

    def test_empty_batch_with_paths_still_mismatches(self) -> None:
        project = _project()
        with pytest.raises(ValueError, match="one output path per composition"):
            project.enqueue_compositions([], ["x.mov"])


class TestSingleComposition:
    def test_single_comp_creates_one_queued_item(self, tmp_path: Path) -> None:
        project = _project()
        comp = project.compositions[0]
        target = tmp_path / "render" / "comp.mov"

        result = project.enqueue_compositions([comp], [target])

        assert len(result) == 1
        rqi = result[0]
        assert rqi.comp is comp
        assert rqi is project.render_queue.items[0]
        assert rqi.num_output_modules == 1
        # Setting the output file takes the item out of "Needs Output".
        assert rqi.status == RQItemStatus.QUEUED
        assert rqi.render is True
        om = rqi.output_modules[0]
        assert om.file == os.fspath(target)
        assert om.file_template == os.fspath(target)

    def test_accepts_pathlib_and_string_paths(self, tmp_path: Path) -> None:
        project = _project()
        comp = project.compositions[0]

        rqi = project.enqueue_compositions([comp], [tmp_path / "out.mov"])[0]
        assert rqi.output_modules[0].file == str(tmp_path / "out.mov")

    def test_single_comp_roundtrip(self, tmp_path: Path) -> None:
        project = _project()
        comp = project.compositions[0]
        target = tmp_path / "render" / "comp.mov"
        project.enqueue_compositions([comp], [target])

        out = tmp_path / "project.aep"
        project.save(out)
        reparsed = _project(out)
        items = reparsed.render_queue.items

        assert len(items) == 1
        rqi = items[0]
        assert rqi.comp.id == comp.id
        assert rqi.comp.name == "TestComp"
        assert rqi.num_output_modules == 1
        assert rqi.output_modules[0].file == os.fspath(target)
        # No template named: the add() defaults are kept.
        assert rqi.name == "Best Settings"
        assert rqi.output_modules[0].name == "TIFF Sequence with Alpha"
        _assert_rout_consistent(reparsed)


class TestMultipleCompositions:
    def test_items_appended_in_input_order_after_existing(self) -> None:
        project = _project(TWO_COMPS_SAMPLE)
        rq = project.render_queue
        comp1 = next(c for c in project.compositions if c.name == "Comp1")
        comp2 = next(c for c in project.compositions if c.name == "Comp2")

        # Reverse the input order on purpose: it must become queue order.
        result = project.enqueue_compositions(
            [comp2, comp1], ["/renders/c2.mov", "/renders/c1.mov"]
        )

        names = [item.comp.name for item in rq.items]
        assert names == ["Comp1", "Comp2", "Comp2", "Comp1"]
        assert result == rq.items[2:]
        assert result[0].comp is comp2
        assert result[1].comp is comp1
        assert result[0].output_modules[0].file == "/renders/c2.mov"
        assert result[1].output_modules[0].file == "/renders/c1.mov"

    def test_multiple_roundtrip_preserve_order_paths_and_links(
        self, tmp_path: Path
    ) -> None:
        project = _project(TWO_COMPS_SAMPLE)
        comp1 = next(c for c in project.compositions if c.name == "Comp1")
        comp2 = next(c for c in project.compositions if c.name == "Comp2")
        project.enqueue_compositions(
            [comp2, comp1],
            [tmp_path / "c2.mov", tmp_path / "nested" / "c1.mov"],
        )

        out = tmp_path / "project.aep"
        project.save(out)
        reparsed = _project(out)
        items = reparsed.render_queue.items

        assert [item.comp.name for item in items] == [
            "Comp1",
            "Comp2",
            "Comp2",
            "Comp1",
        ]
        assert [item.comp.id for item in items] == [
            comp1.id,
            comp2.id,
            comp2.id,
            comp1.id,
        ]
        assert Path(items[2].output_modules[0].file) == tmp_path / "c2.mov"
        assert Path(items[3].output_modules[0].file) == (tmp_path / "nested" / "c1.mov")
        # Every new item owns exactly one default output module.
        assert all(item.num_output_modules == 1 for item in items)
        _assert_rout_consistent(reparsed)

    def test_re_enqueueing_an_already_queued_comp_is_allowed(
        self, tmp_path: Path
    ) -> None:
        # AE itself allows the same comp to appear in the queue several
        # times; only within-batch duplicates are rejected.
        project = _project(TWO_COMPS_SAMPLE)
        comp1 = next(c for c in project.compositions if c.name == "Comp1")

        project.enqueue_compositions([comp1], [tmp_path / "again.mov"])

        names = [item.comp.name for item in project.render_queue.items]
        assert names == ["Comp1", "Comp2", "Comp1"]


# ---------------------------------------------------------------------------
# Precheck failures: every case leaves queue and comp models untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "comps",
    [
        pytest.param(["not a comp"], id="wrong_element_type"),
        pytest.param([1], id="integer_element"),
    ],
)
def test_non_comp_entries_raise_type_error(comps: list[Any]) -> None:
    project = _project()
    with pytest.raises(TypeError, match="must be a CompItem"):
        project.enqueue_compositions(comps, ["x.mov"])
    assert project.render_queue.num_items == 0


def test_composition_from_another_project_is_rejected() -> None:
    project = _project(EMPTY_SAMPLE)
    other = _project(EMPTY_SAMPLE)
    foreign = other.compositions[0]
    # Same id as this project's comp, but a different object.
    assert foreign.id == project.compositions[0].id

    with pytest.raises(ValueError, match="not an item of this project"):
        project.enqueue_compositions([foreign], ["/renders/x.mov"])
    assert project.render_queue.num_items == 0


def test_duplicate_composition_is_rejected() -> None:
    project = _project(TWO_COMPS_SAMPLE)
    comp1 = next(c for c in project.compositions if c.name == "Comp1")

    with pytest.raises(ValueError, match="listed more than once"):
        project.enqueue_compositions(
            [comp1, comp1], ["/renders/a.mov", "/renders/b.mov"]
        )
    assert project.render_queue.num_items == 2


def test_path_count_mismatch_is_rejected(tmp_path: Path) -> None:
    project = _project(TWO_COMPS_SAMPLE)
    comp1 = next(c for c in project.compositions if c.name == "Comp1")
    comp2 = next(c for c in project.compositions if c.name == "Comp2")

    with pytest.raises(ValueError, match="one output path per composition"):
        project.enqueue_compositions([comp1, comp2], [tmp_path / "only.mov"])
    with pytest.raises(ValueError, match="one output path per composition"):
        project.enqueue_compositions([comp1], [])
    assert project.render_queue.num_items == 2


def test_non_path_entry_raises_type_error() -> None:
    project = _project()
    comp = project.compositions[0]
    with pytest.raises(TypeError, match="string or os.PathLike"):
        project.enqueue_compositions([comp], [123])
    assert project.render_queue.num_items == 0


def test_duplicate_target_is_rejected(tmp_path: Path) -> None:
    project = _project(TWO_COMPS_SAMPLE)
    comp1 = next(c for c in project.compositions if c.name == "Comp1")
    comp2 = next(c for c in project.compositions if c.name == "Comp2")

    target = tmp_path / "out" / "same.mov"
    with pytest.raises(ValueError, match="same target"):
        project.enqueue_compositions([comp1, comp2], [target, target])

    # Two spellings of the same file are still duplicates: a redundant
    # double separator and Windows-style separators normalize together.
    alt1 = f"{tmp_path}/out//same.mov"
    alt2 = str(tmp_path).replace("/", os.sep) + os.sep + "out" + os.sep + ("same.mov")
    with pytest.raises(ValueError, match="same target"):
        project.enqueue_compositions([comp1, comp2], [alt1, alt2])
    assert project.render_queue.num_items == 2


def test_overwrite_must_be_bool() -> None:
    project = _project()
    comp = project.compositions[0]
    with pytest.raises(TypeError, match="expected a bool"):
        project.enqueue_compositions([comp], ["/renders/x.mov"], overwrite=1)
    assert project.render_queue.num_items == 0


class TestExistingTargets:
    def test_existing_target_without_policy_is_rejected(self, tmp_path: Path) -> None:
        project = _project()
        comp = project.compositions[0]
        target = tmp_path / "existing.mov"
        target.write_text("rendered earlier")

        with pytest.raises(ValueError, match="conflict policy must be explicit"):
            project.enqueue_compositions([comp], [target])
        assert project.render_queue.num_items == 0

    def test_existing_target_refused_when_overwrite_false(self, tmp_path: Path) -> None:
        project = _project()
        comp = project.compositions[0]
        target = tmp_path / "existing.mov"
        target.write_text("rendered earlier")

        with pytest.raises(FileExistsError, match="already exists"):
            project.enqueue_compositions([comp], [target], overwrite=False)
        assert project.render_queue.num_items == 0

    def test_existing_target_allowed_when_overwrite_true(self, tmp_path: Path) -> None:
        project = _project()
        comp = project.compositions[0]
        target = tmp_path / "existing.mov"
        target.write_text("rendered earlier")

        result = project.enqueue_compositions([comp], [target], overwrite=True)

        assert len(result) == 1
        assert result[0].output_modules[0].file == str(target)
        # The pre-existing file is not touched by enqueueing.
        assert target.read_text() == "rendered earlier"

    def test_nonexistent_target_works_with_every_policy(self, tmp_path: Path) -> None:
        for policy in (None, True, False):
            project = _project()
            comp = project.compositions[0]
            target = tmp_path / f"fresh_{policy}.mov"
            assert not target.exists()

            result = project.enqueue_compositions([comp], [target], overwrite=policy)
            assert len(result) == 1


class TestUnknownTemplates:
    # Without preferences a template name cannot resolve at all: this is a
    # precheck failure in CI (no AE_PREFS_DIR), and still leaves no trace.
    def test_unknown_render_template_without_preferences(self) -> None:
        project = _project()
        comp = project.compositions[0]
        with pytest.raises(ValueError):
            project.enqueue_compositions(
                [comp], ["/renders/x.mov"], render_template="Best Settings"
            )
        assert project.render_queue.num_items == 0

    def test_unknown_output_template_without_preferences(self) -> None:
        project = _project()
        comp = project.compositions[0]
        with pytest.raises(ValueError):
            project.enqueue_compositions(
                [comp], ["/renders/x.mov"], output_template="Lossless"
            )
        assert project.render_queue.num_items == 0

    @pytest.mark.skipif(not AE_PREFS_DIR, reason="AE_PREFS_DIR env var not set")
    def test_unknown_template_names_are_rejected(self) -> None:
        project = _project(EMPTY_SAMPLE, ae_preferences_dir=AE_PREFS_DIR)
        comp = project.compositions[0]
        with pytest.raises(ValueError, match="not found"):
            project.enqueue_compositions(
                [comp], ["/renders/x.mov"], render_template="No Such RS ###"
            )
        with pytest.raises(ValueError, match="not found"):
            project.enqueue_compositions(
                [comp], ["/renders/x.mov"], output_template="No Such OM ###"
            )
        assert project.render_queue.num_items == 0

    @pytest.mark.skipif(not AE_PREFS_DIR, reason="AE_PREFS_DIR env var not set")
    def test_named_templates_apply_and_roundtrip(self, tmp_path: Path) -> None:
        project = _project(EMPTY_SAMPLE, ae_preferences_dir=AE_PREFS_DIR)
        # Pick template names that really exist in these preferences.
        probe = _project(TWO_COMPS_SAMPLE, ae_preferences_dir=AE_PREFS_DIR)
        render_name = probe.render_queue.items[0].templates[0]
        output_name = probe.render_queue.items[0].output_modules[0].templates[0]
        comp = project.compositions[0]

        result = project.enqueue_compositions(
            [comp],
            [tmp_path / "templated.mov"],
            render_template=render_name,
            output_template=output_name,
        )
        assert result[0].name == render_name
        assert result[0].output_modules[0].name == output_name

        out = tmp_path / "project.aep"
        project.save(out)
        reparsed = _project(out, ae_preferences_dir=AE_PREFS_DIR)
        rqi = reparsed.render_queue.items[0]
        assert rqi.name == render_name
        assert rqi.output_modules[0].name == output_name
        assert rqi.output_modules[0].file == str(tmp_path / "templated.mov")


def _write_synthetic_prefs(prefs_dir: Path) -> None:
    """Write minimal AE render/output preferences listing a couple of
    named templates, so template validation and application run in CI
    without a real After Effects install (same layout the parsers read;
    see tests/unit/test_models_renderqueue.py)."""
    render_raw = b"\x00" * 32
    for rs_name in ("Best Settings", "Draft Settings"):
        render_raw += RenderSettingsItem(template_name=rs_name).tobytes()[:2182]
    (prefs_dir / "Adobe After Effects 26.0 Prefs-indep-render.txt").write_text(
        '["Render Settings Preference Section"]\n'
        '\t"Render Settings List" = ' + render_raw.hex() + "\n"
        '\t"Default RS Index" = "0"\n',
        encoding="utf-8",
    )

    om_names = ("TIFF Sequence with Alpha", "PNG Sequence")
    output_raw = b"\x00" * 32 + OutputModuleSettingsItem().tobytes() * len(om_names)
    rouu_hex = RouuChunk().tobytes().hex()
    ropt_hex = TiffRoptChunk().tobytes().hex()
    lines = [
        '["Output Module Preference Section"]',
        '\t"Output Module List v28" = ' + output_raw.hex(),
        '\t"Default OM Index" = "0"',
    ]
    for i, om_name in enumerate(om_names):
        lines.append(f'\t"Output Module Spec Strings Name #{i}" = "{om_name}"')
    lines.append('["Output File Info Preference Section v28"]')
    for i in range(len(om_names)):
        lines.append(f'\t"Output File Info ID # {i}" = {rouu_hex}')
    lines.append('["Output File Options Preference Section v28"]')
    for i in range(len(om_names)):
        lines.append(f'\t"Output File Options ID # {i}" = {ropt_hex}')
    (prefs_dir / "Adobe After Effects 26.0 Prefs-indep-output.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


class TestTemplatesWithSyntheticPreferences:
    """Template prechecks/application against a synthetic prefs dir, so
    they run in CI (the AE_PREFS_DIR-gated cases above cover real AE
    preferences)."""

    def test_unknown_template_names_are_rejected(self, tmp_path: Path) -> None:
        prefs_dir = tmp_path / "prefs"
        prefs_dir.mkdir()
        _write_synthetic_prefs(prefs_dir)
        project = _project(EMPTY_SAMPLE, ae_preferences_dir=prefs_dir)
        comp = project.compositions[0]

        with pytest.raises(ValueError, match="Render settings template"):
            project.enqueue_compositions(
                [comp], [tmp_path / "x.mov"], render_template="No Such RS"
            )
        with pytest.raises(ValueError, match="Output module template"):
            project.enqueue_compositions(
                [comp], [tmp_path / "x.mov"], output_template="No Such OM"
            )
        assert project.render_queue.num_items == 0

    def test_named_templates_apply_to_every_item_and_roundtrip(
        self, tmp_path: Path
    ) -> None:
        prefs_dir = tmp_path / "prefs"
        prefs_dir.mkdir()
        _write_synthetic_prefs(prefs_dir)
        project = _project(EMPTY_SAMPLE, ae_preferences_dir=prefs_dir)
        project.root_folder.add_comp("Second", 64, 36, 1.0, 1.0, 24)
        comps = project.compositions

        result = project.enqueue_compositions(
            comps,
            [tmp_path / "a.mov", tmp_path / "b.mov"],
            render_template="Draft Settings",
            output_template="PNG Sequence",
        )

        assert [rqi.name for rqi in result] == [
            "Draft Settings",
            "Draft Settings",
        ]
        assert [rqi.output_modules[0].name for rqi in result] == [
            "PNG Sequence",
            "PNG Sequence",
        ]

        out = tmp_path / "project.aep"
        project.save(out)
        reparsed = _project(out, ae_preferences_dir=prefs_dir)
        items = reparsed.render_queue.items
        assert len(items) == 2
        assert [rqi.name for rqi in items] == ["Draft Settings", "Draft Settings"]
        assert [rqi.output_modules[0].name for rqi in items] == [
            "PNG Sequence",
            "PNG Sequence",
        ]
        assert [rqi.output_modules[0].file for rqi in items] == [
            str(tmp_path / "a.mov"),
            str(tmp_path / "b.mov"),
        ]
        assert [rqi.comp.name for rqi in items] == [c.name for c in comps]
        _assert_rout_consistent(reparsed)

    def test_omitting_templates_uses_existing_defaults(self, tmp_path: Path) -> None:
        prefs_dir = tmp_path / "prefs"
        prefs_dir.mkdir()
        _write_synthetic_prefs(prefs_dir)
        project = _project(EMPTY_SAMPLE, ae_preferences_dir=prefs_dir)
        comp = project.compositions[0]

        result = project.enqueue_compositions([comp], [tmp_path / "a.mov"])

        # The add() defaults: default render template from prefs, fresh
        # TIFF output module.
        assert result[0].name == "Best Settings"
        assert result[0].output_modules[0].name == "TIFF Sequence with Alpha"


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def _failed_batch_kwargs(
    project: Project, other: Project, tmp_path: Path
) -> dict[str, dict[str, Any]]:
    """Kwargs that each fail the precheck for a two-comp project."""
    comp1 = next(c for c in project.compositions if c.name == "Comp1")
    comp2 = next(c for c in project.compositions if c.name == "Comp2")
    existing = tmp_path / "existing.mov"
    existing.write_text("x")
    return {
        "duplicate comp": {
            "compositions": [comp1, comp1],
            "output_files": [tmp_path / "a", tmp_path / "b"],
        },
        "foreign comp": {
            "compositions": [comp1, other.compositions[0]],
            "output_files": [tmp_path / "a", tmp_path / "b"],
        },
        "count mismatch": {
            "compositions": [comp1, comp2],
            "output_files": [tmp_path / "a"],
        },
        "duplicate target": {
            "compositions": [comp1, comp2],
            "output_files": [tmp_path / "same", tmp_path / "same"],
        },
        "existing target, no policy": {
            "compositions": [comp1, comp2],
            "output_files": [existing, tmp_path / "b"],
        },
        "existing target, refused": {
            "compositions": [comp1, comp2],
            "output_files": [existing, tmp_path / "b"],
            "overwrite": False,
        },
        "unknown render template": {
            "compositions": [comp1, comp2],
            "output_files": [tmp_path / "a", tmp_path / "b"],
            "render_template": "No Such RS ###",
        },
        "unknown output template": {
            "compositions": [comp1, comp2],
            "output_files": [tmp_path / "a", tmp_path / "b"],
            "output_template": "No Such OM ###",
        },
        "bad overwrite type": {
            "compositions": [comp1, comp2],
            "output_files": [tmp_path / "a", tmp_path / "b"],
            "overwrite": "yes",
        },
    }


@pytest.mark.parametrize(
    "case",
    [
        "duplicate comp",
        "foreign comp",
        "count mismatch",
        "duplicate target",
        "existing target, no policy",
        "existing target, refused",
        "unknown render template",
        "unknown output template",
        "bad overwrite type",
    ],
)
def test_precheck_failure_has_zero_side_effects(tmp_path: Path, case: str) -> None:
    # One untouched parse serialized first; the mutated-then-rejected parse
    # must serialize byte-for-byte identically.
    _project(TWO_COMPS_SAMPLE).save(tmp_path / "untouched.aep")
    project = _project(TWO_COMPS_SAMPLE)
    other = _project(EMPTY_SAMPLE)
    comps_before = [c.name for c in project.compositions]
    queue_before = [
        (i.comp.id, i.output_modules[0].file_template)
        for i in project.render_queue.items
    ]

    kwargs = _failed_batch_kwargs(project, other, tmp_path)[case]
    with pytest.raises((ValueError, TypeError, FileExistsError)):
        project.enqueue_compositions(**kwargs)

    # Model state is unchanged...
    assert project.render_queue.num_items == 2
    assert [c.name for c in project.compositions] == comps_before
    assert [
        (i.comp.id, i.output_modules[0].file_template)
        for i in project.render_queue.items
    ] == queue_before
    _assert_rout_consistent(project)

    # ...and the rejected batch leaves no trace in the saved file.
    project.save(tmp_path / "rejected.aep")
    assert (tmp_path / "rejected.aep").read_bytes() == (
        tmp_path / "untouched.aep"
    ).read_bytes()


def test_failure_during_apply_rolls_back_the_half_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prechecks pass, then the second add blows up unexpectedly: the item
    # created for the first comp must be rolled back too - never a half
    # batch.
    project = _project(TWO_COMPS_SAMPLE)
    project.root_folder.add_comp("New1", 64, 36, 1.0, 1.0, 24)
    project.root_folder.add_comp("New2", 64, 36, 1.0, 1.0, 24)
    new1 = next(c for c in project.compositions if c.name == "New1")
    new2 = next(c for c in project.compositions if c.name == "New2")
    rq = project.render_queue
    original_add = rq.add
    calls = 0

    def flaky_add(comp: CompItem):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated mid-batch failure")
        return original_add(comp)

    monkeypatch.setattr(rq, "add", flaky_add)

    with pytest.raises(RuntimeError, match="simulated"):
        project.enqueue_compositions(
            [new1, new2], [tmp_path / "n1.mov", tmp_path / "n2.mov"]
        )

    # Both new items gone; the two original items remain untouched.
    assert [i.comp.name for i in rq.items] == ["Comp1", "Comp2"]
    _assert_rout_consistent(project)
    project.save(tmp_path / "rolled_back.aep")
    reparsed = _project(tmp_path / "rolled_back.aep")
    assert [i.comp.name for i in reparsed.render_queue.items] == [
        "Comp1",
        "Comp2",
    ]
    _assert_rout_consistent(reparsed)
