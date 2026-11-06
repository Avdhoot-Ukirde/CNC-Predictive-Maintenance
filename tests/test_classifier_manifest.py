from pathlib import Path

from scripts.train_bearing_classifier import FILE_LABELS, CLASS_NAMES


def test_manifest_has_four_classes_and_four_recordings_each():
    assert set(FILE_LABELS.values()) == set(CLASS_NAMES)
    for label in CLASS_NAMES:
        assert sum(value == label for value in FILE_LABELS.values()) == 4


def test_three_hp_fault_files_match_normal_baseline():
    assert FILE_LABELS["100.mat"] == "normal"
    assert FILE_LABELS["108.mat"] == "inner_race"
    assert FILE_LABELS["121.mat"] == "ball"
    assert FILE_LABELS["133.mat"] == "outer_race"


def test_manifest_files_are_mat_files():
    assert all(Path(name).suffix == ".mat" for name in FILE_LABELS)
