import inspect

import numpy as np

from scripts.exp3_v8_semisupervised import (
    adapt_semisupervised,
    configurations,
    stratified_target_partition,
    x_loader,
)


def test_unlabeled_loader_has_no_label_argument_or_tensor():
    assert list(inspect.signature(x_loader).parameters) == ["X", "batch", "shuffle"]
    loader = x_loader(np.zeros((8, 3), dtype=np.float32), batch=4, shuffle=False)
    batch = next(iter(loader))
    assert len(batch) == 1


def test_partition_is_disjoint_complete_and_class_balanced():
    X = np.arange(150 * 2, dtype=np.float32).reshape(150, 2)
    y = np.repeat(np.arange(15), 10).astype(np.int64)
    result = stratified_target_partition(X, y, .2, 42)
    labeled = set(result["labeled_indices"].tolist())
    unlabeled = set(result["unlabeled_indices"].tolist())
    assert not labeled & unlabeled
    assert labeled | unlabeled == set(range(150))
    assert set(result["labels_per_class"].values()) == {2}


def test_adaptation_cannot_receive_hidden_unlabeled_labels():
    parameters = inspect.signature(adapt_semisupervised).parameters
    assert "target_unlabeled_X" in parameters
    assert "target_unlabeled_y" not in parameters
    source = inspect.getsource(adapt_semisupervised)
    assert "target_unlabeled_y" not in source
    assert "criterion(model(lx), ly)" in source


def test_v8_compares_baseline_and_fixmatch_at_five_label_budgets():
    configs = configurations()
    assert {row["labeled_fraction"] for row in configs} == {.05, .10, .20, .30, .40}
    for fraction in (.05, .10, .20, .30, .40):
        selected = [row for row in configs if row["labeled_fraction"] == fraction]
        assert [row["method"] for row in selected].count("labeled_only") == 1
        assert [row["method"] for row in selected].count("fixmatch") == 2
