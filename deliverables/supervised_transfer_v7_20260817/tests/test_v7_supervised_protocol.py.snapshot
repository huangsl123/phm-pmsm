import inspect

from scripts.exp3_v7_supervised_strategies import adapt, configs, empirical_fisher


def test_v7_contains_four_separate_strategy_families():
    methods = [config["method"] for config in configs()]
    assert set(methods) == {"lwf", "distill", "ewc", "replay"}
    assert all(methods.count(method) == 3 for method in set(methods))


def test_v7_is_explicitly_supervised_target_adaptation():
    source = inspect.getsource(adapt)
    assert 'target["y_train"]' in source
    assert "target_ce = criterion(target_logits, ty)" in source
    assert 'config["method"] == "lwf"' in source
    assert 'config["method"] == "distill"' in source
    assert 'config["method"] == "ewc"' in source


def test_v7_ewc_fisher_uses_source_training_examples_only():
    source = inspect.getsource(empirical_fisher)
    assert 'source["X_train"]' in source
    assert 'source["y_train"]' in source
    assert "batch=1" in source
    assert "X_val" not in source
    assert "y_val" not in source


def test_v7_lwf_and_distillation_are_not_replay_aliases():
    source = inspect.getsource(adapt)
    lwf_block = source.split('if config["method"] == "lwf":', 2)[2].split(
        'elif config["method"] == "distill":', 1
    )[0]
    distill_block = source.split('elif config["method"] == "distill":', 1)[1].split(
        'elif config["method"] == "ewc":', 1
    )[0]
    ewc_and_replay = source.split('elif config["method"] == "ewc":', 1)[1]
    replay_block = ewc_and_replay.split("else:", 1)[1].split("loss.backward()", 1)[0]
    assert "teacher(tx)" in lwf_block
    assert "criterion(model(sx), sy)" not in lwf_block
    assert "teacher(sx)" in distill_block
    assert "criterion(model(sx), sy)" not in distill_block
    assert "criterion(model(sx), sy)" in replay_block
