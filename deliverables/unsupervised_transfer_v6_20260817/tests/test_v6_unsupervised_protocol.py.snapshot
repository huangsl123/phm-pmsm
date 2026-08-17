import inspect

from scripts.exp3_v6_unsupervised_da import adapt_unlabeled


def test_v6_adaptation_interface_cannot_receive_target_labels():
    parameters = inspect.signature(adapt_unlabeled).parameters
    assert "target_X" in parameters
    assert "target" not in parameters
    source = inspect.getsource(adapt_unlabeled)
    assert "target_data" not in source
    assert 'target["y_' not in source
