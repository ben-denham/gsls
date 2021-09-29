from pyquantification.utils import (
    select_keys,
    prefix_keys,
    dict_first,
    normalise_dict,
    check_dict_almost_equal,
)


def test_select_keys() -> None:
    assert (select_keys({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, ['b', 'd'])
            == {'b': 2, 'd': 4})


def test_prefix_keys() -> None:
    assert prefix_keys({'a': 1, 'b': 2}, 'foo_') == {'foo_a': 1, 'foo_b': 2}


def test_dict_first() -> None:
    assert dict_first({'a': 1, 'b': 2}) == 1


def test_normalise_dict() -> None:
    assert (normalise_dict({'a': 1, 'b': 2, 'c': 3, 'd': 4})
            == {'a': 1/10, 'b': 2/10, 'c': 3/10, 'd': 4/10})


def test_check_dict_almost_equal() -> None:
    assert check_dict_almost_equal({}, {}) is True
    assert check_dict_almost_equal({'a': 1}, {'b': 1}) is False
    assert check_dict_almost_equal({'a': 1, 'b': 2}, {'a': 1}) is False
    assert check_dict_almost_equal({'a': 1}, {'a': 1, 'b': 2}) is False

    assert check_dict_almost_equal({'a': 1, 'b': 2.345},
                                   {'a': 1, 'b': 2.346}) is False
    assert check_dict_almost_equal({'a': 1, 'b': 2.345},
                                   {'a': 1, 'b': 2.346},
                                   2) is True
