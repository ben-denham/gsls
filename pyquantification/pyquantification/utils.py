from typing import Any, Dict, Sequence


def select_keys(my_dict: Dict, keys: Sequence) -> Dict:
    """Return a copy of my_dict with only the given keys."""
    keyset = set(keys)
    return {k: v for k, v in my_dict.items() if k in keyset}


def prefix_keys(my_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Return a copy of my_dict with prefix prepended to each key"""
    return {'{}{}'.format(prefix, key): value
            for key, value in my_dict.items()}


def dict_first(my_dict: Dict) -> Any:
    """Return first value from the given dictionary."""
    return list(my_dict.values())[0]


def normalise_dict(my_dict: Dict[Any, float]) -> Dict[Any, float]:
    """Normalise the values in the dictionary so they sum to 1.0."""
    values_sum = sum(my_dict.values())
    return {k: v / values_sum for k, v in my_dict.items()}


def check_dict_almost_equal(dict_a: Dict[Any, float],
                            dict_b: Dict[Any, float],
                            decimal: int = 7) -> bool:
    """Checks if the dicts contain the same keys, and that those keys have
    the same numeric value to a specified number of decimal places."""
    if set(dict_a.keys()) != set(dict_b.keys()):
        return False
    for key in dict_a.keys():
        # Same test as np.testing.assert_almost_equal
        if abs(dict_a[key] - dict_b[key]) >= (1.5 * 10**(-decimal)):
            return False
    return True
