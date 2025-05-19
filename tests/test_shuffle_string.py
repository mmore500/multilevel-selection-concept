from pylib._shuffle_string import shuffle_string


def test_empty_string():
    assert shuffle_string("") == ""


def test_single_character():
    assert shuffle_string("a") == "a"
    assert shuffle_string("b") == "b"


def test_two_characters():
    assert shuffle_string("ab") in ["ab", "ba"]
    assert shuffle_string("xy") in ["xy", "yx"]
    while shuffle_string("ab") == "ab":
        pass
    while shuffle_string("xy") == shuffle_string("xy"):
        pass


def test_three_characters():
    assert shuffle_string("abc") in ["abc", "acb", "bac", "bca", "cab", "cba"]
    assert shuffle_string("xyz") in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    while shuffle_string("abc") == "abc":
        pass
    while shuffle_string("xyz") == shuffle_string("xyz"):
        pass
