import random


def shuffle_string(s: str) -> str:
    """
    Shuffle the characters of a string.

    Args:
        s (str): The input string to shuffle.

    Returns:
        str: A new string with the characters shuffled.
    """
    return "".join(random.sample(s, len(s)))
