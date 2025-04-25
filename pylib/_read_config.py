import yaml


def read_config(input_source: object) -> object:
    """Read and parse YAML from either:
    - a file path (str)
    - a file-like object with .read()
    """
    # Determine if input_source is a path or a file-like
    if hasattr(input_source, "read"):
        stream = input_source
        close_after = False
    else:
        stream = open(input_source, "r")
        close_after = True

    try:
        data = yaml.safe_load(stream)
    finally:
        if close_after:
            stream.close()

    return data
