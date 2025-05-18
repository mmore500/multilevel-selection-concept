from contextlib import contextmanager
import os
import tempfile
import typing


@contextmanager
def cd_tempdir_context() -> typing.Generator[str, None, None]:
    """
    Context manager that creates a TemporaryDirectory,
    chdirs into it for the duration of the with-block,
    then returns to the original CWD and cleans it up.
    Yields the path of the temp dir.
    """
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            yield tmpdir
        finally:
            os.chdir(old_cwd)
