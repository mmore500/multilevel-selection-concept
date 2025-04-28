import os
import subprocess

import pytest

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")


@pytest.mark.parametrize("hsurf_bits", [0, 64])
def test_run_volzscreen_smoke(hsurf_bits: int):
    subprocess.run(
        [
            "python3",
            "-m",
            "pylib.cli.run_volzscreen",
        ],
        check=True,
        input=f"""
cfg_refphylos: "{assets}/../assets/a=run_covaphastsim.pqt"
screen_num: 0
trt_hsurf_bits: {hsurf_bits}
trt_n_downsample: 10000
trt_clade_size_thresh: 4
""".encode(),
    )
