import os
import subprocess
import sys

from pylib._cd_tmpdir_context import cd_tempdir_context

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
pwd = os.getcwd()


@cd_tempdir_context()
def test_run_covaphastsim_smoke():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pylib.cli.run_covaphastsim",
        ],
        check=True,
        env={"PYTHONPATH": f"{pwd}"},
        input=f"""
cfg_p_wt_to_mut: 0.01
cfg_pop_size: 10000
cfg_refseqs: "{assets}/alignedsequences.csv"
cfg_suffix_mut: "'"
cfg_suffix_wt: "+"
replicate_num: 0
trt_mutmx_active_strain_factor: 1.0
trt_mutmx_rel_beta: 1.0
trt_mutmx_withinhost_r: 1.0
trt_name: "test"
trt_seed: 1
""".encode(),
    )
