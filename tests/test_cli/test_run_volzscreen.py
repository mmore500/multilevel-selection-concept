import glob
import os
import subprocess
import sys

import more_itertools as mit
import pytest

from pylib._cd_tmpdir_context import cd_tempdir_context

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
pwd = os.getcwd()


@cd_tempdir_context()
@pytest.mark.parametrize("hsurf_bits", [0, 64])
def test_run_volzscreen_smoke(hsurf_bits: int):

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pylib.cli.run_volzscreen",
        ],
        check=True,
        env={"PYTHONPATH": f"{pwd}"},
        input=f"""
cfg_clade_size_thresh: "[0]"
cfg_mut_count_thresh_lb: 5
cfg_mut_count_thresh_ub: {sys.maxsize}
cfg_mut_freq_thresh_lb: 0.0
cfg_mut_freq_thresh_ub: 0.05
cfg_mut_quant_thresh_lb: 0.8
cfg_mut_quant_thresh_ub: 1.0
cfg_refphylos: "{assets}/a=run_covaphastsim.pqt"
screen_num: 0
trt_hsurf_bits: {hsurf_bits}
trt_n_downsample: 1000
""".encode(),
    )


@cd_tempdir_context()
@pytest.mark.parametrize("hsurf_bits", [0, 64])
def test_run_volzscreen_covaphastsim_smoke(hsurf_bits: int):

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pylib.cli.run_covaphastsim",
        ],
        check=True,
        env={"PYTHONPATH": f"{pwd}"},
        input=f"""
cfg_maxseqlen: 300
cfg_make_cv_sim_recipe: "make_cv_sim_uk"
cfg_make_wt_specs_recipe: "make_wt_specs_uk"
cfg_num_mut_sites: 1
cfg_p_wt_to_mut: 0.01
cfg_p_seq_mut: 2.74e-6
cfg_pop_size: 4000
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

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pylib.cli.run_volzscreen",
        ],
        check=True,
        env={"PYTHONPATH": pwd},
        input=f"""
cfg_clade_size_thresh: "[0]"
cfg_mut_count_thresh_lb: 5
cfg_mut_count_thresh_ub: {sys.maxsize}
cfg_mut_freq_thresh_lb: 0.0
cfg_mut_freq_thresh_ub: 0.05
cfg_mut_quant_thresh_lb: 0.8
cfg_mut_quant_thresh_ub: 1.0
cfg_refphylos: "{mit.one(glob.glob('*.pqt'))}"
screen_num: 0
trt_hsurf_bits: {hsurf_bits}
trt_n_downsample: 1000
""".encode(),
    )


@cd_tempdir_context()
@pytest.mark.parametrize("hsurf_bits", [0, 64])
def test_run_volzscreen_covasim_smoke(hsurf_bits: int):

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pylib.cli.run_covasim",
        ],
        check=True,
        env={"PYTHONPATH": f"{pwd}"},
        input=f"""
cfg_maxseqlen: 20
cfg_make_cv_sim_recipe: "make_cv_sim_uk"
cfg_make_wt_specs_recipe: "make_wt_specs_uk"
cfg_num_mut_sites: 1
cfg_p_wt_to_mut: 0.01
cfg_p_seq_mut: 2.74e-6
cfg_pop_size: 4000
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

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pylib.cli.run_volzscreen",
        ],
        check=True,
        env={"PYTHONPATH": pwd},
        input=f"""
cfg_clade_size_thresh: "[0]"
cfg_mut_count_thresh_lb: 5
cfg_mut_count_thresh_ub: {sys.maxsize}
cfg_mut_freq_thresh_lb: 0.0
cfg_mut_freq_thresh_ub: 0.05
cfg_mut_quant_thresh_lb: 0.8
cfg_mut_quant_thresh_ub: 1.0
cfg_refphylos: "{mit.one(glob.glob('*.pqt'))}"
screen_num: 0
trt_hsurf_bits: {hsurf_bits}
trt_n_downsample: 1000
""".encode(),
    )
