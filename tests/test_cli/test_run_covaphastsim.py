import subprocess


def test_run_covaphastsim_smoke():
    subprocess.run(
        [
            "python3",
            "-m",
            "pylib.cli.run_covaphastsim",
        ],
        check=True,
        input="""
cfg_p_wt_to_mut: 0.01
cfg_pop_size: 10000
cfg_suffix_mut: "'"
cfg_suffix_wt: "+"
trt_mutmx_active_strain_factor: 1.0
trt_mutmx_rel_beta: 1.0
trt_mutmx_withinhost_r: 1.0
trt_seed: 1
""".encode(),
    )
