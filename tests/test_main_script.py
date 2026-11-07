"""Tests for config.yaml and an end-to-end run of configure12_main.py.

The end-to-end run uses the synthetic case and the fake CityCAT from
synthetic_case.py with a small population, so it finishes in seconds.
"""
import py_compile
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import synthetic_case as sc

ROOT = Path(__file__).resolve().parents[1]

MAIN = ROOT / "configure12_main.py"
EXPECTED_KEYS = {"run_path", "rainfall_num", "config_num", "pop_size",
                 "max_generations", "pond_unit_vol_lcc", "plot_x_limit",
                 "plot_y_limit"}


def test_config_file():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert set(config) == EXPECTED_KEYS
    # values used in the published study
    assert config["rainfall_num"] == [1060, 2060, 3060, 5060, 10060]
    assert config["config_num"] == 3
    assert config["pop_size"] == 100
    assert config["max_generations"] == 50
    assert config["pond_unit_vol_lcc"] == 153.2
    assert config["plot_x_limit"] == [-0.05, 3.1]
    assert config["plot_y_limit"] == [5.84, 6.4]


def test_main_script_compiles(tmp_path):
    py_compile.compile(str(MAIN), cfile=str(tmp_path / "main.pyc"), doraise=True)


@pytest.mark.parametrize("seed", [1])
def test_main_script_end_to_end(tmp_path, monkeypatch, fake_citycat, seed):
    run_path = sc.build_case(tmp_path / "run")
    cfg = sc.write_config(tmp_path / "config.yaml", run_path,
                          pop_size=8, max_generations=2)

    monkeypatch.chdir(tmp_path)               # the script changes directory
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [str(MAIN), "--config", str(cfg)])
    np.random.seed(seed)
    runpy.run_path(str(MAIN), run_name="__main__")

    for name in ["simulated_data.csv", "generation_data.csv",
                 "optimised_data.csv", "Generation_No_0.png",
                 "Generation_No_1.png", "Generation_No_2.png",
                 "Generation_No_2_optimal.png"]:
        assert (run_path / name).exists(), name

    expected_cols = (["SN", "Generation"]
                     + [f"Size_Steps_Pond_{i}" for i in range(3)]
                     + [f"Pond_Volume_{i}" for i in range(3)]
                     + ["Cost", "DD_1030", "DD_2030", "DD_3030", "DD_5030",
                        "DD_10030", "EAD"])
    sim = pd.read_csv(run_path / "simulated_data.csv")
    gen = pd.read_csv(run_path / "generation_data.csv")
    opt = pd.read_csv(run_path / "optimised_data.csv")
    for df in (sim, gen, opt):
        assert list(df.columns) == expected_cols

    # 8 solutions per generation for generations 0, 1 and 2
    assert len(gen) == 8 * 3
    assert sorted(gen["Generation"].unique()) == [0, 1, 2]
    # every simulated design is unique
    sizes = sim[[f"Size_Steps_Pond_{i}" for i in range(3)]]
    assert not sizes.duplicated().any()
    # the no-pond baseline is simulated and costs nothing
    assert ((sizes.sum(axis=1) == 0) & (sim["Cost"] == 0)).any()
    # the optimal set is a non-empty Pareto front of the final generation
    assert len(opt) > 0 and (opt["Generation"] == 2).all()
    assert (opt["EAD"] > 0).all()
