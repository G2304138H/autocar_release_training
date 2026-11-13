#!/usr/bin/env python3
"""Preflight the exact joint Hydra configuration, then launch shared training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--split-report", type=Path, default=ROOT / "logs/joint_effective_splits.json")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides after --")
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        config = compose(config_name="train", overrides=["experiment=stage2_npz_joint", *overrides])
    datamodule = instantiate(config.data)
    from src.dataset.joint_stage2_datamodule import JointStage2NPZDataModule
    if not isinstance(datamodule, JointStage2NPZDataModule):
        raise ValueError("This launcher requires the joint data module")
    datamodule.setup()
    for artery, module in datamodule.modules.items():
        counts = {name: len(ids) for name, ids in datamodule.effective_splits[artery].items()}
        print(f"{artery}: {counts}; excluded={module.applied_train_case_exclusions}")
        for split, dataset in (("train", module.data_train), ("val", module.data_val), ("test", module.data_test)):
            dataset.validate_projection_metadata()
            sample = dataset[0]
            print(f"  {split}: case={sample['case_id']}, images={tuple(sample['images'].shape)}, "
                  f"GT={tuple(sample['gt_volume_zyx'].shape)}, "
                  f"pixel_spacing={float(sample['imager_pixel_spacing_mm']):g} mm")
    args.split_report.parent.mkdir(parents=True, exist_ok=True)
    args.split_report.write_text(json.dumps(datamodule.effective_splits, indent=2) + "\n")
    for artery, splits in datamodule.effective_splits.items():
        path = args.split_report.with_name(f"{args.split_report.stem}_{artery}.json")
        path.write_text(json.dumps(splits, indent=2) + "\n")
    print(f"Preflight passed. Effective patient splits: {args.split_report}")
    if not args.preflight_only:
        command = [sys.executable, "-m", "src.train", "experiment=stage2_npz_joint", *overrides]
        print("Launching:", shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
