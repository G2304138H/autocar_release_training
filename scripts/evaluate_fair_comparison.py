#!/usr/bin/env python3
"""Shared 0.5-mm graph comparison for saved DeepCA and AutoCAR predictions.

The core lives in vessel_tree_generator to prevent protocol drift. Set
VESSEL_TREE_GENERATOR_ROOT when the repositories are not siblings.
"""
import os
import sys
from pathlib import Path

AUTOCAR_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = Path(os.environ.get('VESSEL_TREE_GENERATOR_ROOT', AUTOCAR_ROOT.parent / 'vessel_tree_generator')).expanduser().resolve()
if not (SHARED_ROOT / 'methods/fair_comparison/core.py').is_file():
    raise SystemExit('Set VESSEL_TREE_GENERATOR_ROOT to the vessel_tree_generator checkout containing methods/fair_comparison/core.py')
sys.path.insert(0, str(AUTOCAR_ROOT))
sys.path.insert(0, str(SHARED_ROOT))
from methods.fair_comparison.offline import main

if __name__ == '__main__':
    raise SystemExit(main())
