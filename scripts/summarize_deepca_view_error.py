#!/usr/bin/env python3
"""Combine overlap metrics with fair-comparison CD and radius MAE.

Branch metrics are averaged within each case, then across cases. SE uses
sample SD / sqrt(number of valid cases); missing values are never zero-filled.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

CONDITIONS = [('error_0p5deg', 0.5), ('error_1deg', 1), ('error_2deg', 2), ('error_4deg', 4)]


def read(path):
    return json.loads(path.read_text())


def case_id(row):
    value = str(row['case_id'])
    return str(int(value)) if value.isdecimal() else value


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {'mean': statistics.mean(values) if values else None,
            'se': statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
            'n': len(values)}


def branch_case_means(rows, metric, radius=False):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        if radius and not (row['region'] == 'all' and row['gt_radius_bin'] == 'all' and row['num_gt_points']):
            continue
        key = (case_id(row), row['branch'])
        if key in seen:
            raise ValueError(f'Duplicate case/branch row: {key}')
        seen.add(key)
        value = row.get(metric)
        if value is not None and math.isfinite(float(value)):
            grouped[key[0]].append(float(value))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def collect(root):
    output, case_output = [], []
    for artery in ('rca', 'lca'):
        expected_cases = None
        for condition, degrees in CONDITIONS:
            fair = root / f'deepca_{artery}_views_2_inacc_view' / condition
            overlap = root / f'deepca_{artery}_views_2_inacc_view_overlap' / condition
            cases = read(fair / 'cases.json')
            volume = read(overlap / 'per_case_metrics.json')
            ids = {case_id(row) for row in cases}
            volume_ids = {case_id(row) for row in volume}
            if len(ids) != len(cases) or len(volume_ids) != len(volume) or ids != volume_ids:
                raise ValueError(f'{artery}/{condition}: duplicate or different case sets between evaluators')
            if expected_cases is not None and ids != expected_cases:
                raise ValueError(f'{artery}: case sets differ across offsets')
            expected_cases = ids
            if any(row.get('ground_truth_volume_file') is not None for row in volume):
                raise ValueError('Expected raw-vessel tube reference, found voxel GT')
            cd = branch_case_means(read(fair / 'branches.json'), 'chamfer_mm')
            radius = branch_case_means(read(fair / 'radius.json'), 'mae_mm', radius=True)
            if not (set(cd) | set(radius)) <= ids:
                raise ValueError('Branch metrics contain unexpected cases')
            metrics = {'dice_percent': [], 'cldice_percent': [], 'cd_mm': [], 'radius_mae_mm': []}
            for row in volume:
                case = case_id(row)
                values = {'dice_percent': None if row['dice_3d'] is None else 100 * row['dice_3d'],
                          'cldice_percent': None if row['cldice_3d'] is None else 100 * row['cldice_3d'],
                          'cd_mm': cd.get(case), 'radius_mae_mm': radius.get(case)}
                case_output.append({'artery': artery.upper(), 'offset_deg': degrees, 'case_id': case, **values})
                for metric, value in values.items():
                    metrics[metric].append(value)
            result = {'artery': artery.upper(), 'offset_deg': degrees, 'num_cases': len(ids)}
            for name, values in metrics.items():
                result.update({f'{name}_{key}': value for key, value in stats(values).items()})
            output.append(result)
    return output, case_output


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    rows, cases = collect(args.output_root)
    write_csv(args.output_root / 'deepca_view_error_table.csv', rows)
    write_csv(args.output_root / 'deepca_view_error_per_case.csv', cases)
    lines = ['| Artery | Offset | Dice (%), mean ± SE | clDice (%), mean ± SE | CD (mm), mean ± SE | Radius MAE (mm), mean ± SE |',
             '|---|---|---|---|---|---|']
    for row in rows:
        cells = [row['artery'], f"{row['offset_deg']:g}°"]
        for metric, digits in [('dice_percent', 2), ('cldice_percent', 2), ('cd_mm', 2), ('radius_mae_mm', 3)]:
            mean, se = row[metric + '_mean'], row[metric + '_se']
            cells.append('—' if mean is None else f'{mean:.{digits}f} ± ' + ('NA' if se is None else f'{se:.{digits}f}'))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'SE = sample SD / sqrt(valid cases). Branch metrics are averaged within each case first.',
              'Dice/clDice: raw-vessel tubes, 0.5 mm grid, bounds [-58.5, 58.5] mm on each axis.',
              'CD: fair-comparison all-branch Chamfer. Radius: fair-comparison radius.json (mae_mm), all branches, all radii.',
              'Valid case counts per metric are included in the CSV. No-offset baseline is not evaluated.']
    text = '\n'.join(lines) + '\n'
    (args.output_root / 'deepca_view_error_table.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
