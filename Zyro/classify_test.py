import csv
import math
from pathlib import Path

path = Path(__file__).with_name('asl_glove_dataset.csv')
cols = [
    'flex_thumb', 'flex_index', 'flex_middle', 'flex_ring', 'flex_pinky',
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z'
]

with path.open(newline='', encoding='utf-8') as f:
    data = list(csv.DictReader(f))

means = {}
counts = {}
for row in data:
    key = (row['label'], row['hand'])
    if key not in means:
        means[key] = {c: 0.0 for c in cols}
        counts[key] = 0
    counts[key] += 1
    for c in cols:
        means[key][c] += float(row[c])
for key in means:
    for c in cols:
        means[key][c] /= counts[key]

obs = {
    'flex_thumb': 780.0,
    'flex_index': 810.0,
    'flex_middle': 805.0,
    'flex_ring': 798.0,
    'flex_pinky': 802.0,
    'accel_x': 0.12,
    'accel_y': 9.71,
    'accel_z': 0.48,
    'gyro_x': 0.01,
    'gyro_y': 0.00,
    'gyro_z': 0.02,
}

results = []
for (label, hand), vec in means.items():
    d = math.sqrt(sum((vec[c] - obs[c]) ** 2 for c in cols))
    results.append((d, label, hand, vec))

results.sort()
for dist, label, hand, vec in results[:10]:
    print(label, hand, f'{dist:.4f}', ','.join(f'{c}:{vec[c]:.3f}' for c in cols))

print('---')

meanlabel = {}
countlabel = {}
for row in data:
    lab = row['label']
    if lab not in meanlabel:
        meanlabel[lab] = {c: 0.0 for c in cols}
        countlabel[lab] = 0
    countlabel[lab] += 1
    for c in cols:
        meanlabel[lab][c] += float(row[c])
for lab in meanlabel:
    for c in cols:
        meanlabel[lab][c] /= countlabel[lab]

global_results = []
for lab, vec in meanlabel.items():
    d = math.sqrt(sum((vec[c] - obs[c]) ** 2 for c in cols))
    global_results.append((d, lab, vec))
global_results.sort()
for dist, lab, vec in global_results[:10]:
    print('GLOBAL', lab, f'{dist:.4f}', ','.join(f'{c}:{vec[c]:.3f}' for c in cols))
