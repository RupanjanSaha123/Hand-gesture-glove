import csv
import math
from itertools import combinations
from pathlib import Path

path = Path(__file__).with_name('asl_glove_dataset.csv')
cols = [
    'flex_thumb', 'flex_index', 'flex_middle', 'flex_ring', 'flex_pinky',
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z'
]
with path.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data = [row for row in reader]

labels = sorted({row['label'] for row in data})
label_vals = {label: {c: [] for c in cols} for label in labels}
for row in data:
    lab = row['label']
    for c in cols:
        label_vals[lab][c].append(float(row[c]))

means = {
    lab: {c: sum(vals) / len(vals) for c, vals in label_vals[lab].items()}
    for lab in labels
}
feature_means = {c: [means[lab][c] for lab in labels] for c in cols}
feature_variance = {
    c: sum((x - sum(vals) / len(vals)) ** 2 for x in vals) / len(vals)
    for c, vals in feature_means.items()
}
label_vecs = {lab: [means[lab][c] for c in cols] for lab in labels}

dists = []
for a, b in combinations(labels, 2):
    d = math.sqrt(sum((label_vecs[a][i] - label_vecs[b][i]) ** 2 for i in range(len(cols))))
    cos = sum(label_vecs[a][i] * label_vecs[b][i] for i in range(len(cols))) / (
        math.sqrt(sum(x * x for x in label_vecs[a])) * math.sqrt(sum(x * x for x in label_vecs[b]))
    )
    man = sum(abs(label_vecs[a][i] - label_vecs[b][i]) for i in range(len(cols)))
    inf_ = max(abs(label_vecs[a][i] - label_vecs[b][i]) for i in range(len(cols)))
    dists.append((d, a, b, cos, man, inf_))

dists.sort(key=lambda x: x[0])

print('labels_total', len(labels))
print('top_similar')
for d, a, b, cos, man, inf_ in dists[:10]:
    print(f'{a},{b},{d:.4f},{cos:.6f},{man:.4f},{inf_:.4f}')
print('feature_variance')
for c, v in sorted(feature_variance.items(), key=lambda x: -x[1]):
    print(f'{c},{v:.6f}')
for lab in labels:
    row = means[lab]
    print('LABEL', lab, ','.join(f'{c}:{row[c]:.3f}' for c in cols))
