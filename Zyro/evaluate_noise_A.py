import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)
path = 'asl_glove_dataset.csv'
feature_cols = [
    'flex_thumb', 'flex_index', 'flex_middle', 'flex_ring', 'flex_pinky',
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z'
]
flex_cols = ['flex_thumb', 'flex_index', 'flex_middle', 'flex_ring', 'flex_pinky']

# Load data
full_df = pd.read_csv(path)

# Select 5 random rows for label A
A_df = full_df[full_df['label'] == 'A'].sample(n=5, random_state=42)
A_indices = A_df.index.tolist()
ref_df = full_df.drop(index=A_indices)

# Encode hand
hand_le = LabelEncoder()
ref_df['hand_encoded'] = hand_le.fit_transform(ref_df['hand'])
A_df['hand_encoded'] = hand_le.transform(A_df['hand'])

# Prepare training data from the reference dataset
X_ref = ref_df[feature_cols + ['hand_encoded']].to_numpy(dtype=float)
y_ref = LabelEncoder().fit_transform(ref_df['label'])
label_le = LabelEncoder().fit(ref_df['label'])

# Scale features
scaler = StandardScaler().fit(X_ref)
X_ref_scaled = scaler.transform(X_ref)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_ref_scaled, y_ref)

# Prepare clean and noisy test samples
X_A_clean = A_df[feature_cols + ['hand_encoded']].to_numpy(dtype=float)
X_A_clean_scaled = scaler.transform(X_A_clean)

def add_noise(sample):
    noisy = sample.copy()
    flex_indices = list(range(5))
    accel_indices = [5, 6, 7]
    gyro_indices = [8, 9, 10]
    noisy[flex_indices] += np.random.normal(0, 20, size=5)
    noisy[accel_indices] += np.random.normal(0, 0.3, size=3)
    noisy[gyro_indices] += np.random.normal(0, 0.3, size=3)
    return noisy

X_A_noisy = np.vstack([add_noise(x) for x in X_A_clean])
X_A_noisy_scaled = scaler.transform(X_A_noisy)

# Predict
pred_clean = clf.predict(X_A_clean_scaled)
pred_noisy = clf.predict(X_A_noisy_scaled)

true_labels = ['A'] * len(A_df)
clean_labels = label_le.inverse_transform(pred_clean)
noisy_labels = label_le.inverse_transform(pred_noisy)

# Report
print('selected_indices', A_indices)
print('original_rows:')
for idx, row in A_df.iterrows():
    print(idx, row[feature_cols + ['hand']].to_dict())

print('\npredictions_clean:')
for i, pred in enumerate(clean_labels):
    print(f'{i}: {pred}')

print('\npredictions_noisy:')
for i, pred in enumerate(noisy_labels):
    print(f'{i}: {pred}')

misclassified = [i for i, pred in enumerate(noisy_labels) if pred != 'A']
print(f'\nmisclassified_count: {len(misclassified)} / {len(noisy_labels)}')
if misclassified:
    print('misclassified_indices:', misclassified)

# Robustness summary
accuracy_noisy = sum(pred == 'A' for pred in noisy_labels) / len(noisy_labels)
print(f'noisy_accuracy: {accuracy_noisy:.2f}')

# Distance to nearest label mean for noisy samples
label_means = ref_df.groupby('label')[flex_cols].mean()
for i, noisy in enumerate(X_A_noisy):
    distances = {}
    for label, mean_vec in label_means.iterrows():
        distances[label] = np.linalg.norm(noisy[:5] - mean_vec.values)
    closest = sorted(distances.items(), key=lambda x: x[1])[:3]
    print(f'\nnoisy sample {i} top 3 nearest labels by flex distance:')
    for label, dist in closest:
        print(f'  {label}: {dist:.2f}')
