import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = 'asl_glove_dataset.csv'
df = pd.read_csv(path)
feature_cols = ['flex_thumb','flex_index','flex_middle','flex_ring','flex_pinky']

# Compute mean flex values per label across all hands
label_means = df.groupby('label')[feature_cols].mean()

# Find pairs with all 5 flex differences <= 50
pairs = []
labels = sorted(label_means.index)
for i, a in enumerate(labels):
    for b in labels[i+1:]:
        diffs = np.abs(label_means.loc[a] - label_means.loc[b])
        if np.all(diffs <= 50):
            pairs.append((a,b,diffs.values))

print('confusable_pairs')
for a,b,diffs in pairs:
    print(a,b,','.join(f'{v:.2f}' for v in diffs))

# Evaluate hand discriminating power
X = df[['flex_thumb','flex_index','flex_middle','flex_ring','flex_pinky','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']].to_numpy(dtype=float)
hand_enc = LabelEncoder().fit_transform(df['hand'])
y = LabelEncoder().fit_transform(df['label'])
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Train with hand feature
X_with_hand = np.column_stack([X_scaled, hand_enc])
X_train, X_test, y_train, y_test = train_test_split(X_with_hand, y, test_size=0.2, random_state=42, stratify=y)
clf_with_hand = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
y_pred_with_hand = clf_with_hand.predict(X_test)
acc_with_hand = accuracy_score(y_test, y_pred_with_hand)

# Train without hand feature
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf_no_hand = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train2, y_train2)
y_pred_no_hand = clf_no_hand.predict(X_test2)
acc_no_hand = accuracy_score(y_test2, y_pred_no_hand)

print('\naccuracy_with_hand', acc_with_hand)
print('accuracy_without_hand', acc_no_hand)
print('hand_improvement', acc_with_hand - acc_no_hand)

# Feature importance from with-hand model
importances = clf_with_hand.feature_importances_
feature_names = ['flex_thumb','flex_index','flex_middle','flex_ring','flex_pinky','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','hand']
print('\nfeature_importance')
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(name, imp)

# Check per-label hand distribution
hand_label = df.groupby(['label','hand']).size().unstack(fill_value=0)
print('\nhand_distribution')
print(hand_label)
