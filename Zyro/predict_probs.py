import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

path = 'asl_glove_dataset.csv'
df = pd.read_csv(path)
cols = ['flex_thumb','flex_index','flex_middle','flex_ring','flex_pinky','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']
X = df[cols].to_numpy(dtype=float)
hand_enc = LabelEncoder().fit_transform(df['hand'])
X = np.column_stack([X, hand_enc])
label_encoder = LabelEncoder().fit(df['label'])
y = label_encoder.transform(df['label'])
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X, y)
obs = np.array([780.0, 810.0, 805.0, 798.0, 802.0, 0.12, 9.71, 0.48, 0.01, 0.0, 0.02, 1.0]).reshape(1, -1)
probs = clf.predict_proba(obs)[0]
top3 = sorted(zip(label_encoder.inverse_transform(range(len(probs))), probs), key=lambda x: -x[1])[:3]
for label, prob in top3:
    print(label, f'{prob:.4f}')
