import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('asl_glove_dataset.csv')
cols = ['flex_thumb','flex_index','flex_middle','flex_ring','flex_pinky','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']
X = df[cols].to_numpy(dtype=float)
hand_enc = LabelEncoder().fit_transform(df['hand'])
X = np.column_stack([X, hand_enc])
label_encoder = LabelEncoder().fit(df['label'])
y = label_encoder.transform(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(labels)))
support = cm.sum(axis=1)
recall = np.diag(cm) / support
for lab, r in zip(labels, recall):
    print(f"{lab}: {r:.4f}")
print('\nbelow_85:')
for lab, r in zip(labels, recall):
    if r < 0.85:
        print(f"{lab}: {r:.4f}")
