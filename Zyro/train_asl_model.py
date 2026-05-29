import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

try:
    from micromlgen import port
except ImportError:
    port = None
    print("WARNING: micromlgen is not installed. Model export to C will be skipped.")


def load_data(path):
    df = pd.read_csv(path)
    feature_cols = [
        'flex_thumb', 'flex_index', 'flex_middle', 'flex_ring', 'flex_pinky',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]
    return df, feature_cols


def encode_columns(df):
    hand_encoder = LabelEncoder()
    label_encoder = LabelEncoder()

    df = df.copy()
    df['hand_encoded'] = hand_encoder.fit_transform(df['hand'])
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    print('hand encoding:', dict(zip(hand_encoder.classes_, hand_encoder.transform(hand_encoder.classes_))))
    print('label encoding:')
    for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f'  {label} -> {code}')

    return df, hand_encoder, label_encoder


def print_cpp_array(name, values):
    values_str = ', '.join(f'{v:.6f}f' for v in values)
    print(f'const float {name}[{len(values)}] = {{ {values_str} }};')


def export_model(clf, filename='asl_model.h'):
    if port is None:
        print('micromlgen is not installed; skipping model export.')
        return
    c_code = port(clf, class_name='ASLModel')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(c_code)
    print(f'Exported model to {filename}')


def plot_confusion_matrix(y_true, y_pred, labels):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap='Blues',
        normalize=None,
        xticks_rotation='vertical'
    )
    plt.tight_layout()
    plt.title('ASL Gesture Confusion Matrix')
    plt.show()


def main():
    df, feature_cols = load_data('asl_glove_dataset.csv')
    df, hand_encoder, label_encoder = encode_columns(df)

    X = df[feature_cols + ['hand_encoded']].to_numpy(dtype=float)
    y = df['label_encoded'].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    target_names = label_encoder.classes_
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=target_names))

    plot_confusion_matrix(y_test, y_pred, target_names)

    export_model(clf, filename='asl_model.h')

    print('\n// StandardScaler parameters for Arduino')
    print_cpp_array('SCALER_MEAN', scaler.mean_)
    print_cpp_array('SCALER_STD', scaler.scale_)

    print('\n// Feature order')
    print('const char* FEATURE_NAMES[] = {')
    for col in feature_cols + ['hand_encoded']:
        print(f'    "{col}",')
    print('};')

    print('\n// Label mapping')
    for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f'// {code} -> {label}')

    print('\n// Hand mapping')
    for hand, code in zip(hand_encoder.classes_, hand_encoder.transform(hand_encoder.classes_)):
        print(f'// {code} -> {hand}')


if __name__ == '__main__':
    main()
