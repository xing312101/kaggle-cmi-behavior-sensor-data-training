import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score
import datetime
import os

# 定義資料路徑
WORKING_DIR = '/kaggle/working'
TRAIN_CSV_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
TRAIN_DEMO_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv'

def load_and_preprocess_data(train_csv_path, demos_csv_path):
    """
    載入並預處理訓練資料。
    """
    print("正在載入資料...")
    train_df = pd.read_csv(train_csv_path)
    demographics_df = pd.read_csv(demos_csv_path)
    train_df = pd.merge(train_df, demographics_df, on='subject', how='left')

    # --- 新增的邏輯：動態定義非目標手勢 ---
    # 根據 'sequence_type' 欄位來區分目標與非目標手勢
    # 取得所有非目標序列的 gesture 名稱
    non_target_gestures_df = train_df[train_df['sequence_type'] != 'Target']
    non_target_gestures = non_target_gestures_df['gesture'].unique().tolist()
    print(f"自動偵測到的非目標手勢類別有: {non_target_gestures}")

    # 確保所有數值型和類別型特徵都正確定義
    sensor_features = [
        'acc_x', 'acc_y', 'acc_z',
        'rot_w', 'rot_x', 'rot_y', 'rot_z',
        'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'
    ]
    tof_features = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
    
    demographic_features = [
        'adult_child', 'age', 'sex', 'handedness', 'height_cm',
        'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
    ]
    
    numerical_features = sensor_features + tof_features + demographic_features
    categorical_features = ['orientation', 'behavior', 'phase']
    
    print("正在處理類別型特徵...")
    for col in categorical_features:
        train_df[f'{col}_encoded'] = train_df[col].astype('category').cat.codes
    
    print("正在處理缺失值...")
    train_df[numerical_features] = train_df[numerical_features].fillna(0)
    
    all_features = numerical_features + [f'{col}_encoded' for col in categorical_features]
    
    print("正在進行 gesture 標籤編碼...")
    label_encoder = LabelEncoder()
    train_df['gesture_encoded'] = label_encoder.fit_transform(train_df['gesture'])
    
    print("正在將資料分組為序列...")
    grouped_sequences = train_df.groupby('sequence_id')
    sequences = []
    labels = []
    subjects = []
    
    for _, group in grouped_sequences:
        sequences.append(group[all_features].values)
        labels.append(group['gesture_encoded'].iloc[0])
        subjects.append(group['subject'].iloc[0])

    print("正在統一序列長度...")
    sequence_lengths = [len(s) for s in sequences]
    max_seq_len = int(np.percentile(sequence_lengths, 99))
    print(f"原始最大序列長度: {max(sequence_lengths)}, 調整後的最大序列長度: {max_seq_len}")
    
    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_seq_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )
    
    print("正在標準化數值型數據...")
    scaler = StandardScaler()
    num_feat_count = len(numerical_features)
    original_shape = padded_sequences.shape
    padded_sequences[:, :, :num_feat_count] = scaler.fit_transform(
        padded_sequences[:, :, :num_feat_count].reshape(-1, num_feat_count)
    ).reshape(original_shape[0], original_shape[1], num_feat_count)

    return padded_sequences, np.array(labels), np.array(subjects), label_encoder, non_target_gestures

def build_model(input_shape, num_classes):
    """
    建立一個更強大的 Bidirectional LSTM 模型。
    """
    print("正在建立模型架構...")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_metrics(y_true, y_pred, label_encoder, non_target_gestures):
    """
    計算競賽所需的二元 F1 和宏觀 F1 分數。
    """
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    is_target_true = np.isin(y_true_labels, non_target_gestures, invert=True).astype(int)
    is_target_pred = np.isin(y_pred_labels, non_target_gestures, invert=True).astype(int)
    binary_f1 = f1_score(is_target_true, is_target_pred)
    
    y_true_macro = np.where(np.isin(y_true_labels, non_target_gestures), 'non_target', y_true_labels)
    y_pred_macro = np.where(np.isin(y_pred_labels, non_target_gestures), 'non_target', y_pred_labels)
    
    all_classes = np.unique(np.concatenate([y_true_macro, y_pred_macro]))
    macro_f1 = f1_score(y_true_macro, y_pred_macro, labels=all_classes, average='macro', zero_division=0)
    
    final_score = (binary_f1 + macro_f1) / 2
    
    return binary_f1, macro_f1, final_score

# --- 主要訓練流程 ---
# 移除了手動定義的 NON_TARGET_GESTURES 列表
X, y_gesture, subjects, label_encoder, non_target_gestures = load_and_preprocess_data(TRAIN_CSV_PATH, TRAIN_DEMO_PATH)
num_gesture_classes = len(label_encoder.classes_)
print(f"總共 {num_gesture_classes} 個動作類別。")
print(f"輸入數據形狀: {X.shape}")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
best_model_path = os.path.join(WORKING_DIR, f'best_overall_model_{timestamp}.h5')
best_final_score = -1.0
fold_scores = []

print("\n開始交叉驗證訓練...")
gkf = GroupKFold(n_splits=5)
for fold, (train_index, val_index) in enumerate(gkf.split(X, y_gesture, groups=subjects)):
    print(f"\n--- 折疊 (Fold) {fold+1}/5 ---")
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_gesture[train_index], y_gesture[val_index]
    
    y_train_one_hot = to_categorical(y_train, num_classes=num_gesture_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=num_gesture_classes)

    model = build_model(X_train.shape[1:], num_gesture_classes)
    
    temp_checkpoint_filepath = os.path.join(WORKING_DIR, f'temp_model_fold_{fold+1}.h5')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=temp_checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model.fit(
        X_train,
        y_train_one_hot,
        epochs=50,
        batch_size=256,
        validation_data=(X_val, y_val_one_hot),
        verbose=1,
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )

    model.load_weights(temp_checkpoint_filepath)
    
    y_pred_one_hot = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred_one_hot, axis=1)
    
    # 傳入動態生成的 NON_TARGET_GESTURES 列表
    binary_f1, macro_f1, final_score = evaluate_metrics(
        y_val, y_pred_classes, label_encoder, non_target_gestures
    )
    
    print(f"\n折疊 {fold+1} 評估結果:")
    print(f"  二元 F1 (Binary F1): {binary_f1:.4f}")
    print(f"  宏觀 F1 (Macro F1): {macro_f1:.4f}")
    print(f"  最終得分 (Final Score): {final_score:.4f}")
    fold_scores.append(final_score)
    
    if final_score > best_final_score:
        print(f"  新最佳得分 {final_score:.4f}！正在儲存模型...")
        best_final_score = final_score
        model.save(best_model_path)
    
    os.remove(temp_checkpoint_filepath)

print("\n訓練流程完成。")
print(f"所有折疊的最終得分: {np.round(fold_scores, 4)}")
print(f"平均最終得分: {np.mean(fold_scores):.4f}")
print(f"\n最佳模型已儲存至: {best_model_path}，最終得分為: {best_final_score:.4f}")

