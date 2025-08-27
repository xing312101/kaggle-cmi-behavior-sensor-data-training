import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score
import os
from sklearn.utils import class_weight
from tensorflow.keras import backend as K

# =========================================================================
# 設定隨機種子以確保可重現性
# =========================================================================
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"隨機種子已設定為 {seed}")

set_seeds()

# 定義資料路徑
WORKING_DIR = '/kaggle/working'
TRAIN_CSV_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
TRAIN_DEMO_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv'
TEST_CSV_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
TEST_DEMO_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'

# 全域定義特徵列表，確保一致性
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
manual_features = ['acc_mag']

numerical_features = sensor_features + tof_features + demographic_features + manual_features
categorical_features = ['orientation', 'behavior', 'phase']

def load_and_preprocess_data(csv_path, demos_path, is_training=True):
    """
    載入並預處理資料，並加入手動特徵工程。
    """
    print(f"正在載入和處理資料: {csv_path}")
    df = pd.read_csv(csv_path)
    demographics_df = pd.read_csv(demos_path)
    df = pd.merge(df, demographics_df, on='subject', how='left')

    non_target_gestures = []
    if is_training:
        print("正在動態定義非目標手勢...")
        non_target_gestures_df = df[df['sequence_type'] != 'Target']
        non_target_gestures = non_target_gestures_df['gesture'].unique().tolist()
        print(f"非目標手勢類別: {non_target_gestures}")
        
    df = df.copy()

    print("正在進行手動特徵工程...")
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

    new_cols = {}
    print("正在處理類別型特徵...")
    for col in categorical_features:
        if col in df.columns:
            new_cols[f'{col}_encoded'] = df[col].astype('category').cat.codes
        else:
            new_cols[f'{col}_encoded'] = 0
            
    if is_training:
        print("正在進行 gesture 標籤編碼...")
        label_encoder = LabelEncoder()
        new_cols['gesture_encoded'] = label_encoder.fit_transform(df['gesture'])
    else:
        label_encoder = None

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print("正在處理缺失值...")
    df[numerical_features] = df[numerical_features].fillna(0)
    
    all_features = numerical_features + [f'{col}_encoded' for col in categorical_features]

    print("正在將資料分組為序列...")
    grouped_sequences = df.groupby('sequence_id')
    sequences = []
    labels = []
    subjects = []
    sequence_ids = []
    
    for sequence_id, group in grouped_sequences:
        sequences.append(group[all_features].values)
        sequence_ids.append(sequence_id)
        if is_training:
            labels.append(group['gesture_encoded'].iloc[0])
            subjects.append(group['subject'].iloc[0])

    return sequences, np.array(labels) if is_training else None, np.array(subjects) if is_training else None, np.array(sequence_ids), label_encoder, non_target_gestures


def build_model(input_shape, num_classes):
    """建立 Bidirectional LSTM 模型。"""
    print("建立模型架構...")
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_metrics(y_true, y_pred, label_encoder, non_target_gestures):
    """計算競賽所需的二元 F1 和宏觀 F1 分數。"""
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

def predict_and_generate_submission(
    model_path, test_sequences, test_sequence_ids, label_encoder, max_seq_len, scaler
):
    """
    使用最佳模型進行預測，並生成 submission.parquet 檔案。
    """
    print("\n--- 正在進行預測 ---")
    padded_test_sequences = pad_sequences(
        test_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post'
    )
    
    num_feat_count = len(numerical_features)
    original_shape = padded_test_sequences.shape
    padded_test_sequences[:, :, :num_feat_count] = scaler.transform(
        padded_test_sequences[:, :, :num_feat_count].reshape(-1, num_feat_count)
    ).reshape(original_shape[0], original_shape[1], num_feat_count)

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"無法載入模型：{e}")
        return

    predictions = model.predict(padded_test_sequences, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_gestures = label_encoder.inverse_transform(predicted_classes)

    submission_df = pd.DataFrame({
        'sequence_id': test_sequence_ids,
        'gesture': predicted_gestures
    })
    
    submission_file_path = os.path.join(WORKING_DIR, 'submission.parquet')
    submission_df.to_parquet(submission_file_path, index=False)
    
    print(f"\n--- 提交檔案 '{submission_file_path}' 成功生成！ ---")
    print(submission_df.head())

# =========================================================================
# 主程式
# =========================================================================

# 載入和預處理訓練資料 (只需要做一次)
train_sequences, y_gesture, subjects, _, label_encoder, non_target_gestures = load_and_preprocess_data(TRAIN_CSV_PATH, TRAIN_DEMO_PATH)

# =========================================================================
# 關鍵修改: 在這裡打亂序列的順序，而不是序列內的數據點
# =========================================================================
print("正在打亂動作序列的順序...")
# 將所有相關列表打包
combined = list(zip(train_sequences, y_gesture, subjects))
np.random.shuffle(combined) # 打亂打包的列表
train_sequences[:], y_gesture[:], subjects[:] = zip(*combined) # 解包回各自的列表

# 計算並標準化，同時保存 scaler 和 max_seq_len
sequence_lengths = [len(s) for s in train_sequences]
max_seq_len = int(np.percentile(sequence_lengths, 99))
print(f"最大序列長度: {max_seq_len}")

padded_sequences = pad_sequences(
    train_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post'
)
scaler = StandardScaler()
num_feat_count = len(numerical_features)
original_shape = padded_sequences.shape
X = padded_sequences.copy()
X[:, :, :num_feat_count] = scaler.fit_transform(
    padded_sequences[:, :, :num_feat_count].reshape(-1, num_feat_count)
).reshape(original_shape[0], original_shape[1], num_feat_count)

num_gesture_classes = len(label_encoder.classes_)
print(f"總共 {num_gesture_classes} 個動作類別。")
print(f"輸入數據形狀: {X.shape}")

best_model_path = os.path.join(WORKING_DIR, 'best_overall_model.h5')
best_final_score = -1.0
fold_scores = []

# 計算類別權重
print("\n正在計算類別權重...")
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_gesture),
    y=y_gesture
)
class_weights_dict = dict(enumerate(class_weights))
print("計算出的類別權重：", class_weights_dict)

# 開始交叉驗證訓練
print("\n開始交叉驗證訓練...")
n_sp = 5 
gkf = GroupKFold(n_splits=n_sp)
for fold, (train_index, val_index) in enumerate(gkf.split(X, y_gesture, groups=subjects)):
    print(f"\n- 折疊 (Fold) {fold+1}/{n_sp}")
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_gesture[train_index], y_gesture[val_index]
    
    y_train_one_hot = to_categorical(y_train, num_classes=num_gesture_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=num_gesture_classes)

    K.clear_session()
    set_seeds()

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
        callbacks=[model_checkpoint_callback, early_stopping_callback],
        class_weight=class_weights_dict
    )

    model.load_weights(temp_checkpoint_filepath)
    
    y_pred_one_hot = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred_one_hot, axis=1)
    
    binary_f1, macro_f1, final_score = evaluate_metrics(
        y_val, y_pred_classes, label_encoder, non_target_gestures
    )
    
    print(f"折疊 {fold+1} 評估結果:")
    print(f"Binary F1: {binary_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Final Score: {final_score:.4f}")

    if not np.isnan(final_score):
        fold_scores.append(final_score)
        if final_score > best_final_score:
            print(f"新的最佳得分 {final_score:.4f}！ 儲存模型")
            best_final_score = final_score
            model.save(best_model_path)
    else:
        print("Final_score是 NaN，不計入平均。")
    
    os.remove(temp_checkpoint_filepath)

print("\n訓練流程完成。")
if len(fold_scores) > 0:
    print(f"所有有效折疊的最終得分: {np.round(fold_scores, 4)}")
    print(f"平均最終得分: {np.mean(fold_scores):.4f}")
else:
    print("沒有有效的折疊分數可供平均。")
print(f"\n最佳模型已儲存至: {best_model_path}，最終得分為: {best_final_score:.4f}")

# 載入和預處理測試資料
test_sequences, _, _, test_sequence_ids, _, _ = load_and_preprocess_data(TEST_CSV_PATH, TEST_DEMO_PATH, is_training=False)

# 進行預測並生成提交檔案，傳遞訓練時得到的 scaler 和 max_seq_len
print("Run predict_and_generate_submission")
predict_and_generate_submission(
    model_path=best_model_path,
    test_sequences=test_sequences,
    test_sequence_ids=test_sequence_ids,
    label_encoder=label_encoder,
    max_seq_len=max_seq_len,
    scaler=scaler
)