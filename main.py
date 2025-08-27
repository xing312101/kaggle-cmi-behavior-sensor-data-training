import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score
import os
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from numpy.fft import rfft

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
# 增加手動特徵：角速度的變化率
angular_velocity_features = ['rot_x_rate', 'rot_y_rate', 'rot_z_rate']


numerical_features = sensor_features + tof_features + demographic_features + manual_features + angular_velocity_features
categorical_features = ['orientation', 'behavior', 'phase']

def generate_static_features(df, numerical_features):
    """
    使用 groupby().agg() 高效計算靜態統計特徵。
    """
    # 選擇更小的特徵子集進行靜態計算
    subset_features = sensor_features + manual_features
    print(f"正在對以下特徵計算靜態量：{subset_features}")
    
    agg_funcs = ['mean', 'std', 'min', 'max', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    stats_df = df.groupby('sequence_id')[subset_features].agg(agg_funcs)
    stats_df.columns = [f'{col}_{stat}' for col, stat in stats_df.columns]
    
    # 頻域特徵計算仍使用 apply，因為沒有更好的內建函式
    def get_fft_features_group(group):
        features = {}
        for col in subset_features: # 這裡也使用更小的子集
            fft_result = rfft(group[col].values)
            fft_amplitude = np.abs(fft_result)[:10] 
            features.update({f'{col}_fft_{i}': val for i, val in enumerate(fft_amplitude)})
        return pd.Series(features)
    
    fft_df = df.groupby('sequence_id')[subset_features].apply(get_fft_features_group)
    
    static_features = pd.concat([stats_df, fft_df], axis=1)
    return static_features

def load_and_preprocess_data(csv_path, demos_path, is_training=True):
    """
    載入並預處理資料，將特徵工程與資料分組分離。
    """
    print(f"正在載入和處理資料: {csv_path}")
    df = pd.read_csv(csv_path)
    demographics_df = pd.read_csv(demos_path)
    df = pd.merge(df, demographics_df, on='subject', how='left')
    
    print("正在進行手動特徵工程...")
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # 增加角速度的變化率特徵
    df['rot_x_rate'] = df.groupby('sequence_id')['rot_x'].diff().fillna(0)
    df['rot_y_rate'] = df.groupby('sequence_id')['rot_y'].diff().fillna(0)
    df['rot_z_rate'] = df.groupby('sequence_id')['rot_z'].diff().fillna(0)
    
    print("正在處理類別型特徵...")
    for col in categorical_features:
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        else:
            df[f'{col}_encoded'] = 0

    print("正在處理缺失值...")
    df[numerical_features] = df[numerical_features].fillna(0)

    # 產生靜態特徵並合併
    static_features_df = generate_static_features(df, numerical_features)
    df = df.set_index('sequence_id').join(static_features_df).reset_index()

    non_target_gestures = []
    label_encoder = None
    if is_training:
        print("正在動態定義非目標手勢並進行標籤編碼...")
        non_target_gestures_df = df[df['sequence_type'] != 'Target']
        non_target_gestures = non_target_gestures_df['gesture'].unique().tolist()
        label_encoder = LabelEncoder()
        df['gesture_encoded'] = label_encoder.fit_transform(df['gesture'])
        print(f"非目標手勢類別: {non_target_gestures}")
    
    print("正在將資料分組為序列...")
    time_series_features = numerical_features + [f'{col}_encoded' for col in categorical_features]
    static_feature_names = list(static_features_df.columns)
    
    sequences = []
    labels = []
    subjects = []
    sequence_ids = []
    static_data = []

    grouped_sequences = df.groupby('sequence_id')
    for sequence_id, group in grouped_sequences:
        sequences.append(group[time_series_features].values)
        sequence_ids.append(sequence_id)
        if is_training:
            labels.append(group['gesture_encoded'].iloc[0])
            subjects.append(group['subject'].iloc[0])
        static_data.append(group[static_feature_names].iloc[0].values)

    return sequences, np.array(labels) if is_training else None, np.array(subjects) if is_training else None, np.array(sequence_ids), label_encoder, non_target_gestures, np.array(static_data)


def build_model(ts_input_shape, static_input_shape, num_classes):
    """使用 Functional API 建立多輸入模型。"""
    print("建立多輸入模型架構...")
    
    # 時間序列輸入分支
    ts_input = Input(shape=ts_input_shape, name='ts_input')
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(ts_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.2)(x)

    # 靜態特徵輸入分支
    static_input = Input(shape=static_input_shape, name='static_input')
    y = Dense(32, activation='relu')(static_input)
    y = Dropout(0.2)(y)
    
    # 合併兩個分支
    combined = tf.keras.layers.concatenate([x, y])
    
    # 輸出層
    output = Dense(num_classes, activation='softmax')(combined)
    
    model = Model(inputs=[ts_input, static_input], outputs=output)
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
    model_path, test_sequences, test_static_data, test_sequence_ids, label_encoder, max_seq_len, scaler
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

    predictions = model.predict({'ts_input': padded_test_sequences, 'static_input': test_static_data}, verbose=1)
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

train_sequences, y_gesture, subjects, _, label_encoder, non_target_gestures, static_data = load_and_preprocess_data(TRAIN_CSV_PATH, TRAIN_DEMO_PATH)

print("正在打亂動作序列的順序...")
combined = list(zip(train_sequences, y_gesture, subjects, static_data))
np.random.shuffle(combined)
train_sequences[:], y_gesture[:], subjects[:], static_data[:] = zip(*combined)

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
print(f"輸入時間序列數據形狀: {X.shape}")
print(f"輸入靜態數據形狀: {static_data.shape}")

best_model_path = os.path.join(WORKING_DIR, 'best_overall_model.h5')
best_final_score = -1.0
fold_scores = []

print("\n正在計算類別權重...")
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_gesture),
    y=y_gesture
)
class_weights_dict = dict(enumerate(class_weights))
print("計算出的類別權重：", class_weights_dict)

print("\n開始交叉驗證訓練...")
n_sp = 5 
gkf = GroupKFold(n_splits=n_sp)
for fold, (train_index, val_index) in enumerate(gkf.split(X, y_gesture, groups=subjects)):
    print(f"\n- 折疊 (Fold) {fold+1}/{n_sp}")
    
    X_train_ts, X_val_ts = X[train_index], X[val_index]
    X_train_static, X_val_static = static_data[train_index], static_data[val_index]
    y_train, y_val = y_gesture[train_index], y_gesture[val_index]
    
    y_train_one_hot = to_categorical(y_train, num_classes=num_gesture_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=num_gesture_classes)

    K.clear_session()
    set_seeds()

    model = build_model(X_train_ts.shape[1:], X_train_static.shape[1:], num_gesture_classes)
    
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
        {'ts_input': X_train_ts, 'static_input': X_train_static}, 
        y_train_one_hot, 
        epochs=50, 
        batch_size=256,
        validation_data=({'ts_input': X_val_ts, 'static_input': X_val_static}, y_val_one_hot), 
        verbose=1,
        callbacks=[model_checkpoint_callback, early_stopping_callback],
        class_weight=class_weights_dict
    )

    model.load_weights(temp_checkpoint_filepath)
    
    y_pred_one_hot = model.predict({'ts_input': X_val_ts, 'static_input': X_val_static})
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

test_sequences, _, _, test_sequence_ids, _, _, test_static_data = load_and_preprocess_data(TEST_CSV_PATH, TEST_DEMO_PATH, is_training=False)

print("執行 predict_and_generate_submission")
predict_and_generate_submission(
    model_path=best_model_path,
    test_sequences=test_sequences,
    test_static_data=test_static_data,
    test_sequence_ids=test_sequence_ids,
    label_encoder=label_encoder,
    max_seq_len=max_seq_len,
    scaler=scaler
)