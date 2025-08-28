import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score
import os
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from numpy.fft import rfft
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 引入必要的層，用於構建複雜模型
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout, Bidirectional, LSTM, GRU, Dense, Concatenate, Lambda, Multiply, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Conv2D, Reshape, MaxPooling2D, TimeDistributed

# 調整學習率和正則化強度
LEARNING_RATE = 2e-4
L2_REG_STRENGTH = 0.005
DROPOUT_RATE_CNN = 0.5
DROPOUT_RATE_DENSE = 0.6
EARLY_STOPPING_PATIENCE = 20

# 全域變數用於模型集成，儲存每個模型的權重
model_scores = {}

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
manual_features = ['acc_mag', 'rot_x_rate', 'rot_y_rate', 'rot_z_rate']
categorical_features = ['orientation', 'behavior', 'phase']
numerical_features_ts = sensor_features + manual_features + ['tof_mean', 'tof_std']

# 計算時序特徵的總數，避免在預處理階段重複計算
NUM_TS_FEAT = len(numerical_features_ts) + len(categorical_features)

# =========================================================================
# 新增的物理特徵工程函式
# =========================================================================
def remove_gravity_from_acc(df):
    """移除加速度數據中的重力影響，以獲得線性加速度。"""
    print("正在移除加速度數據中的重力...")
    
    new_df = df[['sequence_id', 'acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']].copy()
    new_df[['rot_w', 'rot_x', 'rot_y', 'rot_z']] = new_df[['rot_w', 'rot_x', 'rot_y', 'rot_z']].fillna(0)
    
    processed_acc = []
    
    for seq_id, group in new_df.groupby('sequence_id'):
        acc_values = group[['acc_x', 'acc_y', 'acc_z']].values
        quat_values = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
        linear_accel = np.zeros_like(acc_values)
        gravity_world = np.array([0, 0, 9.81])

        for i in range(len(group)):
            if np.all(np.isclose(quat_values[i], 0)):
                linear_accel[i, :] = acc_values[i, :]
                continue
            
            try:
                rotation = R.from_quat(quat_values[i])
                gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
                linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
            except ValueError:
                linear_accel[i, :] = acc_values[i, :]
        
        processed_acc.append(pd.DataFrame(linear_accel, columns=['acc_x', 'acc_y', 'acc_z'], index=group.index))

    if processed_acc:
        processed_acc_df = pd.concat(processed_acc, axis=0)
        df.loc[processed_acc_df.index, ['acc_x', 'acc_y', 'acc_z']] = processed_acc_df.values
        
    return df

def generate_static_features(df):
    """
    使用 groupby().agg() 高效計算靜態統計特徵。
    此函數將計算時序數據的聚合統計量和頻域特徵。
    """
    print("正在計算靜態特徵...")
    
    current_static_features = sensor_features + manual_features + ['tof_mean', 'tof_std']
    
    agg_funcs = ['mean', 'std', 'min', 'max', 'median']
    
    existing_ts_features = [feat for feat in current_static_features if feat in df.columns]
    stats_df = df.groupby('sequence_id')[existing_ts_features].agg(agg_funcs)
    stats_df.columns = [f'{col}_{stat}' for col, stat in stats_df.columns]
    
    def get_fft_features_group(group):
        features = {}
        fft_subset_features = ['acc_x', 'acc_y', 'acc_z', 'acc_mag']
        for col in fft_subset_features:
            if col not in group.columns:
                continue
            fft_result = rfft(group[col].values)
            fft_amplitude = np.abs(fft_result)[:50] 
            features.update({f'{col}_fft_{i}': val for i, val in enumerate(fft_amplitude)})
        return pd.Series(features)
    
    existing_fft_features = list(set(['acc_x', 'acc_y', 'acc_z', 'acc_mag']) & set(df.columns))
    if existing_fft_features:
        fft_df = df.groupby('sequence_id')[existing_fft_features].apply(get_fft_features_group)
    else:
        fft_df = pd.DataFrame()
    
    print("正在計算類別特徵的靜態統計...")
    cat_stats = pd.DataFrame()
    grouped_df = df.groupby('sequence_id')
    
    for col in categorical_features:
        encoded_col = f'{col}_encoded'
        if encoded_col in df.columns:
            cat_stats[f'{col}_mode'] = grouped_df[encoded_col].apply(lambda x: x.mode()[0])
            cat_stats[f'{col}_changes'] = grouped_df[encoded_col].apply(lambda x: x.diff().ne(0).sum())

    print("正在計算 TOF 類別統計...")
    
    existing_cat_features = [col for col in categorical_features if col in df.columns]
    
    if existing_cat_features:
        grouped_by_cat = df.groupby(['sequence_id'] + existing_cat_features)
        tof_cat_mean_df = grouped_by_cat['tof_mean'].mean().unstack(level=existing_cat_features, fill_value=0)
        new_mean_cols = [f'tof_mean_by_{"__".join(map(str, col))}' for col in tof_cat_mean_df.columns]
        tof_cat_mean_df.columns = new_mean_cols
    else:
        tof_cat_mean_df = pd.DataFrame()

    static_features = pd.concat([stats_df, fft_df, cat_stats, tof_cat_mean_df], axis=1)
    
    return static_features

def load_and_preprocess_data(csv_path, demos_path, is_training=True, train_label_encoder=None, train_non_target_gestures=None):
    print(f"正在載入和處理資料: {csv_path}")
    df = pd.read_csv(csv_path)
    demographics_df = pd.read_csv(demos_path)
    df = pd.merge(df, demographics_df, on='subject', how='left')
    
    for col in sensor_features + tof_features + ['acc_mag', 'rot_x_rate', 'rot_y_rate', 'rot_z_rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    print("正在進行手動特徵工程...")

    df = remove_gravity_from_acc(df)
    
    new_ts_features_df = pd.DataFrame({
        'acc_mag': np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2),
        'rot_x_rate': df.groupby('sequence_id')['rot_x'].diff().fillna(0),
        'rot_y_rate': df.groupby('sequence_id')['rot_y'].diff().fillna(0),
        'rot_z_rate': df.groupby('sequence_id')['rot_z'].diff().fillna(0),
        'tof_mean': df[tof_features].mean(axis=1),
        'tof_std': df[tof_features].std(axis=1)
    }, index=df.index)

    print("正在處理類別型特徵...")
    new_cat_features_df = pd.DataFrame(index=df.index)
    for col in categorical_features:
        if col in df.columns:
            new_cat_features_df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        else:
            new_cat_features_df[f'{col}_encoded'] = 0
            
    if is_training:
        print("正在動態定義非目標手勢並進行標籤編碼...")
        non_target_gestures_df = df[df['sequence_type'] != 'Target']
        non_target_gestures = non_target_gestures_df['gesture'].unique().tolist()
        label_encoder = LabelEncoder()
        new_cat_features_df['gesture_encoded'] = label_encoder.fit_transform(df['gesture'])
        print(f"非目標手勢類別: {non_target_gestures}")
    else:
        label_encoder = train_label_encoder
        non_target_gestures = train_non_target_gestures
        new_cat_features_df['gesture_encoded'] = 0

    df = pd.concat([df, new_ts_features_df, new_cat_features_df], axis=1)

    print("正在處理缺失值...")
    df[tof_features] = df[tof_features].fillna(-999)
    all_numerical_features = sensor_features + manual_features + ['tof_mean', 'tof_std'] + demographic_features
    df[all_numerical_features] = df[all_numerical_features].fillna(0)
    
    static_features_df = generate_static_features(df)
    
    df = df.set_index('sequence_id').join(static_features_df).reset_index()
    df[static_features_df.columns] = df[static_features_df.columns].fillna(0)
    
    print("正在處理靜態特徵中的極端值...")
    for col in static_features_df.columns:
        if col in df.columns:
            q1 = df[col].quantile(0.05)
            q99 = df[col].quantile(0.95)
            df[col] = np.clip(df[col], q1, q99)

    print("正在將資料分組為序列...")
    time_series_features = numerical_features_ts + [f'{col}_encoded' for col in categorical_features if f'{col}_encoded' in df.columns]
    static_feature_names = list(static_features_df.columns)
    
    sequences = []
    labels = []
    subjects = []
    sequence_ids = []
    static_data = []
    tof_data = []

    grouped_sequences = df.groupby('sequence_id')
    for sequence_id, group in grouped_sequences:
        sequences.append(group[time_series_features].values)
        tof_data.append(group[tof_features].values)
        sequence_ids.append(sequence_id)
        if is_training:
            labels.append(group['gesture_encoded'].iloc[0])
            subjects.append(group['subject'].iloc[0])
        static_data.append(group[static_feature_names].iloc[0].values)

    return sequences, np.array(labels) if is_training else None, np.array(subjects) if is_training else None, np.array(sequence_ids), label_encoder, non_target_gestures, np.array(static_data), tof_data

# =========================================================================
# 新增的資料增強函式
# =========================================================================
def data_augmentation(data, augmentation_factor=0.15):
    """
    對時序資料進行資料增強。
    目前包含時間抖動 (Jittering) 和時間縮放 (Scaling)。
    **此版本僅對非填充值進行增強。**
    """
    aug_data = np.copy(data)

    # 找到所有非填充值（-999）的索引
    non_padded_indices = aug_data != -999.0

    # 時間抖動 (Jittering)：在非填充值上加入隨機雜訊
    jitter = np.random.normal(0, augmentation_factor, size=aug_data.shape)
    aug_data[non_padded_indices] += jitter[non_padded_indices]

    # 時間縮放 (Scaling)：乘以一個隨機縮放因子
    scaling_factor = np.random.uniform(1 - augmentation_factor, 1 + augmentation_factor)
    aug_data[non_padded_indices] *= scaling_factor

    return aug_data

# --- 自定義層次 ---
def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def attention_layer(inputs):
    # 這裡的 Dense 層輸出 shape 是 (None, seq_len, 1)
    score = Dense(1, activation='tanh')(inputs)
    
    # 修正: 明確指定 Lambda 層的輸出形狀
    # 移除最後一個維度 1，形狀變為 (None, seq_len)
    score = Lambda(lambda x: K.squeeze(x, axis=-1), output_shape=lambda s: (s[0], s[1]))(score)
    
    # softmax 輸出形狀不變
    weights = Activation('softmax')(score)
    
    # 修正: 明確指定 Lambda 層的輸出形狀
    # 增加一個維度 1，形狀變為 (None, seq_len, 1)
    weights = Lambda(lambda x: K.expand_dims(x, axis=-1), output_shape=lambda s: (s[0], s[1], 1))(weights)
    
    # 按權重相乘，形狀不變
    context = Multiply()([inputs, weights])
    
    # 修正: 明確指定 Lambda 層的輸出形狀
    # 沿著時間軸 (axis=1) 求和，形狀變為 (None, feature_dim)
    context = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(context)
    
    return context

def residual_block(x, filters, kernel_size, drop=0.2):
    """具有殘差連接的 CNN 區塊"""
    shortcut = x
    
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    
    # 調整捷徑的維度以匹配主路徑
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(drop)(x)
    return x

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Transformer 編碼器區塊"""
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(attn_output)
    res1 = x + inputs
    
    x = LayerNormalization(epsilon=1e-6)(res1)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res1

# =========================================================================
# 新增的單輸入模型，僅使用時序數據
# =========================================================================
def build_ts_only_model(input_shape, num_classes):
    print("建立單輸入模型架構...")
    ts_input = layers.Input(shape=input_shape, name='ts_only_input')
    
    x = residual_block(ts_input, 64, 5, drop=DROPOUT_RATE_CNN) 
    x = residual_block(x, 128, 5, drop=DROPOUT_RATE_CNN) 
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=DROPOUT_RATE_CNN) 
    x = attention_layer(x)

    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE_DENSE)(x)

    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE_DENSE)(x)

    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=ts_input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =========================================================================
# 升級後的多輸入模型，新增 TOF 的 2D CNN 分支
# =========================================================================
def build_multi_input_model(ts_input_shape, static_input_shape, tof_input_shape, num_classes):
    print("建立升級版多輸入模型架構...")
    
    # ------------------- 主輸入層 -------------------
    ts_input = layers.Input(shape=ts_input_shape, name='ts_input')
    tof_input = layers.Input(shape=tof_input_shape, name='tof_input')
    static_input = layers.Input(shape=static_input_shape, name='static_input')

    # ------------------- IMU 分支 -------------------
    imu_branch = residual_block(ts_input, 64, 5, drop=DROPOUT_RATE_CNN) 
    imu_branch = residual_block(imu_branch, 128, 5, drop=DROPOUT_RATE_CNN) 
    imu_branch = transformer_encoder(imu_branch, head_size=64, num_heads=4, ff_dim=128, dropout=DROPOUT_RATE_CNN) 
    imu_context = attention_layer(imu_branch)

    # ------------------- TOF 分支 (使用 TimeDistributed 2D CNN) -------------------
    tof_reshaped = Reshape((tof_input_shape[0], 5, 64, 1))(tof_input)
    
    tof_branch = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(tof_reshaped)
    tof_branch = TimeDistributed(BatchNormalization())(tof_branch)
    tof_branch = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(tof_branch) 
    tof_branch = TimeDistributed(Dropout(DROPOUT_RATE_CNN))(tof_branch)
    
    tof_branch = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(tof_branch)
    tof_branch = TimeDistributed(BatchNormalization())(tof_branch)
    tof_branch = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(tof_branch)
    tof_branch = TimeDistributed(Dropout(DROPOUT_RATE_CNN))(tof_branch)
    
    # 將 5D 輸出扁平化為 3D，以供 attention_layer 處理
    tof_branch = Reshape((-1, np.prod(tof_branch.shape[2:])))(tof_branch)
    tof_context = attention_layer(tof_branch)
    
    # ------------------- 靜態資料分支 -------------------
    static_branch = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH))(static_input)
    static_branch = BatchNormalization()(static_branch)
    static_branch = Dropout(DROPOUT_RATE_DENSE)(static_branch)

    # ------------------- 合併與輸出 -------------------
    merged = Concatenate()([imu_context, tof_context, static_branch]) 
    
    merged = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH))(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(DROPOUT_RATE_DENSE)(merged)

    merged = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH))(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(DROPOUT_RATE_DENSE)(merged)

    output = Dense(num_classes, activation='softmax')(merged)
    
    model = Model(inputs=[ts_input, tof_input, static_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_metrics(y_true, y_pred, label_encoder, non_target_gestures):
    """計算競賽所需的二元 F1 和宏觀 F1 分數。"""
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    is_target_true = np.isin(y_true_labels, non_target_gestures, invert=True).astype(int)
    is_target_pred = np.isin(y_pred_labels, non_target_gestures, invert=True).astype(int)
    binary_f1 = f1_score(is_target_true, is_target_pred, zero_division=0)
    
    y_true_macro = np.where(np.isin(y_true_labels, non_target_gestures), 'non_target', y_true_labels)
    y_pred_macro = np.where(np.isin(y_pred_labels, non_target_gestures), 'non_target', y_pred_labels)
    
    all_classes = np.unique(np.concatenate([y_true_macro, y_pred_macro]))
    macro_f1 = f1_score(y_true_macro, y_pred_macro, labels=all_classes, average='macro', zero_division=0)
    
    final_score = (binary_f1 + macro_f1) / 2
    return binary_f1, macro_f1, final_score

def predict_and_generate_submission_ensemble(
    model_paths, model_scores, test_sequences, test_static_data, test_tof_data, test_sequence_ids, label_encoder, max_seq_len, ts_scaler, static_scaler
):
    """
    使用加權集成學習模型進行預測，並生成 submission.parquet 檔案。
    """
    print("\n--- 正在進行加權集成學習預測 ---")
    
    padded_test_sequences = pad_sequences(test_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    test_tof_padded = pad_sequences(test_tof_data, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    
    original_shape = padded_test_sequences.shape
    padded_test_sequences_scaled = ts_scaler.transform(
        padded_test_sequences.reshape(-1, original_shape[2])
    ).reshape(original_shape[0], original_shape[1], original_shape[2])
    
    test_static_data_scaled = static_scaler.transform(test_static_data)

    ensemble_predictions = []
    
    # 計算權重
    total_score = sum(model_scores.values())
    if total_score == 0:
        weights = {name: 1.0 / len(model_scores) for name in model_scores}
    else:
        weights = {name: score / total_score for name, score in model_scores.items()}
    
    for i, model_info in enumerate(model_paths):
        model_name, model_path = model_info
        try:
            print(f"載入模型 {i+1}/{len(model_paths)} ({model_name}): {model_path}，權重: {weights.get(model_name, 0.0):.4f}")
            custom_objects = {
                'squeeze_last_axis': squeeze_last_axis, 
                'expand_last_axis': expand_last_axis, 
                'attention_layer': attention_layer, 
                'residual_block': residual_block,
                'transformer_encoder': transformer_encoder,
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            if 'ts_only' in model_name:
                predictions = model.predict(padded_test_sequences_scaled, verbose=0)
            elif 'multi_input' in model_name:
                predictions = model.predict({
                    'ts_input': padded_test_sequences_scaled, 
                    'tof_input': test_tof_padded,
                    'static_input': test_static_data_scaled
                }, verbose=0)
                
            ensemble_predictions.append(predictions * weights.get(model_name, 0.0))
            K.clear_session()
        except Exception as e:
            print(f"無法載入模型 {model_path}：{e}")
            continue

    if not ensemble_predictions:
        print("沒有可用的模型進行預測。")
        return

    averaged_predictions = np.sum(ensemble_predictions, axis=0)
    predicted_classes = np.argmax(averaged_predictions, axis=1)
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
def train_and_predict_models():
    train_sequences, y_gesture, subjects, _, label_encoder, non_target_gestures, static_data, tof_data = load_and_preprocess_data(TRAIN_CSV_PATH, TRAIN_DEMO_PATH)

    print("正在打亂動作序列的順序...")
    combined = list(zip(train_sequences, y_gesture, subjects, static_data, tof_data))
    np.random.shuffle(combined)
    train_sequences[:], y_gesture[:], subjects[:], static_data[:], tof_data[:] = zip(*combined)

    sequence_lengths = [len(s) for s in train_sequences]
    max_seq_len = int(np.percentile(sequence_lengths, 99))
    print(f"最大序列長度: {max_seq_len}")

    padded_ts = pad_sequences(
        train_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post'
    )
    
    padded_tof_data = pad_sequences(
        tof_data, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post'
    )

    num_gesture_classes = len(label_encoder.classes_)
    print(f"總共 {num_gesture_classes} 個動作類別。")
    
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
    
    ensemble_model_paths = []
    
    ts_scaler = StandardScaler()
    static_scaler = StandardScaler()
    
    for fold, (train_index, val_index) in enumerate(gkf.split(padded_ts, y_gesture, groups=subjects)):
        print(f"\n- 折疊 (Fold) {fold+1}/{n_sp}")
        
        X_train_ts, X_val_ts = padded_ts[train_index], padded_ts[val_index]
        X_train_tof, X_val_tof = padded_tof_data[train_index], padded_tof_data[val_index]
        X_train_static, X_val_static = static_data[train_index], static_data[val_index]
        y_train, y_val = y_gesture[train_index], y_gesture[val_index]

        # 資料增強
        X_train_ts_aug = data_augmentation(X_train_ts)
        X_train_tof_aug = data_augmentation(X_train_tof)
        
        ts_data_to_fit = X_train_ts_aug.reshape(-1, X_train_ts_aug.shape[2])
        static_data_to_fit = X_train_static
        
        ts_scaler.fit(ts_data_to_fit)
        static_scaler.fit(static_data_to_fit)
        
        X_train_ts_scaled = ts_scaler.transform(ts_data_to_fit).reshape(X_train_ts_aug.shape)
        X_val_ts_scaled = ts_scaler.transform(X_val_ts.reshape(-1, X_val_ts.shape[2])).reshape(X_val_ts.shape)
        
        X_train_static_scaled = static_scaler.transform(X_train_static)
        X_val_static_scaled = static_scaler.transform(X_val_static)
        
        y_train_one_hot = to_categorical(y_train, num_classes=num_gesture_classes)
        y_val_one_hot = to_categorical(y_val, num_classes=num_gesture_classes)

        for model_name_suffix in ['ts_only', 'multi_input']:
            K.clear_session()
            set_seeds()
            
            model = None
            train_data, val_data = None, None
            
            if model_name_suffix == 'ts_only':
                model = build_ts_only_model(X_train_ts_scaled.shape[1:], num_gesture_classes)
                train_data, val_data = X_train_ts_scaled, X_val_ts_scaled
            elif model_name_suffix == 'multi_input':
                model = build_multi_input_model(
                    X_train_ts_scaled.shape[1:], 
                    X_train_static_scaled.shape[1:], 
                    X_train_tof_aug.shape[1:], 
                    num_gesture_classes
                )
                train_data = {'ts_input': X_train_ts_scaled, 'tof_input': X_train_tof_aug, 'static_input': X_train_static_scaled}
                val_data = {'ts_input': X_val_ts_scaled, 'tof_input': X_val_tof, 'static_input': X_val_static_scaled}

            print(f"--- 訓練模型: {model_name_suffix} ---")
            temp_checkpoint_filepath = os.path.join(WORKING_DIR, f'best_model_fold_{fold+1}_{model_name_suffix}.h5')
            model_checkpoint_callback = ModelCheckpoint(
                filepath=temp_checkpoint_filepath, 
                monitor='val_loss', 
                mode='min', 
                save_best_only=True, 
                verbose=0
            )
            early_stopping_callback = EarlyStopping(
                monitor='val_loss', 
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
            
            reduce_lr_callback = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=8,
                min_lr=1e-6
            )

            history = model.fit(
                train_data, 
                y_train_one_hot, 
                epochs=100, 
                batch_size=64,
                validation_data=(val_data, y_val_one_hot), 
                verbose=1,
                callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback],
                class_weight=class_weights_dict
            )
            
            model.load_weights(temp_checkpoint_filepath)
            
            predictions = model.predict(val_data)
            predicted_classes = np.argmax(predictions, axis=1)
            binary_f1, macro_f1, final_score = evaluate_metrics(
                y_val, predicted_classes, label_encoder, non_target_gestures
            )
            
            print(f"模型 {model_name_suffix} 在折疊 {fold+1} 的驗證得分: Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")

            if not np.isnan(final_score):
                model_name = f'{model_name_suffix}_fold_{fold+1}'
                ensemble_model_paths.append((model_name, temp_checkpoint_filepath))
                model_scores[model_name] = final_score

    print("\n訓練流程完成。")
    print(f"\n最佳模型已儲存至: {ensemble_model_paths}")

    test_sequences, _, _, test_sequence_ids, _, _, test_static_data, test_tof_data = load_and_preprocess_data(TEST_CSV_PATH, TEST_DEMO_PATH, is_training=False, train_label_encoder=label_encoder, train_non_target_gestures=non_target_gestures)

    print("執行 predict_and_generate_submission_ensemble")
    predict_and_generate_submission_ensemble(
        model_paths=ensemble_model_paths,
        model_scores=model_scores,
        test_sequences=test_sequences,
        test_static_data=test_static_data,
        test_tof_data=test_tof_data,
        test_sequence_ids=test_sequence_ids,
        label_encoder=label_encoder,
        max_seq_len=max_seq_len,
        ts_scaler=ts_scaler,
        static_scaler=static_scaler
    )

if __name__ == '__main__':
    train_and_predict_models()