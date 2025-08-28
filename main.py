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
from tensorflow.keras import regularizers
from scipy.stats import skew, kurtosis
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout, Bidirectional, LSTM, GRU, Dense, Concatenate, Lambda, Multiply, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.mixed_precision import set_global_policy

# 啟用混合精度訓練以加速，請確保你的 GPU 支援
# set_global_policy('mixed_float16')

N_COMPONENTS = 20

# =========================================================================
# 設定隨機種子以確保可重現性
# =========================================================================
def set_seeds(seed=42):
    """設定所有必要的隨機種子以確保實驗的可重現性。"""
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
angular_velocity_features = ['rot_x_rate', 'rot_y_rate', 'rot_z_rate']

numerical_features_ts = sensor_features + manual_features + angular_velocity_features
categorical_features = ['orientation', 'behavior', 'phase']

# =========================================================================
# 新增的物理特徵工程函式
# =========================================================================
def remove_gravity_from_acc(df):
    """
    移除加速度數據中的重力影響，以獲得線性加速度。
    此函數已進行向量化優化，避免了繁重的 groupby().apply()。
    """
    print("正在移除加速度數據中的重力...")
    
    # 填充缺失值並轉為 NumPy 陣列
    acc_values = df[['acc_x', 'acc_y', 'acc_z']].fillna(0).values
    quat_values = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].fillna(0).values
    sequence_ids = df['sequence_id'].values
    
    linear_accel = np.copy(acc_values)
    gravity_world = np.array([0, 0, 9.81])

    # 批次處理每個序列
    for seq_id in np.unique(sequence_ids):
        indices = np.where(sequence_ids == seq_id)[0]
        
        # 僅處理非零四元數
        non_zero_indices = np.where(np.any(quat_values[indices] != 0, axis=1))[0]
        if non_zero_indices.size > 0:
            quats_to_process = quat_values[indices[non_zero_indices]]
            
            try:
                # 創建 Rotation 對象並應用旋轉
                rotations = R.from_quat(quats_to_process)
                gravity_sensor_frame = rotations.apply(gravity_world, inverse=True)
                linear_accel[indices[non_zero_indices]] -= gravity_sensor_frame
            except ValueError:
                # 處理無效的四元數
                pass

    df['acc_x'] = linear_accel[:, 0]
    df['acc_y'] = linear_accel[:, 1]
    df['acc_z'] = linear_accel[:, 2]
        
    return df

def generate_static_features(df):
    """
    使用 groupby().agg() 高效計算靜態統計特徵。
    此函數將計算時序數據的聚合統計量和頻域特徵。
    """
    print("正在計算靜態特徵...")
    
    current_static_features = sensor_features + manual_features + angular_velocity_features + ['tof_mean', 'tof_std']
    
    agg_funcs = ['mean', 'std', 'min', 'max', 'median']
    
    existing_ts_features = [feat for feat in current_static_features if feat in df.columns]
    stats_df = df.groupby('sequence_id')[existing_ts_features].agg(agg_funcs)
    stats_df.columns = [f'{col}_{stat}' for col, stat in stats_df.columns]
    
    def get_fft_features_group(group):
        features = {}
        # 對 IMU 和 acc_mag 進行 FFT
        fft_subset_features = ['acc_x', 'acc_y', 'acc_z', 'acc_mag', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        for col in fft_subset_features:
            if col not in group.columns:
                continue
            # 使用 rfft 以獲得實數訊號的單側頻譜
            fft_result = rfft(group[col].values)
            fft_amplitude = np.abs(fft_result)[:20] # 提取前 20 個頻率分量
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
            # 使用 try-except 處理 groupby mode() 可能的錯誤
            try:
                cat_stats[f'{col}_mode'] = grouped_df[encoded_col].apply(lambda x: x.mode()[0])
            except IndexError:
                cat_stats[f'{col}_mode'] = 0
            cat_stats[f'{col}_changes'] = grouped_df[encoded_col].apply(lambda x: x.diff().ne(0).sum())

    print("正在計算 TOF 類別統計...")
    existing_cat_features = [col for col in categorical_features if col in df.columns]
    
    if existing_cat_features:
        grouped_by_cat = df.groupby(['sequence_id'] + existing_cat_features)
        try:
            tof_cat_mean_df = grouped_by_cat['tof_mean'].mean().unstack(level=existing_cat_features, fill_value=0)
            new_mean_cols = [f'tof_mean_by_{"__".join(map(str, col))}' for col in tof_cat_mean_df.columns]
            tof_cat_mean_df.columns = new_mean_cols
        except:
            tof_cat_mean_df = pd.DataFrame()
    else:
        tof_cat_mean_df = pd.DataFrame()

    static_features = pd.concat([stats_df, fft_df, cat_stats, tof_cat_mean_df], axis=1)
    
    return static_features

def load_and_preprocess_data(csv_path, demos_path, is_training=True):
    print(f"正在載入和處理資料: {csv_path}")
    df = pd.read_csv(csv_path)
    demographics_df = pd.read_csv(demos_path)
    df = pd.merge(df, demographics_df, on='subject', how='left')
    
    for col in sensor_features + tof_features + ['acc_mag', 'rot_x_rate', 'rot_y_rate', 'rot_z_rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    print("正在進行手動特徵工程...")
    
    new_cols_df = pd.DataFrame(index=df.index)
    
    new_cols_df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    new_cols_df['tof_mean'] = df[tof_features].mean(axis=1)
    new_cols_df['tof_std'] = df[tof_features].std(axis=1)
    
    for col in categorical_features:
        if col in df.columns:
            new_cols_df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        else:
            new_cols_df[f'{col}_encoded'] = 0
            
    new_cols_df['rot_x_rate'] = df.groupby('sequence_id')['rot_x'].diff().fillna(0)
    new_cols_df['rot_y_rate'] = df.groupby('sequence_id')['rot_y'].diff().fillna(0)
    new_cols_df['rot_z_rate'] = df.groupby('sequence_id')['rot_z'].diff().fillna(0)
    
    non_target_gestures = []
    if is_training:
        print("正在動態定義非目標手勢並進行標籤編碼...")
        non_target_gestures_df = df[df['sequence_type'] != 'Target']
        non_target_gestures = non_target_gestures_df['gesture'].unique().tolist()
        label_encoder = LabelEncoder()
        new_cols_df['gesture_encoded'] = label_encoder.fit_transform(df['gesture'])
        print(f"非目標手勢類別: {non_target_gestures}")
    else:
        new_cols_df['gesture_encoded'] = 0
    
    df = pd.concat([df, new_cols_df], axis=1)

    print("正在處理缺失值...")
    df[tof_features] = df[tof_features].fillna(-999)
    all_numerical_features = numerical_features_ts + ['tof_mean', 'tof_std'] + demographic_features
    df[all_numerical_features] = df[all_numerical_features].fillna(0)
    
    df = remove_gravity_from_acc(df)
    
    static_features_df = generate_static_features(df)
    
    df = df.set_index('sequence_id').join(static_features_df).reset_index()
    df[static_features_df.columns] = df[static_features_df.columns].fillna(0)
    
    if not is_training:
        label_encoder = None

    print("正在將資料分組為序列...")
    time_series_features = numerical_features_ts + ['tof_mean', 'tof_std'] + [f'{col}_encoded' for col in categorical_features if f'{col}_encoded' in df.columns]
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

def build_ts_only_model(ts_input_shape, num_classes, learning_rate):
    print("建立單一時間序列模型架構...")
    ts_input = layers.Input(shape=ts_input_shape, name='ts_input')
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(ts_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(24))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=ts_input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 新增的自定義層次 ---
def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def attention_layer(inputs):
    """
    自定義注意力層，計算加權平均。
    """
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)
    return context

def residual_block(x, filters, kernel_size, drop=0.2):
    """具有殘差連接的 CNN 區塊"""
    shortcut = x
    
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(drop)(x)
    return x

def positional_encoding(sequence_length, d_model):
    """
    生成位置編碼。
    """
    position = np.arange(sequence_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((sequence_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = tf.constant(pe, dtype=tf.float32)
    return pe

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Transformer 編碼器區塊 (已加入位置編碼)"""
    seq_len = K.int_shape(inputs)[1]
    d_model = K.int_shape(inputs)[2]
    pos_encoding = positional_encoding(seq_len, d_model)
    inputs_with_pos = inputs + pos_encoding
    
    x = LayerNormalization(epsilon=1e-6)(inputs_with_pos)
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(attn_output)
    res1 = x + inputs
    
    x = LayerNormalization(epsilon=1e-6)(res1)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res1

# =========================================================================
# 升級後的多輸入模型 (已簡化和加入 L2 正則化)
# =========================================================================
def build_multi_input_model(ts_input_shape, static_input_shape, num_classes, learning_rate):
    print("建立升級版多輸入模型架構...")
    
    ts_input = layers.Input(shape=ts_input_shape, name='ts_input')
    
    num_imu_features = len(sensor_features) + len(manual_features) + len(angular_velocity_features) + len([f'{col}_encoded' for col in categorical_features])
    
    # IMU 分支
    imu_branch = Lambda(lambda t: t[:, :, :num_imu_features], name='imu_splitter')(ts_input)
    imu_branch = residual_block(imu_branch, 64, 5, drop=0.4) 
    imu_branch = residual_block(imu_branch, 128, 5, drop=0.4)
    imu_branch = transformer_encoder(imu_branch, head_size=64, num_heads=4, ff_dim=128, dropout=0.4)
    
    # TOF 分支
    tof_branch = Lambda(lambda t: t[:, :, num_imu_features:], name='tof_splitter')(ts_input)
    tof_branch = Conv1D(64, 3, activation='relu', padding='same')(tof_branch)
    tof_branch = BatchNormalization()(tof_branch)
    tof_branch = MaxPooling1D(2)(tof_branch)
    tof_branch = Dropout(0.4)(tof_branch) 
    tof_branch = Conv1D(128, 3, activation='relu', padding='same')(tof_branch)
    tof_branch = BatchNormalization()(tof_branch)
    tof_branch = MaxPooling1D(2)(tof_branch)
    tof_branch = Dropout(0.4)(tof_branch) 
    
    # 加入注意力層
    imu_context = attention_layer(imu_branch)
    tof_context = attention_layer(tof_branch)
    
    # 靜態資料分支
    static_input = layers.Input(shape=static_input_shape, name='static_input')
    static_branch = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(static_input)
    static_branch = layers.BatchNormalization()(static_branch)
    static_branch = layers.Dropout(0.5)(static_branch) 

    # 合併與輸出
    merged = Concatenate()([imu_context, tof_context, static_branch])
    
    merged = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)

    merged = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)

    output = Dense(num_classes, activation='softmax')(merged)
    
    model = Model(inputs=[ts_input, static_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
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

# =========================================================================
# 修正後的資料增強函式
# =========================================================================
def data_augmentation(data, augmentation_factor=0.2):
    """
    對時序資料進行資料增強。
    目前包含時間抖動 (Jittering) 和時間縮放 (Scaling)，並避免對填充值操作。
    """
    aug_data = np.copy(data)
    
    non_padded_indices = aug_data != -999.0
    
    jitter = np.random.normal(0, augmentation_factor, size=aug_data.shape)
    aug_data[non_padded_indices] += jitter[non_padded_indices]
    
    scaling_factor = np.random.uniform(1 - augmentation_factor, 1 + augmentation_factor, size=(aug_data.shape[0], 1, 1))
    
    aug_data = aug_data * scaling_factor
    aug_data[~non_padded_indices] = -999.0
    
    return aug_data

def predict_and_generate_submission_ensemble(
    model_paths, test_sequences, test_static_data, test_sequence_ids, label_encoder, max_seq_len, ts_scaler, static_scaler, pca, time_series_features
):
    """
    使用集成學習模型進行預測，並生成 submission.parquet 檔案。
    """
    print("\n--- 正在進行集成學習預測 ---")
    
    num_ts_feat = len(time_series_features)
    test_ts_data_base = [seq[:, :num_ts_feat] for seq in test_sequences]
    
    num_tof_feat = len(tof_features)
    test_tof_data_raw = [seq[:, num_ts_feat:] for seq in test_sequences]
    
    test_tof_padded = pad_sequences(test_tof_data_raw, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post', value=-999.0)
    
    test_tof_pca = pca.transform(test_tof_padded.reshape(-1, num_tof_feat)).reshape(test_tof_padded.shape[0], test_tof_padded.shape[1], -1)

    padded_test_sequences_base = pad_sequences(test_ts_data_base, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    original_shape = padded_test_sequences_base.shape
    padded_test_sequences_scaled = ts_scaler.transform(
        padded_test_sequences_base.reshape(-1, original_shape[2])
    ).reshape(original_shape[0], original_shape[1], original_shape[2])
    
    padded_test_sequences_final = np.concatenate([padded_test_sequences_scaled, test_tof_pca], axis=-1)

    test_static_data_scaled = static_scaler.transform(test_static_data)

    ensemble_predictions = []
    
    for i, model_info in enumerate(model_paths):
        model_name, model_path = model_info
        try:
            print(f"載入模型 {i+1}/{len(model_paths)} ({model_name}): {model_path}")
            custom_objects = {
                'squeeze_last_axis': squeeze_last_axis, 
                'expand_last_axis': expand_last_axis, 
                'attention_layer': attention_layer, 
                'residual_block': residual_block,
                'transformer_encoder': transformer_encoder,
                'positional_encoding': positional_encoding
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            if model_name == 'ts_only':
                predictions = model.predict(padded_test_sequences_final, verbose=0)
            elif model_name == 'multi_input':
                predictions = model.predict({'ts_input': padded_test_sequences_final, 'static_input': test_static_data_scaled}, verbose=0)
                
            ensemble_predictions.append(predictions)
            K.clear_session()
        except Exception as e:
            print(f"無法載入模型 {model_path}：{e}")
            continue

    if not ensemble_predictions:
        print("沒有可用的模型進行預測。")
        return

    averaged_predictions = np.mean(ensemble_predictions, axis=0)
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
    
    num_gesture_classes = len(label_encoder.classes_)
    print(f"偵測到 {num_gesture_classes} 個手勢類別。")
    
    print("正在打亂動作序列的順序...")
    combined = list(zip(train_sequences, y_gesture, subjects, static_data, tof_data))
    np.random.shuffle(combined)
    train_sequences[:], y_gesture[:], subjects[:], static_data[:], tof_data[:] = zip(*combined)

    sequence_lengths = [len(s) for s in train_sequences]
    max_seq_len = int(np.percentile(sequence_lengths, 99))
    print(f"最大序列長度: {max_seq_len}")
    
    print("\n正在對 TOF 數據進行 PCA 降維...")
    padded_tof_data = pad_sequences(
        tof_data, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post'
    )
    tof_data_flat = padded_tof_data.reshape(-1, len(tof_features))
    
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(tof_data_flat[tof_data_flat[:, 0] != -999])
    
    print("正在計算類別權重...")
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_gesture),
        y=y_gesture
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("計算出的類別權重：", class_weights_dict)

    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"使用分散式策略，偵測到的裝置數量：{strategy.num_replicas_in_sync}")
    except ValueError:
        print("未偵測到多個 GPU，使用單一 GPU 或 CPU。")
        strategy = tf.keras.distribute.get_strategy()

    print("\n開始交叉驗證訓練...")
    n_sp = 5 
    gkf = GroupKFold(n_splits=n_sp)
    
    ensemble_model_paths = []
    ts_scaler = StandardScaler()
    static_scaler = StandardScaler()
    
    print("正在對訓練集進行標準化擬合...")
    all_padded_ts = pad_sequences(train_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    ts_scaler.fit(all_padded_ts.reshape(-1, all_padded_ts.shape[2]))
    static_scaler.fit(static_data)
    
    for fold, (train_index, val_index) in enumerate(gkf.split(all_padded_ts, y_gesture, groups=subjects)):
        print(f"\n- 折疊 (Fold) {fold+1}/{n_sp}")
        
        X_train_ts_raw, X_val_ts_raw = all_padded_ts[train_index], all_padded_ts[val_index]
        X_train_tof_raw, X_val_tof_raw = padded_tof_data[train_index], padded_tof_data[val_index]
        X_train_static, X_val_static = static_data[train_index], static_data[val_index]
        y_train, y_val = y_gesture[train_index], y_gesture[val_index]

        X_train_ts_aug = data_augmentation(X_train_ts_raw)
        X_train_tof_aug = data_augmentation(X_train_tof_raw)

        X_train_tof_pca = pca.transform(X_train_tof_aug.reshape(-1, len(tof_features))).reshape(X_train_tof_aug.shape[0], X_train_tof_aug.shape[1], N_COMPONENTS)
        X_val_tof_pca = pca.transform(X_val_tof_raw.reshape(-1, len(tof_features))).reshape(X_val_tof_raw.shape[0], X_val_tof_raw.shape[1], N_COMPONENTS)
        
        X_train_ts_scaled = ts_scaler.transform(X_train_ts_aug.reshape(-1, X_train_ts_aug.shape[2])).reshape(X_train_ts_aug.shape)
        X_val_ts_scaled = ts_scaler.transform(X_val_ts_raw.reshape(-1, X_val_ts_raw.shape[2])).reshape(X_val_ts_raw.shape)
        
        X_train_static_scaled = static_scaler.transform(X_train_static)
        X_val_static_scaled = static_scaler.transform(X_val_static)
        
        X_train_ts_final = np.concatenate([X_train_ts_scaled, X_train_tof_pca], axis=-1)
        X_val_ts_final = np.concatenate([X_val_ts_scaled, X_val_tof_pca], axis=-1)
        
        y_train_one_hot = to_categorical(y_train, num_classes=num_gesture_classes)
        y_val_one_hot = to_categorical(y_val, num_classes=num_gesture_classes)

        for model_name in ['ts_only', 'multi_input']:
            K.clear_session()
            set_seeds()
            
            # 根據模型名稱設定獨立的學習率
            if model_name == 'ts_only':
                model_learning_rate = 1e-4
            else: # multi_input
                model_learning_rate = 5e-5
            
            with strategy.scope():
                model = None
                if model_name == 'ts_only':
                    model = build_ts_only_model(X_train_ts_final.shape[1:], num_gesture_classes, learning_rate=model_learning_rate)
                elif model_name == 'multi_input':
                    model = build_multi_input_model(X_train_ts_final.shape[1:], X_train_static_scaled.shape[1:], num_gesture_classes, learning_rate=model_learning_rate)
            
            train_data, val_data = None, None
            
            if model_name == 'ts_only':
                train_data, val_data = X_train_ts_final, X_val_ts_final
            elif model_name == 'multi_input':
                train_data = {'ts_input': X_train_ts_final, 'static_input': X_train_static_scaled}
                val_data = {'ts_input': X_val_ts_final, 'static_input': X_val_static_scaled}

            print(f"--- 訓練模型: {model_name}, 學習率: {model_learning_rate} ---")
            temp_checkpoint_filepath = os.path.join(WORKING_DIR, f'best_model_fold_{fold+1}_{model_name}.h5')
            model_checkpoint_callback = ModelCheckpoint(
                filepath=temp_checkpoint_filepath, 
                monitor='val_loss', 
                mode='min', 
                save_best_only=True, 
                verbose=0
            )
            early_stopping_callback = EarlyStopping(
                monitor='val_loss', 
                patience=15,
                restore_best_weights=True
            )
            reduce_lr_callback = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )

            history = model.fit(
                train_data, 
                y_train_one_hot, 
                epochs=200,
                batch_size=64,
                validation_data=(val_data, y_val_one_hot), 
                verbose=1,
                callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback],
                class_weight=class_weights_dict
            )
            
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Fold {fold+1}, Model {model_name} Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Fold {fold+1}, Model {model_name} Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

            print(f"{fold+1} {model_name} 訓練集準確率: {history.history['accuracy'][-1]:.4f}")
            print(f"{fold+1} {model_name} 驗證集準確率: {history.history['val_accuracy'][-1]:.4f}")

            model.load_weights(temp_checkpoint_filepath)
            
            predictions = model.predict(val_data)
            predicted_classes = np.argmax(predictions, axis=1)
            binary_f1, macro_f1, final_score = evaluate_metrics(
                y_val, predicted_classes, label_encoder, non_target_gestures
            )
            
            print(f"模型 {model_name} 在折疊 {fold+1} 的驗證得分: Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")

            if not np.isnan(final_score):
                ensemble_model_paths.append((model_name, temp_checkpoint_filepath))

    print("\n訓練流程完成。")
    print(f"\n最佳模型已儲存至: {ensemble_model_paths}")
    
    time_series_features = numerical_features_ts + ['tof_mean', 'tof_std'] + [f'{col}_encoded' for col in categorical_features]
    
    test_sequences, _, _, test_sequence_ids, _, _, test_static_data, test_tof_data = load_and_preprocess_data(TEST_CSV_PATH, TEST_DEMO_PATH, is_training=False)

    print("執行 predict_and_generate_submission_ensemble")
    predict_and_generate_submission_ensemble(
        model_paths=ensemble_model_paths,
        test_sequences=test_sequences,
        test_static_data=test_static_data,
        test_sequence_ids=test_sequence_ids,
        label_encoder=label_encoder,
        max_seq_len=max_seq_len,
        ts_scaler=ts_scaler,
        static_scaler=static_scaler,
        pca=pca,
        time_series_features=time_series_features
    )

if __name__ == '__main__':
    train_and_predict_models()
