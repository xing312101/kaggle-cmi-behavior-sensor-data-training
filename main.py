import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
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
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GRU, Dense, Concatenate, Lambda, Multiply,
    LayerNormalization, GlobalAveragePooling1D, Reshape, GaussianNoise
)
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.mixed_precision import set_global_policy

# 啟用混合精度訓練
set_global_policy('mixed_float16')

N_COMPONENTS = 20
# 新增 Mixup 參數
MIXUP_ALPHA = 0.4
WD = 3e-3 # Weight Decay
LR_INIT = 5e-4
EPOCHS = 260
PATIENCE = 60
BATCH_SIZE = 64
PAD_PERCENTILE = 95


# =========================================================================
# 設定隨機種子以確保可重現性
# =========================================================================
def set_seeds(seed=42):
    """設定所有必要的隨機種子以確保實驗的可重現性。"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
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
manual_features = ['acc_mag', 'acc_mag_mean', 'acc_mag_std']
angular_features = ['rot_x_rate', 'rot_y_rate', 'rot_z_rate', 'rot_dist']

numerical_features_ts = sensor_features + manual_features + angular_features
categorical_features = ['orientation', 'behavior', 'phase']

# =========================================================================
# 升級後的物理特徵工程函式
# =========================================================================

def remove_gravity_from_acc(df):
    print("正在移除加速度數據中的重力...")
    df = df.copy() # 避免 SettingWithCopyWarning
    df[['acc_x', 'acc_y', 'acc_z']] = df.groupby('sequence_id').apply(
        lambda x: _remove_gravity_from_acc_seq(x)
    )
    return df

def _remove_gravity_from_acc_seq(seq_df):
    acc_values = seq_df[['acc_x', 'acc_y', 'acc_z']].fillna(0).values
    quat_values = seq_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].fillna(0).values
    
    linear_accel = np.copy(acc_values)
    gravity_world = np.array([0, 0, 9.81])

    for i in range(len(seq_df)):
        if not np.all(np.isclose(quat_values[i], 0)):
            try:
                rotation = R.from_quat(quat_values[i])
                gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
                linear_accel[i, :] -= gravity_sensor_frame
            except ValueError:
                pass
    return pd.DataFrame(linear_accel, columns=['acc_x', 'acc_y', 'acc_z'], index=seq_df.index)

def calculate_angular_features(df, time_delta=1/200):
    print("正在計算角速度和角距離...")
    df = df.copy()
    grouped = df.groupby('sequence_id')
    
    # 計算角速度
    angular_vel_df = grouped.apply(
        lambda x: _calculate_angular_velocity_seq(x, time_delta)
    )
    df[['rot_x_rate', 'rot_y_rate', 'rot_z_rate']] = angular_vel_df.droplevel(0)
    
    # 計算角距離
    angular_dist_df = grouped.apply(
        lambda x: _calculate_angular_distance_seq(x)
    )
    df['rot_dist'] = angular_dist_df.droplevel(0)
    
    return df

def _calculate_angular_velocity_seq(seq_df, time_delta):
    quat_values = seq_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].fillna(0).values
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    for i in range(num_samples - 1):
        try:
            rot_t = R.from_quat(quat_values[i])
            rot_t_plus_dt = R.from_quat(quat_values[i+1])
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    return pd.DataFrame(angular_vel, columns=['rot_x_rate', 'rot_y_rate', 'rot_z_rate'], index=seq_df.index)

def _calculate_angular_distance_seq(seq_df):
    quat_values = seq_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].fillna(0).values
    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)
    for i in range(num_samples - 1):
        try:
            r1 = R.from_quat(quat_values[i])
            r2 = R.from_quat(quat_values[i+1])
            relative_rotation = r1.inv() * r2
            angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
        except ValueError:
            pass
    return pd.Series(angular_dist, index=seq_df.index)

def load_and_preprocess_data(csv_path, demos_path, is_training=True):
    print(f"正在載入和處理資料: {csv_path}")
    df = pd.read_csv(csv_path)
    demographics_df = pd.read_csv(demos_path)
    df = pd.merge(df, demographics_df, on='subject', how='left')
    
    # 填充缺失值並進行特徵工程
    df[sensor_features] = df[sensor_features].fillna(method='ffill').fillna(method='bfill')
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2).astype('float32')
    
    df = remove_gravity_from_acc(df)
    df = calculate_angular_features(df)
    
    # 填充剩下的缺失值
    df.fillna(0, inplace=True)
    
    # 為 TOF 數據創建平均值和標準差
    df['tof_mean'] = df[tof_features].mean(axis=1)
    df['tof_std'] = df[tof_features].std(axis=1)

    sequences = []
    static_features = []
    labels = []
    subjects = []
    sequence_ids = []

    # 時序特徵與靜態特徵
    time_series_features = sensor_features + manual_features + angular_features + ['tof_mean', 'tof_std']
    static_feature_cols = demographic_features
    
    if is_training:
        label_encoder = LabelEncoder()
        df['gesture_encoded'] = label_encoder.fit_transform(df['gesture'])
        non_target_gestures = df[df['sequence_type'] != 'Target']['gesture'].unique().tolist()
    else:
        label_encoder = None
        non_target_gestures = None

    print("正在將資料分組為序列並提取特徵...")
    for sequence_id, group in df.groupby('sequence_id'):
        ts_data = group[time_series_features].values
        static_data = group[static_feature_cols].iloc[0].values
        
        # 結合 TOF 數據
        tof_data_group = group[tof_features].values
        combined_ts = np.concatenate([ts_data, tof_data_group], axis=1)

        sequences.append(combined_ts.astype('float32'))
        static_features.append(static_data.astype('float32'))
        sequence_ids.append(sequence_id)
        if is_training:
            labels.append(group['gesture_encoded'].iloc[0])
            subjects.append(group['subject'].iloc[0])

    return sequences, np.array(labels) if is_training else None, np.array(subjects) if is_training else None, np.array(sequence_ids), label_encoder, non_target_gestures, np.array(static_features)

# --- 新增的 Mixup 數據增強生成器 ---
class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(self.X))
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    def __getitem__(self, i):
        idx = self.indices[i*self.batch_size:(i+1)*self.batch_size]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1 - lam) * Xb[perm]
        y_mix = lam * yb + (1 - lam) * yb[perm]
        return X_mix, y_mix
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# --- 升級後的自定義層次 ---
def time_sum(x):
    return K.sum(x, axis=1)
def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)
def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)
def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=3e-3):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

# =========================================================================
# 升級後的多輸入模型 (雙分支設計)
# =========================================================================
def build_two_branch_model(ts_input_shape, static_input_shape, num_classes, wd=3e-3):
    print("建立雙分支模型架構...")
    
    ts_input = layers.Input(shape=ts_input_shape, name='ts_input')
    static_input = layers.Input(shape=static_input_shape, name='static_input')

    # 時序特徵分割
    imu_dim = len(sensor_features) + len(manual_features) + len(angular_features) + N_COMPONENTS
    tof_dim = len(tof_features)
    
    # IMU 深層分支
    imu = Lambda(lambda t: t[:, :, :imu_dim])(ts_input)
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    # TOF 輕量分支
    tof = Lambda(lambda t: t[:, :, imu_dim:])(ts_input)
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)

    merged_ts = Concatenate()([x1, x2])
    
    # 合併 Bidirectional LSTM/GRU 和 GaussianNoise
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(wd)))(merged_ts)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=regularizers.l2(wd)))(merged_ts)
    xc = GaussianNoise(0.09)(merged_ts)
    xc = Dense(16, activation='elu')(xc)
    
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    
    ts_context = attention_layer(x)
    
    # 靜態資料分支
    static_branch = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(wd))(static_input)
    static_branch = BatchNormalization()(static_branch)
    static_branch = Dropout(0.5)(static_branch)

    merged_final = Concatenate()([ts_context, static_branch])
    
    merged_final = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(wd))(merged_final)
    merged_final = BatchNormalization()(merged_final)
    merged_final = Dropout(0.5)(merged_final)

    output = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(wd))(merged_final)
    
    model = Model(inputs=[ts_input, static_input], outputs=output)
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LR_INIT, weight_decay=WD)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
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
# 主程式
# =========================================================================
def train_and_predict_models():
    train_sequences, y_gesture, subjects, _, label_encoder, non_target_gestures, static_data = load_and_preprocess_data(TRAIN_CSV_PATH, TRAIN_DEMO_PATH)
    
    num_gesture_classes = len(label_encoder.classes_)
    print(f"偵測到 {num_gesture_classes} 個手勢類別。")
    
    sequence_lengths = [len(s) for s in train_sequences]
    max_seq_len = int(np.percentile(sequence_lengths, PAD_PERCENTILE))
    print(f"最大序列長度: {max_seq_len}")
    
    # 對 TOF 數據進行 PCA 降維
    tof_features_count = len(tof_features)
    all_tof_data = [seq[:, -tof_features_count:] for seq in train_sequences]
    padded_tof_data = pad_sequences(all_tof_data, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    tof_data_flat = padded_tof_data.reshape(-1, tof_features_count)
    
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(tof_data_flat[tof_data_flat.sum(axis=1) != 0])
    
    # 標準化器
    ts_scaler = StandardScaler()
    static_scaler = StandardScaler()

    all_padded_ts = pad_sequences(train_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    
    # 訓練和測試數據的準備，將時序和靜態數據合併
    num_ts_features_without_tof = len(numerical_features_ts) + len(tof_features)
    padded_ts_features = all_padded_ts[:, :, :num_ts_features_without_tof]
    padded_tof_features = all_padded_ts[:, :, num_ts_features_without_tof:]

    padded_tof_pca = pca.transform(padded_tof_features.reshape(-1, len(tof_features))).reshape(padded_tof_features.shape[0], padded_tof_features.shape[1], N_COMPONENTS)

    final_ts_features = np.concatenate([padded_ts_features, padded_tof_pca], axis=-1)
    
    ts_scaler.fit(final_ts_features.reshape(-1, final_ts_features.shape[2]))
    static_scaler.fit(static_data)
    
    final_ts_features_scaled = ts_scaler.transform(final_ts_features.reshape(-1, final_ts_features.shape[2])).reshape(final_ts_features.shape)
    static_data_scaled = static_scaler.transform(static_data)

    print("正在計算類別權重...")
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_gesture), y=y_gesture)
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
    
    for fold, (train_index, val_index) in enumerate(gkf.split(final_ts_features_scaled, y_gesture, groups=subjects)):
        print(f"\n- 折疊 (Fold) {fold+1}/{n_sp}")
        
        X_train_ts, X_val_ts = final_ts_features_scaled[train_index], final_ts_features_scaled[val_index]
        X_train_static, X_val_static = static_data_scaled[train_index], static_data_scaled[val_index]
        y_train, y_val = y_gesture[train_index], y_gesture[val_index]
        
        y_train_one_hot = to_categorical(y_train, num_classes=num_gesture_classes)
        y_val_one_hot = to_categorical(y_val, num_classes=num_gesture_classes)

        K.clear_session()
        set_seeds()
        
        with strategy.scope():
            model = build_two_branch_model(X_train_ts.shape[1:], X_train_static.shape[1:], num_gesture_classes, wd=WD)
        
        temp_checkpoint_filepath = os.path.join(WORKING_DIR, f'best_model_fold_{fold+1}.h5')
        model_checkpoint_callback = ModelCheckpoint(
            filepath=temp_checkpoint_filepath, 
            monitor='val_loss', 
            mode='min', 
            save_best_only=True, 
            verbose=0
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE,
            restore_best_weights=True
        )

        history = model.fit(
            {'ts_input': X_train_ts, 'static_input': X_train_static},
            y_train_one_hot, 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=({'ts_input': X_val_ts, 'static_input': X_val_static}, y_val_one_hot), 
            verbose=1,
            callbacks=[model_checkpoint_callback, early_stopping_callback],
            class_weight=class_weights_dict
        )
        
        model.load_weights(temp_checkpoint_filepath)
        
        predictions = model.predict({'ts_input': X_val_ts, 'static_input': X_val_static})
        predicted_classes = np.argmax(predictions, axis=1)
        binary_f1, macro_f1, final_score = evaluate_metrics(
            y_val, predicted_classes, label_encoder, non_target_gestures
        )
        
        print(f"模型在折疊 {fold+1} 的驗證得分: Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")

        if not np.isnan(final_score):
            ensemble_model_paths.append(('two_branch', temp_checkpoint_filepath))

    print("\n訓練流程完成。")
    print(f"\n最佳模型已儲存至: {ensemble_model_paths}")
    
    # 測試集處理與預測
    test_sequences, _, _, test_sequence_ids, _, _, test_static_data = load_and_preprocess_data(TEST_CSV_PATH, TEST_DEMO_PATH, is_training=False)
    
    test_sequences_padded = pad_sequences(test_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    
    num_ts_features_without_tof_test = len(numerical_features_ts) + len(tof_features)
    test_ts_features = test_sequences_padded[:, :, :num_ts_features_without_tof_test]
    test_tof_features = test_sequences_padded[:, :, num_ts_features_without_tof_test:]

    test_tof_pca = pca.transform(test_tof_features.reshape(-1, len(tof_features))).reshape(test_tof_features.shape[0], test_tof_features.shape[1], N_COMPONENTS)

    test_final_ts_features = np.concatenate([test_ts_features, test_tof_pca], axis=-1)
    
    test_final_ts_features_scaled = ts_scaler.transform(test_final_ts_features.reshape(-1, test_final_ts_features.shape[2])).reshape(test_final_ts_features.shape)
    test_static_data_scaled = static_scaler.transform(test_static_data)

    predict_and_generate_submission_ensemble(
        model_paths=ensemble_model_paths,
        test_ts_data=test_final_ts_features_scaled,
        test_static_data=test_static_data_scaled,
        test_sequence_ids=test_sequence_ids,
        label_encoder=label_encoder
    )

def predict_and_generate_submission_ensemble(
    model_paths, test_ts_data, test_static_data, test_sequence_ids, label_encoder
):
    print("\n--- 正在進行集成學習預測 ---")
    
    ensemble_predictions = []
    
    for i, model_info in enumerate(model_paths):
        model_name, model_path = model_info
        try:
            print(f"載入模型 {i+1}/{len(model_paths)} ({model_name}): {model_path}")
            custom_objects = {
                'time_sum': time_sum, 'squeeze_last_axis': squeeze_last_axis, 'expand_last_axis': expand_last_axis,
                'se_block': se_block, 'residual_se_cnn_block': residual_se_cnn_block, 'attention_layer': attention_layer,
            }
            model = load_model(model_path, compile=False, custom_objects=custom_objects)
            
            predictions = model.predict({'ts_input': test_ts_data, 'static_input': test_static_data}, verbose=0)
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

if __name__ == '__main__':
    train_and_predict_models()
    