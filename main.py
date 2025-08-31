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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GRU, Dense, Concatenate, Lambda, Multiply,
    GlobalAveragePooling1D, Reshape, GaussianNoise, Input
)
from tensorflow.keras.mixed_precision import set_global_policy
import joblib

# 啟用混合精度訓練
set_global_policy('mixed_float16')

# =========================================================================
# 全域參數
# =========================================================================
IS_TRAINING = False
N_COMPONENTS = 20
MIXUP_ALPHA = 0.4
WD = 3e-3
LR_INIT = 5e-4
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 64
PAD_PERCENTILE = 95
WORKING_DIR = '/kaggle/working'
TRAIN_CSV_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
TRAIN_DEMO_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv'
TEST_CSV_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
TEST_DEMO_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
TRAINING_OUTPUT_DIR = '/kaggle/input/notebook-cmi-for-training'

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
angular_features = ['rot_x_rate', 'rot_y_rate', 'rot_z_rate', 'rot_dist']

# =========================================================================
# 輔助函式與模型架構
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

def remove_gravity_from_acc(df):
    df = df.copy()
    gravity_removed_acc = df.groupby('sequence_id')[['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']].apply(
        lambda x: _remove_gravity_from_acc_seq(x)
    ).droplevel(0)
    df[['acc_x', 'acc_y', 'acc_z']] = gravity_removed_acc
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
    df = df.copy()
    grouped = df.groupby('sequence_id')
    angular_vel_df = grouped[['rot_x', 'rot_y', 'rot_z', 'rot_w']].apply(
        lambda x: _calculate_angular_velocity_seq(x, time_delta)
    )
    df[['rot_x_rate', 'rot_y_rate', 'rot_z_rate']] = angular_vel_df.droplevel(0)
    angular_dist_df = grouped[['rot_x', 'rot_y', 'rot_z', 'rot_w']].apply(
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

def load_and_preprocess_data_for_training(csv_path, demos_path, is_training=True):
    print(f"正在載入和處理資料: {csv_path}")
    df = pd.read_csv(csv_path)
    demographics_df = pd.read_csv(demos_path)
    df = pd.merge(df, demographics_df, on='subject', how='left')
    
    print("資料載入完成，正在進行特徵工程...")
    df[sensor_features] = df[sensor_features].ffill().bfill()
    
    df = df.assign(
        acc_mag=np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2).astype('float32')
    )
    
    df = remove_gravity_from_acc(df)
    df = calculate_angular_features(df)
    
    numerical_cols = df.select_dtypes(include=np.number).columns
    median_values = df[numerical_cols].median()
    df[numerical_cols] = df[numerical_cols].fillna(median_values)
    
    df = df.assign(
        tof_mean=df[tof_features].mean(axis=1),
        tof_std=df[tof_features].std(axis=1)
    )
    
    sequences = []
    static_features = []
    labels = []
    subjects = []
    sequence_ids = []

    time_series_features = sensor_features + manual_features + angular_features + ['tof_mean', 'tof_std']
    static_feature_cols = demographic_features
    
    if is_training:
        label_encoder = LabelEncoder()
        df = df.assign(gesture_encoded=label_encoder.fit_transform(df['gesture']))
        non_target_gestures = df[df['sequence_type'] != 'Target']['gesture'].unique().tolist()
    else:
        label_encoder = None
        non_target_gestures = None
    
    grouped_stats = df.groupby('sequence_id')['acc_mag'].agg(['mean', 'std']).rename(columns={'mean': 'acc_mag_mean', 'std': 'acc_mag_std'})
    df = df.merge(grouped_stats, on='sequence_id', how='left')
    
    time_series_features = time_series_features + ['acc_mag_mean', 'acc_mag_std']
    
    for sequence_id, group in df.groupby('sequence_id'):
        ts_data = group[time_series_features].values
        static_data = group[static_feature_cols].iloc[0].values
        tof_data_group = group[tof_features].values
        combined_ts = np.concatenate([ts_data, tof_data_group], axis=1)
        sequences.append(combined_ts.astype('float32'))
        static_features.append(static_data.astype('float32'))
        sequence_ids.append(sequence_id)
        if is_training:
            labels.append(group['gesture_encoded'].iloc[0])
            subjects.append(group['subject'].iloc[0])
            
    print("資料分組完成，準備回傳。")
    return (sequences, np.array(labels) if is_training else None, 
            np.array(subjects) if is_training else None, 
            np.array(sequence_ids), label_encoder, non_target_gestures, 
            np.array(static_features), time_series_features)

def time_sum(x): return K.sum(x, axis=1)
def squeeze_last_axis(x): return tf.squeeze(x, axis=-1)
def expand_last_axis(x): return tf.expand_dims(x, axis=-1)
def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=3e-3):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(wd))(shortcut)
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

def build_two_branch_model(ts_input_shape, static_input_shape, num_classes, ts_split_point, wd=3e-3):
    ts_input = Input(shape=ts_input_shape, name='ts_input')
    static_input = Input(shape=static_input_shape, name='static_input')
    
    imu = Lambda(lambda t: t[:, :, :ts_split_point])(ts_input)
    tof = Lambda(lambda t: t[:, :, ts_split_point:])(ts_input)
    
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    merged_ts = Concatenate()([x1, x2])
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(wd)))(merged_ts)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=regularizers.l2(wd)))(merged_ts)
    xc = GaussianNoise(0.09)(merged_ts)
    xc = Dense(16, activation='elu')(xc)
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    ts_context = attention_layer(x)
    
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

def plot_training_history(histories):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, history in enumerate(histories):
        axes[0].plot(history.history['loss'], label=f'Fold {i+1} Train Loss')
        axes[0].plot(history.history['val_loss'], label=f'Fold {i+1} Val Loss')
        axes[1].plot(history.history['accuracy'], label=f'Fold {i+1} Train Acc')
        axes[1].plot(history.history['val_accuracy'], label=f'Fold {i+1} Val Acc')
    axes[0].set_title('Model Loss', fontsize=15)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].legend()
    axes[1].set_title('Model Accuracy', fontsize=15)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def predict_and_generate_submission_ensemble(model_paths, test_ts_data, test_static_data, test_sequence_ids, label_encoder, custom_objects):
    ensemble_predictions = []
    
    # 載入所有模型並進行預測
    for model_type, path in model_paths:
        print(f"正在載入模型: {path}")
        model = load_model(path, compile=False, custom_objects=custom_objects)
        pred = model.predict({'ts_input': test_ts_data, 'static_input': test_static_data})
        ensemble_predictions.append(pred)
        
    # 對所有模型的預測結果取平均
    avg_predictions = np.mean(ensemble_predictions, axis=0)
    predicted_classes = np.argmax(avg_predictions, axis=1)
    predicted_gestures = label_encoder.inverse_transform(predicted_classes)
    
    submission_df = pd.DataFrame({
        'sequence_id': test_sequence_ids, 
        'gesture': predicted_gestures
    })
    
    print(submission_df)
    
    submission_df.to_parquet(os.path.join(WORKING_DIR, 'submission.parquet'), index=False)
    print("提交文件已成功生成：submission.parquet")


def train_and_save_models():
    """
    執行完整的交叉驗證訓練流程並儲存模型和轉換器。
    """
    print("正在執行訓練和模型保存...")
    (train_sequences, y_gesture, subjects, _, 
     label_encoder, non_target_gestures, static_data, 
     time_series_features) = load_and_preprocess_data_for_training(TRAIN_CSV_PATH, TRAIN_DEMO_PATH, is_training=True)
    
    num_gesture_classes = len(label_encoder.classes_)
    sequence_lengths = [len(s) for s in train_sequences]
    max_seq_len = int(np.percentile(sequence_lengths, PAD_PERCENTILE))
    
    imu_and_manual_features_count = len(time_series_features)

    all_padded_ts = pad_sequences(train_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    padded_imu_features = all_padded_ts[:, :, :imu_and_manual_features_count]
    padded_tof_features = all_padded_ts[:, :, imu_and_manual_features_count:]
    
    tof_data_flat_train = padded_tof_features.reshape(-1, len(tof_features))
    valid_tof_rows = tof_data_flat_train[tof_data_flat_train.sum(axis=1) != 0]
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(valid_tof_rows)
    
    padded_tof_pca = pca.transform(tof_data_flat_train).reshape(all_padded_ts.shape[0], all_padded_ts.shape[1], N_COMPONENTS)
    final_ts_features = np.concatenate([padded_imu_features, padded_tof_pca], axis=-1)
    
    ts_scaler = StandardScaler()
    ts_scaler.fit(final_ts_features.reshape(-1, final_ts_features.shape[2]))
    
    static_scaler = StandardScaler()
    static_scaler.fit(static_data)
    
    final_ts_features_scaled = ts_scaler.transform(final_ts_features.reshape(-1, final_ts_features.shape[2])).reshape(final_ts_features.shape)
    static_data_scaled = static_scaler.transform(static_data)
    
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_gesture), y=y_gesture)
    class_weights_dict = dict(enumerate(class_weights))
    
    n_sp = 5
    gkf = GroupKFold(n_splits=n_sp)
    ensemble_model_paths = []
    all_histories = []

    def train_generator(X_ts, X_static, y, batch_size, mixup_alpha):
        num_samples = len(X_ts)
        while True:
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = all_indices[start:end]
                shuffled_batch_indices = np.random.permutation(batch_indices)
                
                x_ts_1 = X_ts[batch_indices]
                x_static_1 = X_static[batch_indices]
                y_1 = y[batch_indices]
                
                x_ts_2 = X_ts[shuffled_batch_indices]
                x_static_2 = X_static[shuffled_batch_indices]
                y_2 = y[shuffled_batch_indices]

                lam = np.random.beta(mixup_alpha, mixup_alpha, size=len(batch_indices))
                lam_ts = lam.reshape(-1, 1, 1)
                lam_static = lam.reshape(-1, 1)

                x_ts_mixed = lam_ts * x_ts_1 + (1 - lam_ts) * x_ts_2
                x_static_mixed = lam_static * x_static_1 + (1 - lam_static) * x_static_2
                y_mixed = lam_static * y_1 + (1 - lam_static) * y_2
                
                yield {'ts_input': x_ts_mixed, 'static_input': x_static_mixed}, y_mixed

    for fold, (train_index, val_index) in enumerate(gkf.split(final_ts_features_scaled, y_gesture, groups=subjects)):
        print(f"\n- 折疊 (Fold) {fold+1}/{n_sp}")
        X_train_ts, X_val_ts = final_ts_features_scaled[train_index], final_ts_features_scaled[val_index]
        X_train_static, X_val_static = static_data_scaled[train_index], static_data_scaled[val_index]
        y_train, y_val = y_gesture[train_index], y_gesture[val_index]
        
        y_train_one_hot = to_categorical(y_train, num_classes=num_gesture_classes)
        y_val_one_hot = to_categorical(y_val, num_classes=num_gesture_classes)

        K.clear_session()
        set_seeds()
        
        ts_split_point = imu_and_manual_features_count
        model = build_two_branch_model(
            ts_input_shape=X_train_ts.shape[1:], 
            static_input_shape=X_train_static.shape[1:], 
            num_classes=num_gesture_classes, 
            ts_split_point=ts_split_point, 
            wd=WD
        )
        
        temp_checkpoint_filepath = os.path.join(WORKING_DIR, f'best_model_fold_{fold+1}.h5')
        
        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        model_checkpoint_callback = ModelCheckpoint(filepath=temp_checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=0)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        
        train_gen = train_generator(X_train_ts, X_train_static, y_train_one_hot, BATCH_SIZE, MIXUP_ALPHA)
        
        history = model.fit(
            train_gen,
            steps_per_epoch=int(np.ceil(len(X_train_ts) / BATCH_SIZE)),
            epochs=EPOCHS,
            validation_data=({'ts_input': X_val_ts, 'static_input': X_val_static}, y_val_one_hot),
            verbose=1,
            callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]
        )
        all_histories.append(history)
        model.load_weights(temp_checkpoint_filepath)
        predictions = model.predict({'ts_input': X_val_ts, 'static_input': X_val_static})
        predicted_classes = np.argmax(predictions, axis=1)
        binary_f1, macro_f1, final_score = evaluate_metrics(y_val, predicted_classes, label_encoder, non_target_gestures)
        print(f"模型在折疊 {fold+1} 的驗證得分: Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final Score: {final_score:.4f}")
        if not np.isnan(final_score):
            ensemble_model_paths.append(('two_branch', temp_checkpoint_filepath))
    
    print("\n訓練流程完成。")
    plot_training_history(all_histories)
    print(f"\n最佳模型已儲存至: {ensemble_model_paths}")
    
    joblib.dump(pca, os.path.join(WORKING_DIR, 'pca.pkl'))
    joblib.dump(ts_scaler, os.path.join(WORKING_DIR, 'ts_scaler.pkl'))
    joblib.dump(static_scaler, os.path.join(WORKING_DIR, 'static_scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(WORKING_DIR, 'label_encoder.pkl'))
    joblib.dump(max_seq_len, os.path.join(WORKING_DIR, 'max_seq_len.pkl'))
    joblib.dump(imu_and_manual_features_count, os.path.join(WORKING_DIR, 'ts_feature_count.pkl'))

def predict_and_generate_submission():
    pca = joblib.load(os.path.join(TRAINING_OUTPUT_DIR, 'pca.pkl'))
    ts_scaler = joblib.load(os.path.join(TRAINING_OUTPUT_DIR, 'ts_scaler.pkl'))
    static_scaler = joblib.load(os.path.join(TRAINING_OUTPUT_DIR, 'static_scaler.pkl'))
    label_encoder = joblib.load(os.path.join(TRAINING_OUTPUT_DIR, 'label_encoder.pkl'))
    max_seq_len = joblib.load(os.path.join(TRAINING_OUTPUT_DIR, 'max_seq_len.pkl'))
    ts_feature_count = joblib.load(os.path.join(TRAINING_OUTPUT_DIR, 'ts_feature_count.pkl'))
    
    model_paths = []
    for i in range(5):
        path = os.path.join(TRAINING_OUTPUT_DIR, f'best_model_fold_{i+1}.h5')
        if os.path.exists(path):
            model_paths.append(('two_branch', path))
    
    print("在離線模式下進行預測並生成提交文件。")
    (test_sequences, _, _, test_sequence_ids, _, _, 
     test_static_data, _) = load_and_preprocess_data_for_training(TEST_CSV_PATH, TEST_DEMO_PATH, is_training=False)
    
    test_sequences_padded = pad_sequences(test_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    
    test_imu_features = test_sequences_padded[:, :, :ts_feature_count]
    test_tof_features = test_sequences_padded[:, :, ts_feature_count:]
    
    test_tof_pca = pca.transform(test_tof_features.reshape(-1, len(tof_features))).reshape(test_tof_features.shape[0], test_tof_features.shape[1], N_COMPONENTS)
    test_final_ts_features = np.concatenate([test_imu_features, test_tof_pca], axis=-1)
    test_final_ts_features_scaled = ts_scaler.transform(test_final_ts_features.reshape(-1, test_final_ts_features.shape[2])).reshape(test_final_ts_features.shape)
    test_static_data_scaled = static_scaler.transform(test_static_data)

    custom_objects = {
        'time_sum': time_sum, 
        'squeeze_last_axis': squeeze_last_axis, 
        'expand_last_axis': expand_last_axis, 
        'se_block': se_block,
        'residual_se_cnn_block': residual_se_cnn_block,
        'attention_layer': attention_layer
    }

    predict_and_generate_submission_ensemble(
        model_paths=model_paths,
        test_ts_data=test_final_ts_features_scaled,
        test_static_data=test_static_data_scaled,
        test_sequence_ids=test_sequence_ids,
        label_encoder=label_encoder,
        custom_objects=custom_objects
    )


# =========================================================================
# 主入口點
# =========================================================================
if __name__ == '__main__':
    # 執行訓練並生成 submission.parquet
    if IS_TRAINING:
        train_and_save_models()
    else:
        predict_and_generate_submission()
    