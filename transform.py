import librosa
import numpy as np
import torch

def audio_to_melspectrogram(audio_path, sr=16000, n_fft=512, hop_length=160, n_mels=128):
    """将音频文件转为梅尔频谱图（[1, n_mels, time_steps]）"""
    # 加载音频
    y, _ = librosa.load(audio_path, sr=sr)
    # 计算梅尔频谱
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # 转为对数刻度（提升低振幅特征的区分度）
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # 标准化到 [0, 1] 并增加通道维度（适配CNN输入）
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    return torch.tensor(mel_spec_db).unsqueeze(0).float()  # 形状：[1, 128, T]