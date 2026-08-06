import os
import glob
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import pyloudnorm as pyln
from tqdm.notebook import tqdm

# 파라미터 설정
TARGET_SR = 16000
LOWPASS_CUTOFF = 7500 
HOP_LENGTH = int(0.010 * TARGET_SR) 
WIN_LENGTH = int(0.025 * TARGET_SR) 
N_FFT = 512
N_MELS = 80

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

meter = pyln.Meter(TARGET_SR)
loudness_records = []

# --- 전처리 메인 루프 ---
for orig_folder, target_folder in folder_mapping.items():
    folder_path = os.path.join(base_path, orig_folder)
    
    save_folder_path = os.path.join(save_base_path, target_folder)
    os.makedirs(save_folder_path, exist_ok=True)
    
    #  파일 검색 (하위 폴더 및 대소문자 모두 허용)
    wav_files = []
    for ext in ('*.wav', '*.WAV'):
        wav_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    
    print(f"[{orig_folder} -> {target_folder}] 처리 시작 (총 {len(wav_files)}개 파일 발견)")
    
    for file_path in tqdm(wav_files):
        # 파일명 충돌 방지를 위해 상대 경로를 파일명으로 변환
        relative_path = os.path.relpath(file_path, folder_path)
        file_name = relative_path.replace(os.sep, '_')
        
        try:
            # 1. 로드 -> 16kHz 모노
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            
            # 2. 대역 통일 (Lowpass Filter)
            y = butter_lowpass_filter(y, cutoff=LOWPASS_CUTOFF, fs=sr)
            
            # 3. DC Offset 제거
            y = y - np.mean(y)
            
            # 4. 앞뒤 무음만 트리밍
            y_trimmed, index = librosa.effects.trim(y, top_db=30)
            
            # 5. 라우드니스 측정 -> 값만 저장
            if len(y_trimmed) >= int(0.4 * sr): 
                loudness = meter.integrated_loudness(y_trimmed)
            else:
                loudness = np.nan
                
            loudness_records.append({
                'Class': target_folder,
                'File_Name': file_name,
                'Loudness_LUFS': loudness
            })
            
            # 6. Log-Mel 스펙트로그램 추출
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y_trimmed, 
                sr=sr, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                win_length=WIN_LENGTH, 
                n_mels=N_MELS
            )
            log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # 저장 (.npy)
            save_name = os.path.splitext(file_name)[0] + '.npy'
            save_file_path = os.path.join(save_folder_path, save_name)
            np.save(save_file_path, log_mel)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# 라우드니스 특징값을 폴더에 CSV로 저장
df_loudness = pd.DataFrame(loudness_records)
csv_save_path = os.path.join(save_base_path, 'loudness_features.csv')
df_loudness.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 전처리가 완료되었습니다!")
print(f"📁 결과물 저장 위치: {save_base_path}")