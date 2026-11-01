import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, df, audio_dir, transform=None):
        self.df = df
        self.audio_dir = audio_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.df.iloc[idx]['filename'])
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Stack features
        features = np.vstack([mfcc, mel_spec, chroma])
        
        if self.transform:
            features = self.transform(features)
            
        label = self.df.iloc[idx]['label']
        
        return torch.FloatTensor(features), torch.FloatTensor([label])