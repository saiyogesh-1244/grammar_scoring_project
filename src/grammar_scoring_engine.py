
#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):

    def __init__(self, df, audio_dir, max_length=3000):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.has_labels = 'label' in df.columns

    def __len__(self):
        return len(self.df)

    def extract_features(self, audio_path):

        y, sr = librosa.load(audio_path, duration=60)

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13
        )

        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr
        )

        spec_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr
        )

        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(y), sr=sr
        )

        zcr = librosa.feature.zero_crossing_rate(y)

        rms = librosa.feature.rms(y=y)

        features = np.vstack([
            mfcc,
            chroma,
            spec_contrast,
            tonnetz,
            zcr,
            rms
        ])

        if features.shape[1] < self.max_length:
            pad_width = (
                (0, 0),
                (0, self.max_length - features.shape[1])
            )

            features = np.pad(
                features,
                pad_width,
                mode='constant'
            )

        else:
            features = features[:, :self.max_length]

        mean = np.mean(features)
        std = np.std(features)

        features = (features - mean) / (std + 1e-8)

        return features.astype(np.float32)

    def __getitem__(self, idx):

        filename = self.df.iloc[idx]['filename']

        audio_path = os.path.join(
            self.audio_dir, filename
        )

        features = self.extract_features(audio_path)

        if self.has_labels:
            label = float(self.df.iloc[idx]['label'])
        else:
            label = 0.0

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32)
        )


class GrammarScoringModel(nn.Module):

    def __init__(self, input_channels, hidden_dim=128):

        super(GrammarScoringModel, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 8))

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(
            128 * 4 * 8, hidden_dim
        )

        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.pool(
            F.relu(self.bn1(self.conv1(x)))
        )

        x = self.pool(
            F.relu(self.bn2(self.conv2(x)))
        )

        x = self.pool(
            F.relu(self.bn3(self.conv3(x)))
        )

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        # Unrestricted regression output during training
        x = self.fc2(x)

        return x


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=10,
    patience=3
):

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    model_path = 'best_model.pth'

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        for inputs, labels in tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{num_epochs}'
        ):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        train_losses.append(train_loss)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        val_losses.append(val_loss)

        print(
            f'Epoch {epoch + 1}/{num_epochs} '
            f'- Train Loss: {train_loss:.4f}, '
            f'Val Loss: {val_loss:.4f}'
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                model_path
            )

            print('Best model saved!')

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(
                    f'Early stopping after {epoch + 1} epochs'
                )

                break

    if os.path.exists(model_path):

        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

    return train_losses, val_losses


def evaluate_model(model, test_loader):

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():

        for inputs, _ in test_loader:

            inputs = inputs.to(device)

            outputs = model(inputs)

            predictions.extend(
                outputs.cpu().numpy().flatten()
            )

    # Keep final grammar scores within 1–5
    predictions = np.clip(predictions, 1.0, 5.0)

    return predictions


def main():

    set_seed(42)

    print(f'Python version: {sys.version}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'NumPy version: {np.__version__}')
    print(f'Pandas version: {pd.__version__}')
    print(f'Librosa version: {librosa.__version__}')

    print('\nLoading data...')

    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    print('Training data shape:', train_df.shape)
    print('Test data shape:', test_df.shape)

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42
    )

    print('\nCreating datasets...')

    train_dataset = AudioDataset(
        train_df, 'data/train'
    )

    val_dataset = AudioDataset(
        val_df, 'data/train'
    )

    test_dataset = AudioDataset(
        test_df, 'data/test'
    )

    sample_features, _ = train_dataset[0]

    input_channels = sample_features.shape[0]

    print('Input channels:', input_channels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32
    )

    print('\nInitializing model...')

    model = GrammarScoringModel(input_channels)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.001
    )

    print('\nTraining model...')

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=10,
        patience=3
    )

    print('\nEvaluating model...')

    predictions = evaluate_model(
        model, test_loader
    )

    submission_df = pd.DataFrame({
        'filename': test_df['filename'],
        'label': predictions
    })

    submission_df.to_csv(
        'data/sample_submission.csv',
        index=False
    )

    print('\nSubmission file created:')
    print('data/sample_submission.csv')

    print('\nPrediction summary:')

    print(
        submission_df['label'].describe()
    )


if __name__ == '__main__':
    main()
