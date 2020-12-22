import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read

sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0 # 2 ** 15

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

class Mel2SampWaveglow(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, filepath):
        audio, sr = load_wav_to_torch(filepath)
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec