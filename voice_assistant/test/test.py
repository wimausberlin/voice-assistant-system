import core.dataset
import torchaudio
import torchaudio.transforms as T

def test_mfcc():
    file_path="../../sound/0/2.wav"
    waveform, sample_rate = torchaudio.load(file_path)
    mfcc_tranf = T.MFCC(sample_rate)
    mfcc=mfcc_tranf(waveform)
    core.dataset.plot_spectrogram(mfcc[0])

test_mfcc()

