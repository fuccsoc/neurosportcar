from __future__ import annotations

import numpy as np

SAMPLE_RATE = 250


class RingBuffer:
    def __init__(self, n_channels: int, maxlen: int):
        self.n_channels = int(n_channels)
        self.maxlen = int(maxlen)
        self.data = np.zeros((self.n_channels, self.maxlen), dtype=float)
        self.idx = 0
        self.count = 0

    def append_block(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=float)
        n_ch, n_s = block.shape
        if n_ch != self.n_channels:
            raise ValueError(f"RingBuffer ожидает {self.n_channels} каналов, получил {n_ch}")

        for i in range(n_s):
            self.data[:, self.idx] = block[:, i]
            self.idx = (self.idx + 1) % self.maxlen
            self.count = min(self.maxlen, self.count + 1)

    def get(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((self.n_channels, 0), dtype=float)

        start = (self.idx - self.count) % self.maxlen
        if start < self.idx:
            out = self.data[:, start:self.idx]
        else:
            out = np.concatenate([self.data[:, start:], self.data[:, :self.idx]], axis=1)
        return out.copy()


class RealTimeFilter:
    """
    Каузальный IIR-фильтр (SOS) по каналам.
    """
    def __init__(self, sfreq: float, l_freq: float | None, h_freq: float | None, n_channels: int, order: int = 4):
        from scipy.signal import butter, sosfilt_zi

        self.sfreq = float(sfreq)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.n_channels = int(n_channels)

        if l_freq is not None and h_freq is not None:
            btype = "bandpass"
            freqs = [float(l_freq), float(h_freq)]
        elif l_freq is not None:
            btype = "highpass"
            freqs = float(l_freq)
        elif h_freq is not None:
            btype = "lowpass"
            freqs = float(h_freq)
        else:
            raise ValueError("Нужно указать хотя бы l_freq или h_freq")

        self.sos = butter(order, freqs, btype=btype, fs=self.sfreq, output="sos")
        self.zi = [sosfilt_zi(self.sos) for _ in range(self.n_channels)]

    def filter_block(self, block: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt

        block = np.asarray(block, dtype=float)
        n_ch, _ = block.shape
        if n_ch != self.n_channels:
            raise ValueError(f"RealTimeFilter ожидает {self.n_channels} каналов, получил {n_ch}")

        out = np.zeros_like(block)
        for ch in range(n_ch):
            out[ch], self.zi[ch] = sosfilt(self.sos, block[ch], zi=self.zi[ch])
        return out

    def reset(self) -> None:
        from scipy.signal import sosfilt_zi
        self.zi = [sosfilt_zi(self.sos) for _ in range(self.n_channels)]


def compute_psd(block: np.ndarray, sfreq: float, fmin: float, fmax: float):
    """
    PSD (канал, частота). Пытаемся через MNE, иначе fallback на scipy.welch.
    """
    block = np.asarray(block, dtype=float)

    try:
        import mne
        psd, freqs = mne.time_frequency.psd_array_welch(
            block,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=int(sfreq * 2),
            verbose=False,
        )
        return freqs, psd
    except Exception:
        pass

    from scipy.signal import welch
    freqs, psd = welch(block, fs=sfreq, nperseg=int(sfreq * 2), axis=1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[:, mask]
