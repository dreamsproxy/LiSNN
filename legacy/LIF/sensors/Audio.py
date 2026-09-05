import numpy as np
import pyaudio
import struct
from sklearn.preprocessing import minmax_scale

class Live:
    def __init__(self, n_bands) -> None:
        print("INIT AUDIO")
        self.n_bands = n_bands
        self.CHUNK = 2**10
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
        self.stream=p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=1)

    def ExtractFrequencyBands(self):
        data = struct.unpack(
            str(self.CHUNK*self.CHANNELS) + 'h',
            self.stream.read(self.CHUNK))
        fft_data = np.fft.rfft(data)
        fft_data = np.abs(fft_data[:self.CHUNK]) * 2 / (256 * self.CHUNK)
        n = fft_data.size // self.n_bands
        bands = np.array([np.mean(fft_data[i:(i + n)]) for i in range(0, fft_data.size, n)])
        bands /= np.max(np.abs(bands),axis=0)
        bands_scaled = minmax_scale(bands, feature_range=(-75,-55))

        return bands_scaled

    def shutdown(self):
        self.stream.close()