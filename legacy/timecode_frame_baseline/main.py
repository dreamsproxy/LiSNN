import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from tqdm import tqdm

import LIF
from WeightMatrix import WeightMatrix


@njit(parallel=True)
def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


class DataLoader:
    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        img = img[:640, :640]
        img = cv2.resize(img, (64, 64))
        img /= 255.0
        img *= 500.0
        return img

    def synthetic(self, size=(32, 32)):
        img = np.zeros(shape=size, dtype=np.float64)
        img[16:17, 16:17] = 30.0
        return img

    def load_video(self, path, num_repeats=8):
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        frames = []

        while success:
            img = image
            img = img[:872, :872, :]
            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float64)
            img = cv2.resize(img, (32, 32))
            img /= 255.0
            img *= 500.0

            repeated = []
            for _ in range(num_repeats):
                repeated.append(img)
            frames.append(np.array(repeated))

            del img
            success, image = vidcap.read()
            print("Read a new frame: ", success)
            count += 1

        frames = [
            frame
            for i, frame in enumerate(frames)
            if i % 10 == 0
        ]
        return frames


class Network:
    def __init__(
        self,
        num_neurons: int,
        n_inputs: int,
        encodings: np.ndarray,
        dt: float = 0.1,
        shape=(28, 28),
    ) -> None:
        self.shape = shape
        self.n_inputs = n_inputs
        self.num_neurons = num_neurons + encodings.shape[0]
        self.weights = WeightMatrix(
            self.num_neurons,
            0.001,
            0.002,
        )
        self.encodings = encodings

        self.weights.weights[-self.encodings.shape[0]:] = 0.5

        self.input_weights = np.random.uniform(
            -0.9,
            0.9,
            (self.n_inputs, self.num_neurons),
        )
        self.output_weights = np.random.uniform(
            -0.9,
            0.9,
            (self.n_inputs, self.num_neurons),
        )

        v_rest = np.random.uniform(
            -66.0,
            -64.0,
            self.num_neurons,
        ).astype(np.float64)
        v_reset = v_rest - np.float64(5.0)
        tau = np.random.uniform(
            19.5,
            20.5,
            self.num_neurons,
        ).astype(np.float64)
        self.thresh = np.random.uniform(
            -55.0,
            -45.0,
            self.num_neurons,
        ).astype(np.float64)
        self.init_potentials = v_rest + np.float64(2.71)

        neurons = {}
        for i in range(self.num_neurons):
            if i >= self.num_neurons - self.encodings.shape[0]:
                neurons[i] = {
                    "potential": np.float64(-65.0),
                    "dt": np.float64(dt),
                    "tau": np.float64(20.0),
                    "v_rest": np.float64(-65.0),
                    "v_reset": np.float64(-70.0),
                    "v_threshold": np.float64(-55.0),
                }
            else:
                neurons[i] = {
                    "potential": self.init_potentials[i],
                    "dt": np.float64(dt),
                    "tau": tau[i],
                    "v_rest": v_rest[i],
                    "v_reset": v_reset[i],
                    "v_threshold": self.thresh[i],
                }

        keys = [
            "potential",
            "dt",
            "tau",
            "v_rest",
            "v_reset",
            "v_threshold",
        ]
        neuron_array = np.empty(
            (self.num_neurons, len(keys)),
            dtype=np.float64,
        )
        for i in neurons:
            neuron = neurons[i]
            neuron_array[i] = tuple(neuron[key] for key in keys)
        self.neurons = neuron_array

        self.init_spikes = np.zeros(
            shape=self.num_neurons,
            dtype=np.float64,
        )
        self.post_spikes = self.init_spikes.copy()
        self.pre_spikes = self.init_spikes.copy()
        self.post_tau = tau
        self.pre_tau = tau

        self.global_step_tick = 0
        self.clip_interval = 8
        self.error_thresholds = np.linspace(1.0, 0.1, num=10)

    def step(self, input_signals):
        for ni in range(self.num_neurons):
            if ni < self.num_neurons - self.encodings.shape[0]:
                wp, new_p = LIF.step(
                    self.neurons[ni],
                    input_signals[ni],
                )
                self.neurons[ni][0] = new_p
                self.post_spikes[ni] = wp
            else:
                self.post_spikes[ni] = self.thresh[ni]

        self.post_spikes = self.weights.compute_spikes(
            self.post_spikes,
            self.thresh,
        )
        self.pre_spikes = self.weights.compute_spikes(
            self.pre_spikes,
            self.thresh,
        )

        norm_input = normalize(
            input_signals[: self.num_neurons - self.encodings.shape[0]]
        )
        norm_output = normalize(
            self.post_spikes[: self.num_neurons - self.encodings.shape[0]]
        )
        error_vector = np.abs(norm_input - norm_output)

        for threshold in self.error_thresholds:
            error_indices = np.where(error_vector > np.float64(threshold))
            if len(error_indices) > 0:
                for idx in error_indices:
                    self.post_spikes[idx] = self.neurons[idx, 4]
                break

        if self.global_step_tick % self.clip_interval:
            clip = True
        else:
            clip = False

        self.weights.update_weights_combined(
            self.pre_spikes,
            self.post_spikes,
            self.pre_tau,
            self.post_tau,
            clip=clip,
            top_k=self.num_neurons // 4,
            error=error_vector,
        )

        signals = self.weights.propagate_signals(
            self.post_spikes,
            method="mean",
        )
        self.pre_spikes = self.post_spikes.copy()
        self.global_step_tick += 1
        return signals

    def run(self, data_stream):
        total_epochs = len(data_stream)
        for frame_index, frame_ticks in enumerate(data_stream):
            recurrent_signal = self.pre_spikes
            print(f"Epoch {frame_index + 1}/{total_epochs}")

            for frame in tqdm(frame_ticks):
                input_signal = frame.flatten()
                input_signal = np.append(
                    input_signal,
                    self.encodings[frame_index],
                )
                input_signal = np.sum(
                    [input_signal, recurrent_signal],
                    axis=0,
                )
                recurrent_signal = self.step(input_signal)

            if frame_index % 4 == 0:
                self.weights.prune_weights(threshold=1e-4)

    def recall(self, encoding, num_ticks=8):
        recall_input = np.zeros(shape=self.num_neurons)
        recall_input[-self.encodings.shape[0]:] = encoding
        recall_spikes = np.zeros(shape=self.num_neurons)
        signals = recall_input

        for tick in range(num_ticks):
            spike_cache = []
            if tick > 0:
                recall_input = np.sum(
                    [recall_input, signals],
                    axis=0,
                )

            for ni in range(self.num_neurons):
                wp, _ = LIF.step(
                    self.neurons[ni],
                    recall_input[ni],
                )
                if np.isnan(wp):
                    raise FloatingPointError(
                        f"NaN during recall at tick={tick}, neuron={ni}"
                    )
                spike_cache.append(wp)

            spike_cache = np.array(
                spike_cache,
                dtype=np.float64,
            )
            spike_cache = self.weights.compute_spikes(
                spike_cache,
                self.thresh,
            )
            signals = self.weights.propagate_signals(
                spike_cache,
                method="sum",
            )
            recall_spikes = np.sum(
                [spike_cache, recall_spikes],
                axis=0,
            )

        return recall_spikes

    def infer(self, samples, encodings):
        fig, axes = plt.subplots(len(encodings), 3, squeeze=False)

        for i, encoding in enumerate(encodings):
            recall_spikes = self.recall(encoding, 255)
            recall_spikes = recall_spikes[
                : self.num_neurons - len(self.encodings)
            ]
            readout = np.reshape(
                recall_spikes[:1024],
                newshape=(32, 32),
            )
            readout = cv2.normalize(
                readout,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
            )

            groundtruth = cv2.normalize(
                samples[i],
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
            )
            diff = np.abs(samples[i] - readout)
            diff = cv2.normalize(
                diff,
                None,
                0,
                1,
                norm_type=cv2.NORM_MINMAX,
            )
            error = np.square(diff).mean()

            axes[i, 0].imshow(groundtruth, cmap="gray")
            axes[i, 0].set_title("Groundtruth")
            axes[i, 1].imshow(readout, cmap="gray")
            axes[i, 1].set_title("Recall")
            axes[i, 2].imshow(diff, cmap="viridis")
            axes[i, 2].set_title(f"Diff: {error}")

        fig.savefig("results.png", dpi=300)
        plt.show()


def main():
    loader = DataLoader()
    frames = loader.load_video("./lichen.mp4", num_repeats=32)
    encodings = np.fliplr(
        np.eye(len(frames), dtype=np.float64) * 500.0
    )

    frame_neurons = 32 * 32
    network = Network(
        num_neurons=frame_neurons,
        n_inputs=frame_neurons,
        encodings=encodings,
        dt=1.0,
        shape=(32, 32),
    )
    network.run(data_stream=frames)
    network.infer(
        samples=[frame[0] for frame in frames],
        encodings=encodings,
    )


if __name__ == "__main__":
    main()
