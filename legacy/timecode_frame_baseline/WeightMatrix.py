import numpy as np
from numba import njit


class WeightMatrix(object):
    def __init__(
        self,
        num_neurons,
        pre_alpha,
        post_alpha,
        init_type="glorot_uniform",
    ):
        self.a_pre = np.float64(pre_alpha)
        self.a_post = np.float64(post_alpha)
        self.num_neurons = num_neurons
        self.weights = self._initialize_weights(init_type)
        self.hebb_weights = np.random.uniform(
            0.0,
            1.0,
            (self.num_neurons, self.num_neurons),
        ).astype(np.float64)
        self.embedding_weights = np.random.uniform(
            0.25,
            0.75,
            (self.num_neurons, 2),
        ).astype(np.float64)

    def _initialize_weights(self, init_type):
        limit = np.sqrt(6 / (self.num_neurons + self.num_neurons))
        return np.random.uniform(
            -limit,
            limit,
            (self.num_neurons, self.num_neurons),
        ).astype(np.float64)

    @property
    def weight_stats(self):
        print(self.weights.std())
        print(self.weights.mean())
        print(self.weights.min())
        print(self.weights.max())

    @staticmethod
    @njit(parallel=True)
    def compute_hebb_updates(post_spikes):
        return np.where(post_spikes > 0.0), np.where(post_spikes == 0.0)

    def update_weights_hebb(self, post_spikes, clip=False) -> None:
        increase, decrease = WeightMatrix.compute_hebb_updates(post_spikes)
        self.hebb_weights[increase] += self.a_post
        self.hebb_weights[decrease] -= self.a_pre
        if clip:
            self.hebb_weights = np.clip(self.hebb_weights, 1e-8, 1.0)

    @staticmethod
    @njit("f8[:], f8[:]", parallel=True)
    def compute_spikes(potentials, threshold):
        return np.maximum(0, potentials - threshold)

    @staticmethod
    @njit("f8[:, :], f8, f8, f8[:], f8[:], f8[:], f8[:]")
    def compute_stdp_updates(
        delta_t,
        a_pre,
        a_post,
        pre_spikes,
        post_spikes,
        tau_pre,
        tau_post,
    ):
        return np.where(
            delta_t > 0,
            a_pre
            * post_spikes[:, None]
            * np.exp(-delta_t / tau_pre),
            -a_post
            * pre_spikes[None, :]
            * np.exp(delta_t / tau_post),
        )

    def select_top_k_neurons(self, spikes, k=5):
        return np.argsort(spikes)[-k:]

    def select_bot_k_neurons(self, spikes, k=5):
        return np.argsort(spikes)[:k]

    def update_weights_stdp(
        self,
        pre_spikes,
        post_spikes,
        tau_pre,
        tau_post,
        clip=False,
    ) -> None:
        delta_t = post_spikes[:, None] - pre_spikes[None, :]
        stdp_update = WeightMatrix.compute_stdp_updates(
            delta_t,
            self.a_pre,
            self.a_post,
            pre_spikes,
            post_spikes,
            tau_pre,
            tau_post,
        )
        self.weights += stdp_update

    def update_weights_combined(
        self,
        pre_spikes,
        post_spikes,
        tau_pre,
        tau_post,
        clip=False,
        top_k=5,
        error=np.ndarray([]),
    ):
        top_neurons = self.select_top_k_neurons(
            post_spikes,
            k=top_k,
        ).tolist()

        restricted_post_spikes = np.zeros_like(post_spikes)
        restricted_post_spikes[top_neurons] = post_spikes[top_neurons]

        delta_t = (
            restricted_post_spikes[:, None]
            - pre_spikes[None, :]
        )
        stdp_update = WeightMatrix.compute_stdp_updates(
            delta_t,
            self.a_pre,
            self.a_post,
            pre_spikes,
            restricted_post_spikes,
            tau_pre,
            tau_post,
        )
        self.weights += stdp_update
        self.update_weights_hebb(
            restricted_post_spikes,
            clip=False,
        )

        self.weights = np.mean(
            [self.weights, self.hebb_weights],
            axis=0,
        )

        if clip:
            self.weights = np.clip(self.weights, 1e-8, 1)

        self.weights = self.weights / np.linalg.norm(
            self.weights,
            axis=1,
            keepdims=True,
        )

    def propagate_signals(self, spikes, method="sum"):
        inputs = np.dot(self.weights, spikes)
        if method == "sum":
            return inputs
        if method == "mean":
            return inputs / self.weights.shape[1]
        if method == "weighted":
            row_sums = (
                self.weights.sum(axis=1, keepdims=True) + 1e-8
            )
            return inputs / row_sums.squeeze()
        raise ValueError(
            "Invalid method. Choose from 'sum', 'mean', or 'weighted'."
        )

    def prune_weights(self, threshold=1e-4):
        self.weights[self.weights < threshold] = 0

    def get_weights(self):
        return self.weights
