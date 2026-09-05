import numpy as np
from tqdm import tqdm
import random
import os
import multiprocessing
import pandas as pd
from time import process_time
class LIF:
    def __init__(self, neuron_id: str, lif_init: str = "default", trim_lim: int = 10, verbose_log: bool = False) -> None:
        self.neuron_id = neuron_id
        
        # Define simulation parameters
        if lif_init == "default":
            self.tau_m = np.float16(1.5)  # Membrane time constant
            self.V_reset = np.float16(-75.0)  # Reset voltage
            self.V_threshold = np.float16(-55.0)  # Spike threshold
        elif lif_init == "random":
            self.tau_m = np.float16(random.uniform(0.000001, 0.999999))  # Membrane time constant
            self.V_reset = np.float16(random.uniform(-80.0, -70.0))  # Reset voltage
            self.V_threshold = np.float16(random.uniform(-55.0, -20.0))  # Spike threshold

        self.V = list()
        self.spike_log = list()
        self.spike_bool = False
        self.trim_lim = trim_lim
        
        self.verbose_log = verbose_log
        self.full_spike_log = list()
        
        self.full_debug = []
        self.full_debug.append("INIT")

    # Define a function to update the LIF neuron's state
    def update(self, current_input: np.float16 = np.float16(0)):
        if len(self.spike_log) >= self.trim_lim:
            self.full_debug.append("TRIM: called!")
            del self.spike_log[0]

        # If the voltage log is empty, assume it is at 0.0, then perform calculation
        if len(self.V) < 1:
            self.full_debug.append("\tDelta Op: V empty, calculated with zeros")
            delta_V = (current_input - self.V_reset) / self.tau_m
            self.V.append(self.V_reset + delta_V)
        else:
            self.full_debug.append("\tDelta Op: Normal Delta Calculation")
            delta_V = (current_input - self.V[-1]) / self.tau_m
            self.V.append(self.V[-1] + delta_V)

        if self.V[-1] >= self.V_threshold:
            self.full_debug.append("\tThresh: Spike detected, setting to reset V")
            self.V[-1] = self.V_reset
            self.spike_log.append(self.V_threshold)
            self.spike_bool = True
            if self.verbose_log:
                self.full_spike_log.append(1)
        else:
            self.full_debug.append("\tThresh: No spike detected, continuing as usual")
            self.spike_log.append(self.V_threshold)
            self.spike_bool = False
            if self.verbose_log:
                self.full_spike_log.append(0)
        self.full_debug.append(f"V: {self.V[-1]}")

class WeightMatrix:
    def __init__(self, neuron_keys: int, w_init: str = "default"):
        self.neuron_keys = neuron_keys
        self.n_neurons = len(neuron_keys)
        if w_init == "default":
            self.matrix = np.zeros(shape=(self.n_neurons, self.n_neurons))+np.float16(0.5)
        elif w_init == "random":
            self.matrix = np.random.rand(self.n_neurons, self.n_neurons)
        else:
            e = "\n\n\tWeight init only takes 'zeros' or 'random'!\n\tDefault is zero.\n"
            raise Exception(e)
        self.matrix = pd.DataFrame(self.matrix, columns=neuron_keys, index=neuron_keys)

class Network:
    def __init__(self,
                 n_neurons: int,
                 audio_input: bool,
                 image_input: bool,
                 lif_init: str = "default",
                 w_init: str = "default",
                 hist_lim: int = 10,
                 verbose_logging:bool = False) -> None:

        self.stdp_lr = 0.01
        self.hebbian_lr = 0.0001

        self.n_neurons = n_neurons
        self.LIFNeurons = dict()
        self.audio_input = audio_input
        self.image_input = image_input

        self.w_init = w_init
        self.weight_log = []
        
        self.signal_cache = dict()
        self.clear_signal_cache = False

        self.hist_lim = hist_lim
        self.lif_init = lif_init
        
        self.verbose_logging = verbose_logging
        
        self.step_debug_log = []

    def InitNetwork(self):
        for i in range(self.n_neurons):
            self.LIFNeurons[str(i)] = LIF(
                i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                verbose_log=self.verbose_logging)

        self.neuron_keys = list(self.LIFNeurons.keys())

        self.weightsclass = WeightMatrix(self.neuron_keys, self.w_init)
        self.weight_matrix = self.weightsclass.matrix

    def Decay(self, n1: str, n2: str, factor: float):
        factor = np.float16(factor)
        old_weight = self.weight_matrix[n1][n2]
        self.weight_matrix[n1][n2] -= factor * old_weight

    def Hebbian(self, n1: str, n2: str):
        latest_pre_synaptic_spikes = self.LIFNeurons[n1].spike_log
        latest_post_synaptic_spikes = self.LIFNeurons[n2].spike_log

        correlation_term = np.dot(latest_pre_synaptic_spikes, latest_post_synaptic_spikes)
        self.weight_matrix[n1][n2] += np.dot(self.hebbian_lr, correlation_term)
        self.Decay(n1, n2, factor=0.5)

    def PrepSignals(self, fired_list:list):
        # Attemp signal propagation on every step
        # If there were no fired neurons, skip.
        cache_dict = dict()
        for fired_k in fired_list:
            for other_k in self.neuron_keys:
                if fired_k != other_k:
                    weight = self.weight_matrix[fired_k][other_k]
                    signal = self.LIFNeurons[fired_k].V_threshold
                    cache_dict[str(other_k)] = (signal*weight)

        return cache_dict

    def step(self, input_current = np.float16(0.0000), input_neuron:str = "0", fired_input_keys = []):
        if input_current != np.float16(0.0000):
            input_current = input_current / np.pi

        fired_neuron_keys = fired_input_keys
        signal_keys = list(self.signal_cache.keys())
        if self.audio_input:
            signal_keys = [s_k for s_k in signal_keys if "Audio" not in s_k]
            filtered_keys = [n_k for n_k in self.neuron_keys if "Audio" not in n_k]
        if self.image_input:
            signal_keys = [s_k for s_k in signal_keys if "Pixel" not in s_k]
            filtered_keys = [n_k for n_k in self.neuron_keys if "Pixel" not in n_k]
        # We need to skip audio sensors as it is always ran first
        if len(signal_keys) > 0:
            for receiver_neuron in signal_keys:
                r_k = receiver_neuron
                recieved_signal = self.signal_cache[str(r_k)]
                print(recieved_signal)
                neu = self.LIFNeurons[r_k]
                if str(r_k) == str(input_neuron):
                    neu.update(input_current+recieved_signal)
                else:
                    
                    neu.update(np.float16(recieved_signal))

                if neu.spike_bool:
                    fired_neuron_keys.append(r_k)
                filtered_keys.remove(r_k)

        for k in filtered_keys:
            neu = self.LIFNeurons[k]
            if str(k) == str(input_neuron):
                neu.update(input_current)
            else:
                neu.update(np.float16(-55))

            if neu.spike_bool:
                fired_neuron_keys.append(k)

        if len(fired_neuron_keys) >= 1:
            self.signal_cache = self.PrepSignals(fired_neuron_keys)

        hebb_start = process_time()
        # Do Global Weight Update
        for k1 in self.neuron_keys:
            for k2 in self.neuron_keys:
                if k1 != k2:
                    self.Hebbian(k2, k1)
        hebb_end = process_time()

        # Normalize the weights to prevent uncontrolled growth
        self.weight_matrix /= np.max(np.abs(self.weight_matrix))
        # Scale the weights to keep it between 0.0 and 1.0
        self.weight_matrix = (self.weight_matrix-np.min(self.weight_matrix))/(np.max(self.weight_matrix)-np.min(self.weight_matrix))

        # Copy weight matrix to a logger
        self.weight_log.append(np.copy(self.weight_matrix.to_numpy()))
        del fired_neuron_keys
        del filtered_keys

    def run(self, ticks):
        for i in tqdm(range(ticks)):
            input_data = np.float16(-55.0)
            snn.step(input_current = input_data, input_neuron= "0",
                     fired_input_keys=[])

    def SaveWeightTables(self, mode = "npy"):
        if mode == "npy":
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))
            np.save("./weight_logs.npy", self.weight_log)
        else:
            print("No save format was defined, saving in .npy format!")
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))
            np.save("./weight_logs.npy", self.weight_log)
    
    def SaveNeuronPotentials(self):
        format_cache = []
        for k in list(self.neuron_keys):
            format_cache.append(np.asarray(self.LIFNeurons[k].V))
        format_cache = np.asarray(format_cache)
        np.save("./neuron_V_logs.npy", format_cache)
    
    def SaveNeuronSpikes(self):
        # Check if the neurons are logged verbosely
        len(self.LIFNeurons["0"].full_spike_log)
        if len(self.LIFNeurons["0"].full_spike_log) <= 1:
            e = Exception("Neurons were not initialized with 'verbose_log' to 'True' !")
            raise e
        format_cache = []
        for k in list(self.neuron_keys):
            format_cache.append(np.asarray(self.LIFNeurons[k].full_spike_log))
        format_cache = np.asarray(format_cache)
        np.save("./neuron_spike_logs.npy", format_cache)


if __name__ == "__main__":
    snn = Network(
        n_neurons = 16,
        lif_init = "default",
        w_init="random",
        hist_lim=17,
        verbose_logging = True,
        audio_input=False,
        image_input=True)
    snn.InitNetwork()
    print(snn.neuron_keys)
    snn.run(10)
    snn.SaveWeightTables()
    snn.SaveNeuronSpikes()
    snn.SaveNeuronPotentials()
    # Dump cols and rows
    with open("ids.txt", "w") as outfile:
        outfile.write(",".join(snn.neuron_keys))
