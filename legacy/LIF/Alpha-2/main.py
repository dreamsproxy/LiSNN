import numpy as np
from tqdm import tqdm
import random
import os
import multiprocessing
import pandas as pd
from time import process_time
import Conv2D
import utils
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
            del self.spike_log[0]

        # If the voltage log is empty, assume it is at 0.0, then perform calculation
        if len(self.V) < 1:
            #delta_V = (current_input - self.V_reset) / self.tau_m
            #self.V.append(self.V_reset + delta_V)
            self.V.append(self.V_reset)
        else:
            delta_V = (current_input - self.V[-1]) / self.tau_m
            self.V.append(self.V[-1] + delta_V)

        if self.V[-1] >= self.V_threshold:
            if self.spike_bool:
                self.V[-1] = self.V_reset
                self.spike_bool = False
            else:
                self.V[-1] = self.V_threshold
                self.spike_log.append(1.0)
                self.spike_bool = True
            if self.verbose_log:
                self.full_spike_log.append(1.0)
        else:
            self.spike_log.append(0.0)
            self.spike_bool = False
            if self.verbose_log:
                self.full_spike_log.append(0.0)

class WeightMatrix:
    def __init__(self, neuron_keys: int, w_init: str = "default"):
        self.n_neurons = len(neuron_keys)
        if w_init == "default":
            self.matrix = np.zeros(shape=(self.n_neurons, self.n_neurons))+np.float16(0.5)
        elif w_init == "random":
            self.matrix = np.random.rand(self.n_neurons, self.n_neurons)
        else:
            e = "\n\n\tWeight init only takes 'zeros' or 'random'!\n\tDefault is zero.\n"
            raise Exception(e)
        self.matrix = pd.DataFrame(self.matrix, columns=neuron_keys, index=neuron_keys)
        for k1 in neuron_keys:
            for k2 in neuron_keys:
                if k2 == k1:
                    self.matrix[k2][k1] = np.nan
    def PrintMatrix(self):
        print(self.matrix)
        print(self.matrix.shape)

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
        self.hebbian_lr = 0.01
        self.weight_decay = 0.5

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
        self.image_sensor = Conv2D.Conv2D(
            resolution = 256, kernel_size = 3)

        for i in range(self.image_sensor.n_sensors):
            self.LIFNeurons[f"Alpha {i}"] = LIF(
                i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                verbose_log=self.verbose_logging)

        for i in range(self.n_neurons):
            self.LIFNeurons[str(i)] = LIF(
                i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                verbose_log=self.verbose_logging)

        self.n_neurons += self.image_sensor.n_sensors
        self.neuron_keys = list(self.LIFNeurons.keys())

        self.weightsclass = WeightMatrix(self.neuron_keys, self.w_init)
        self.weightsclass.PrintMatrix()
        self.weight_matrix = self.weightsclass.matrix

    def get_correlation_term(self, n1: str, n2: str):
        pre_ss = self.LIFNeurons[n1].spike_log
        post_ss = self.LIFNeurons[n2].spike_log
        if len(pre_ss) == len(post_ss) and len(pre_ss) >= 1:
            correlation_term = np.dot(pre_ss, post_ss)
        else:
            correlation_term = 0
        return correlation_term

    def Decay(self, n1: str, n2: str, factor: float):
        factor = np.float16(factor)
        old_weight = self.weight_matrix[n1][n2]
        self.weight_matrix[n1][n2] -= factor * old_weight

    def og_hebbian(self, n1: str, n2: str):
        correlation_term = self.get_correlation_term(n1, n2)
        self.weight_matrix[n1][n2] += np.dot(self.hebbian_lr, correlation_term)
        self.Decay(n1, n2, factor=self.hebbian_lr)

    def adjust_weights(self, w, correlation_term):
        return w + correlation_term

    def vect_hebbian(self, args):
        df, pair_list = args
        for row_index, col_name, func in pair_list:
            # Use vectorized operation to update elements efficiently
            c_term = self.get_correlation_term(row_index, col_name)
            df.at[row_index, col_name] = func(
                df.at[str(row_index), str(col_name)],
                correlation_term = c_term)

    def PrepSignals(self, fired_list:list):
        # Attemp signal propagation on every step
        # If there were no fired neurons, skip.
        cache_dict = dict()
        for fired_k in fired_list:
            for other_k in self.neuron_keys:
                if other_k != fired_k:
                    weight = self.weight_matrix[fired_k][other_k]
                    signal = self.LIFNeurons[fired_k].V_threshold
                    cache_dict[str(other_k)] = (signal*weight)

        return cache_dict

    def step(self, input_current = np.float16(0.0000), input_neurons:list = [""], fired_input_keys = []):
        if self.verbose_logging:
            self.step_debug_log.append("\n\nStep [START]")

        if input_current != np.float16(0.0000):
            if self.verbose_logging:
                self.step_debug_log.append("\tInput Current Detected!")
            input_current = input_current / np.pi
        else:
            if self.verbose_logging:
                self.step_debug_log.append("\tNo Input\n")
        
        fired_neuron_keys = fired_input_keys
        signal_keys = list(self.signal_cache.keys())
        if self.image_input:
            signal_keys = [s_k for s_k in signal_keys if "Pixel" not in s_k]
            filtered_keys = [n_k for n_k in self.neuron_keys if "Pixel" not in n_k]
        # We need to skip sensors as it is always ran first
        self.step_debug_log.append(f"\tAll Neuron Keys: {self.neuron_keys}")
        self.step_debug_log.append(f"\tBacklog Signal Keys: {signal_keys}\n")
        if len(signal_keys) > 0:
            self.step_debug_log.append(f"\tBacklogs detected!")
            self.step_debug_log.append(f"\tBacklog [START]")
            for receiver_neuron in signal_keys:
                r_k = receiver_neuron
                recieved_signal = self.signal_cache[str(r_k)]
                neu = self.LIFNeurons[r_k]
                if str(r_k) == str(input_neurons):
                    neu.update(input_current+recieved_signal)
                    if self.verbose_logging:
                        self.step_debug_log.append(f"\t\tUpdate: input neuron {r_k}")
                else:
                    if self.verbose_logging:
                        self.step_debug_log.append(f"\t\tUpdate: neuron {r_k}")
                    neu.update(np.float16(recieved_signal))

                if neu.spike_bool:
                    fired_neuron_keys.append(r_k)
                if self.verbose_logging:
                    self.step_debug_log.append(f"\t\t\tRemoved: {r_k} from {filtered_keys}")
                filtered_keys.remove(r_k)

        if self.verbose_logging:
            self.step_debug_log.append(f"\tBacklog [ENDED]")
            self.step_debug_log.append(f"\n\tNormal Step [START]")
        for k in filtered_keys:
            if self.verbose_logging:
                self.step_debug_log.append(f"")
            neu = self.LIFNeurons[k]
            if str(k) == str(input_neurons):
                if self.verbose_logging:
                    self.step_debug_log.append(f"\t\tUpdate: input neuron {k}")
                neu.update(input_current)
            else:
                if self.verbose_logging:
                    self.step_debug_log.append(f"\t\tUpdate: neuron {k}")
                neu.update(np.float16(0))

            if neu.spike_bool:
                fired_neuron_keys.append(k)

        if len(fired_neuron_keys) >= 1:
            if self.verbose_logging:
                self.step_debug_log.append(f"\n\tThese Neurons Fired: {fired_neuron_keys}")
            self.signal_cache = self.PrepSignals(fired_neuron_keys)
        else:
            if self.verbose_logging:
                self.step_debug_log.append(f"\t\tNO Neurons Fired: {fired_neuron_keys}")
        if self.verbose_logging:
            self.step_debug_log.append(f"\tNormal Step [ENDED]")
            self.step_debug_log.append(f"\n\tGLOBAL HEBBIAN WEIGHT OPT [START]")

        print("OG HEBB START")
        og_hebb_start = process_time()
        # Do Global Weight Update
        for k1 in self.neuron_keys:
            for k2 in self.neuron_keys:
                if k1 != k2:
                    self.Hebbian(k1, k2)
        og_hebb_end = process_time()
        print("OG HEBB END")
        print(F"TOOK\t{og_hebb_end - og_hebb_start}\n")
        raise

        print("Vectorized HEBB START")
        vect_hebb_start = process_time()
        weight_update_key_pairs = []
        for k1 in self.neuron_keys:
            for k2 in self.neuron_keys:
                if k1 != k2:
                    weight_update_key_pairs.append((k1, k2, self.adjust_weights))
        vect_hebb_end = process_time()
        print("Vectorized HEBB END")
        print(F"TOOK\t{vect_hebb_end - vect_hebb_start}\n")
        
        if self.verbose_logging:
            self.step_debug_log.append(f"\tGLOBAL HEBBIAN WEIGHT OPT [ENDED]")
            self.step_debug_log.append(f"\t\tTOOK {vect_hebb_end - vect_hebb_start}")
            self.step_debug_log.append(f"\n\tWeight Normalization [START]")

        norm_start = process_time()
        # Normalize the weights to prevent uncontrolled growth
        self.weight_matrix /= np.max(np.abs(self.weight_matrix))
        # Scale the weights to keep it between 0.0 and 1.0
        self.weight_matrix = (self.weight_matrix-np.min(self.weight_matrix))/(np.max(self.weight_matrix)-np.min(self.weight_matrix))
        norm_end = process_time()
        if self.verbose_logging:
            self.step_debug_log.append(f"\tWeight Normalization [ENDED]")
            self.step_debug_log.append(f"\t\tTOOK {norm_end - norm_start}")

        # Copy weight matrix to a logger
        if self.verbose_logging:
            self.step_debug_log.append(f"\tWeight Logging [START]")
        self.weight_log.append(np.copy(self.weight_matrix.to_numpy()))
        if self.verbose_logging:
            self.step_debug_log.append(f"\tWeight Logging [ENDED]")

        # NOTE: WARNING START
        #
        # NOTE: When running on long periods of time!
        # NOTE: spawn a separate process to write the logs!
        #
        # NOTE: WARNING END
        del fired_neuron_keys
        del filtered_keys
        if self.verbose_logging:
            self.step_debug_log.append(f"Step [END]")

    def step_vision(self, current_vector: list):
        fired_cache = []
        for input_id in range(self.image_sensor.n_kernels):
            pixel_index = input_id
            pixel = current_vector[pixel_index]
            input_id = str(input_id)
            neu = self.LIFNeurons[f"Pixel {input_id}"]
            neu.update(pixel)
            if neu.spike_bool:
                fired_cache.append(f"Pixel {input_id}")
        return fired_cache

    def RunVision(self, ticks):
        import cv2
        img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        current_data = utils.to_current(img)
        current_vector = utils.to_vector(current_data)
        for i in tqdm(range(ticks)):
            #print("VISION STEP START")
            #fired_cache = snn.step_vision(current_vector)
            #print("VISION STEP ENDED")
            input_data = np.float16(-55.0)
            #print("NORMAL STEP START")
            snn.step(input_current = input_data, input_neuron= "0",
                     fired_input_keys=[])
            #print("NORMAL STEP ENDED")

    def SaveWeightTables(self, mode = "npy"):
        if mode == "npy":
            print(self.weight_log[-1])
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
        lif_init = "random",
        w_init="random",
        hist_lim=17,
        verbose_logging = True,
        audio_input=False,
        image_input=True)
    snn.InitNetwork()
    print(snn.neuron_keys)
    snn.RunVision(50)
    snn.SaveWeightTables()
    #snn.SaveNeuronSpikes()
    #snn.SaveNeuronPotentials()
    # Dump cols and rows
    with open("ids.txt", "w") as outfile:
        outfile.write(",".join(snn.neuron_keys))
    #for debug_msg in snn.step_debug_log:
    #    print(debug_msg)