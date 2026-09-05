import numpy as np
from tqdm import tqdm
import random
import os
import multiprocessing
from pathlib import Path

class WeightMatrix:
    def __init__(self, n_neurons: int, w_init: str = "default"):
        self.n_neurons = n_neurons
        if w_init == "default":
            self.matrix = np.zeros(shape=(n_neurons, n_neurons))+np.float16(0.5)
        elif w_init == "random":
            self.matrix = np.random.rand(n_neurons, n_neurons)
        else:
            e = "\n\n\tWeight init only takes 'zeros' or 'random'!\n\tDefault is zero.\n"
            raise Exception(e)
        
        pairs = [[p, p] for p in range(n_neurons)]
        for p1, p2 in pairs:
            self.matrix[p1, p2] = 0

    def PrintMatrix(self):
        print(self.matrix)

class LIF:
    def __init__(self, neuron_id: str, lif_init: str = "default", trim_lim: int = 10, verbose_log: bool = False) -> None:
        self.neuron_id = neuron_id
        
        # Define simulation parameters
        if lif_init == "default":
            self.tau_m = np.float16(10.0)  # Membrane time constant
            self.V_reset = np.float16(0.0)  # Reset voltage
            self.V_threshold = np.float16(1.0)  # Spike threshold
        elif lif_init == "random":
            self.tau_m = np.float16(random.uniform(5, 15))  # Membrane time constant
            self.V_reset = np.float16(random.uniform(-0.314, 0.314))  # Reset voltage
            self.V_threshold = np.float16(random.uniform(1.0, 3.14))  # Spike threshold

        self.V = list()
        self.spike_log = list()
        self.spike_bool = False
        self.trim_lim = trim_lim
        
        self.verbose_log = verbose_log
        self.full_spike_log = list()

    # Define a function to update the LIF neuron's state
    def update(self, current_input: np.float16 = np.float16(0)):
        if len(self.spike_log) >= self.trim_lim:
            del self.spike_log[0]
        else:
            pass
        # If the voltage log is empty, assume it is at 0.0, then perform calculation
        if len(self.V) < 1:
            delta_V = (current_input - np.float16(0.000)) / self.tau_m
            self.V.append(np.float16(0.000) + delta_V)
        else:
            delta_V = (current_input - self.V[-1]) / self.tau_m
            self.V.append(self.V[-1] + delta_V)

        if self.V[-1] >= self.V_threshold:
            self.V[-1] = self.V_reset
            self.spike_log.append(self.V_threshold)
            self.spike_bool = True
            if self.verbose_log:
                self.full_spike_log.append(1)
        else:
            self.spike_log.append(self.V_threshold)
            self.spike_bool = False
            if self.verbose_log:
                self.full_spike_log.append(0)

class Network:
    def __init__(self, n_neurons: int, lif_init: str = "default", w_init: str = "default", hist_lim: int = 10) -> None:
        self.stdp_lr = 0.01
        self.hebbian_lr = 0.00004

        self.n_neurons = n_neurons
        self.LIFNeurons = dict()
        self.weightsclass = WeightMatrix(n_neurons, w_init)
        self.weight_matrix = self.weightsclass.matrix
        self.weight_log = []

        self.hist_lim = hist_lim
        self.lif_init = lif_init

    def InitNetwork(self):
        for i in range(self.n_neurons):
            self.LIFNeurons[i] = LIF(i, trim_lim=self.hist_lim, lif_init = self.lif_init)

    def Decay(self, n1: str, n2: str, factor: float):
        factor = np.float16(factor)
        old_weight = self.weight_matrix[n1, n2]
        self.weight_matrix[n1, n2] -= factor * old_weight

    def Hebbian(self, n1: str, n2: str):
        # Neurons that fire together,
        # Connects together.

        latest_pre_synaptic_spikes = self.LIFNeurons[n1].spike_log
        latest_post_synaptic_spikes = self.LIFNeurons[n2].spike_log

        correlation_term = np.dot(latest_pre_synaptic_spikes, latest_post_synaptic_spikes)
        new_weight = self.weight_matrix[n1][n2] + np.dot(self.hebbian_lr, correlation_term)
        if new_weight <= 0.25:
            self.Decay(n1, n2, factor=self.hebbian_lr)
        else:
            self.weight_matrix[n1][n2] += new_weight

    def step(self, input_current = np.float16(0.000), input_neuron:str = "0"):
        neuron_keys = list(self.LIFNeurons.keys())
        for k in neuron_keys:
            neu = self.LIFNeurons[k]
            if neu.neuron_id == str(input_neuron):
                neu.update(input_current)
            else:
                neu.update(np.float16(0))

        # Do Global Weight Update
        for k1 in neuron_keys:
            for k2 in neuron_keys:
                if k1 != k2:
                    self.Hebbian(k1, k2)

        # Normalize the weights to prevent uncontrolled growth
        self.weight_matrix /= np.max(np.abs(self.weight_matrix))
        # Scale the weights to keep it between 0.0 and 1.0
        self.weight_matrix = (self.weight_matrix-np.min(self.weight_matrix))/(np.max(self.weight_matrix)-np.min(self.weight_matrix))

        # Copy weight matrix to a logger
        self.weight_log.append(np.copy(self.weight_matrix))

        # NOTE: WARNING START
        #
        # NOTE: When running on long periods of time!
        # NOTE: spawn a separate process to write the logs!
        #
        # NOTE: WARNING END

    def SaveWeightTables(self, mode = "npy", save_to_dir = "", save_as_name = "weight_logs"):
        """Save weight history to disk

        mode: "csv", "npy"
        save_to_dir: str Which directory to save to
        save_as_name: str Save to file (Or sub directory)
        """
        if not save_as_name:
            save_as_name = "weight_logs"

        if mode == "csv":
            import pandas as pd
            cols = list(self.LIFNeurons.keys())
            # for tick, table in tqdm(enumerate(self.weight_log), total=len(self.weight_log)):
            for tick, table in enumerate(self.weight_log):
                frame = pd.DataFrame(table)
                frame.columns = cols
                frame.set_index(cols)

                # frame.to_csv(f"./weight_logs/{tick} WM.csv")
                to_path = Path(".") / save_to_dir / save_as_name
                to_path.mkdir(exist_ok=True, parents=True)
                frame.to_csv(str(to_path / f"{tick} WM.csv"))
                
        elif mode == "npy":
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))

            # np.save("./weight_logs.npy", self.weight_log)
            to_path = Path(".") / save_to_dir
            to_path.mkdir(exist_ok=True, parents=True)
            np.save(str(to_path / "{}.npy".format(save_as_name)), self.weight_log)

        else:
            print("No save format was defined, saving in .npy format!")
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))

            # np.save("./weight_logs.npy", self.weight_log)
            to_path = Path(".") / save_to_dir
            to_path.mkdir(exist_ok=True, parents=True)
            np.save(str(to_path / "{}.npy".format(save_as_name)), self.weight_log)
    
    def SaveNeuronPotentials(self):
        format_cache = []
        for k in list(self.LIFNeurons.keys):
            format_cache.append(np.asarray(self.LIFNeurons[k].V))
        format_cache = np.asarray(format_cache)
        np.save("./neuron_V_logs.npy", format_cache)
    
    def SaveNeuronSpikes(self):
        # Check if the neurons are logged verbosely
        if len(self.LIFNeurons[list(self.LIFNeurons.keys)[0]]) <= 1:
            e = Exception("Neurons were not initialized with 'verbose_log' to 'True';!")
            raise e
        format_cache = []
        for k in list(self.LIFNeurons.keys):
            format_cache.append(np.asarray(self.LIFNeurons[k].full_spike_log))
        format_cache = np.asarray(format_cache)
        np.save("./neuron_spike_logs.npy",format_cache)
        

def neuron_process(args) -> None:

    # Unpack parameters
    save_mode: str = args[0]
    save_to_dir: str = args[1]
    save_as_name: str = args[2]
    network_params: dict = args[3]

    snn = Network(**network_params)
    snn.InitNetwork()

    total_steps = 1000
    for i in range(total_steps):
        if i == random.randint(0, total_steps):
            snn.step(input_current= np.float16(10.0), input_neuron=0)
        else:
            snn.step(input_current= np.float16(0.00), input_neuron=0)
    snn.SaveWeightTables(save_mode, save_to_dir, save_as_name)

    return

def main() -> None:

    with multiprocessing.Pool(4) as p:
        save_mode: str = "csv"
        target_directory: str = "test_dir"
        neuron_param: dict = {"n_neurons": 128, "lif_init": "random", "w_init": "random", "hist_lim": 17}
        parallel_neuron_params = [
            # Neuron and their name
            [save_mode, target_directory, "name1", neuron_param],
            [save_mode, target_directory, "name2", neuron_param],
            [save_mode, target_directory, "name3", neuron_param],
            [save_mode, target_directory, "name4", neuron_param],
            [save_mode, target_directory, "name5", neuron_param],
            [save_mode, target_directory, "name6", neuron_param],
        ]
        for _ in tqdm(p.imap_unordered(neuron_process, parallel_neuron_params), total=len(parallel_neuron_params)):
            pass

    return

def old_main():
    # Original code in `__name__ == "__main__"` condition
    snn = Network(n_neurons = 128, lif_init = "random", w_init="random", hist_lim=17)
    snn.InitNetwork()

    for i in tqdm(range(1000)):
        if i == random.randint(0, 1000):
#        if i % 10 == 0:
            snn.step(input_current= np.float16(10.0), input_neuron= 0)
#        elif i % 30 == 0:
#            snn.step(input_current= np.float16(10.0), input_neuron= 0)
        else:
            snn.step(input_current= np.float16(0.00), input_neuron= 0)
    snn.SaveWeightTables()

if __name__ == "__main__":
    main()