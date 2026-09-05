import numpy as np
from tqdm import tqdm
import random
import multiprocessing
import pandas as pd
from time import process_time
import Conv2D
import utils
from seed_generators import random_coords, generate_grids
import json

class LIF:
    def __init__(self, neuron_id: str, lif_init: str = "default", trim_lim: int = 10, verbose_log: bool = False) -> None:
        self.neuron_id = neuron_id
        
        # Define simulation parameters
        if lif_init == "default":
            self.tau_m = np.float64(1.5)  # Membrane time constant
            self.V_reset = np.float64(-75.0)  # Reset voltage
            self.V_threshold = np.float64(-55.0)  # Spike threshold
        elif lif_init == "random":
            self.tau_m = np.float64(random.uniform(0.000001, 0.999999))  # Membrane time constant
            self.V_reset = np.float64(random.uniform(-80.0, -70.0))  # Reset voltage
            self.V_threshold = np.float64(random.uniform(-55.0, -20.0))  # Spike threshold

        self.V = list()
        self.spike_log = list()
        self.spike_bool = False
        self.trim_lim = trim_lim
        
        self.verbose_log = verbose_log
        self.full_spike_log = list()
        
        self.full_debug = []
        self.full_debug.append("INIT")

    # Define a function to update the LIF neuron's state
    def update(self, current_input: np.float64 = np.float64(0)):
        # Trim essentially cuts the list when it is beyond a specified range
        # Required for STDP + Hebb fusion
        # See class methods that requires this value:
        #   Class: Network
        #       Func: get_correlation_term
        #       Func: adjust_weights
        #       Func: hebbian_optimization
        if len(self.spike_log) >= self.trim_lim:
            del self.spike_log[0]

        # If the voltage log is empty, assume it is at 0.0, then perform calculation
        if len(self.V) < 1:
            delta_V = (current_input - self.V_reset) / self.tau_m
            self.V.append(self.V_reset + delta_V)
            #self.V.append(self.V_reset)
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
    def __init__(self, neuron_keys:list = [], w_init: str = "default"):
        self.neuron_keys = neuron_keys
        
        self.n_neurons = len(neuron_keys)
        if w_init == "default":
            self.matrix = np.zeros(shape=(self.n_neurons, self.n_neurons))+np.float64(0.5)
        elif w_init == "random":
            self.matrix = np.random.default_rng().uniform(np.float64(0.33), np.float64(0.99), (self.n_neurons, self.n_neurons))
            #self.matrix = np.random.rand(self.n_neurons, self.n_neurons)
        else:
            e = "\n\n\tWeight init only takes 'zeros' or 'random'!\n\tDefault is zero.\n"
            raise Exception(e)
        self.matrix = pd.DataFrame(self.matrix, columns=neuron_keys, index=neuron_keys)

    # postproccess essentially removes the below types of connections:
    #   Input -> Input
    #   Itself -> itself (Recurrent)
    def postproccess(self):
        for k1 in self.neuron_keys:
            for k2 in self.neuron_keys:
                if k2 == k1:
                    self.matrix[k2][k1] = np.nan
                    self.matrix[k1][k2] = np.nan
                if "Alpha " in k1 and "Alpha " in k2:
                    self.matrix[k2][k1] = np.nan
                    self.matrix[k1][k2] = np.nan

    # Used for checking matrix (DEBUG)
    def PrintMatrix(self):
        print(self.matrix)
        print(self.matrix.shape)


def adjust_weights_local(k1, k2, w, correlation_term, hebbian_lr, weight_decay):

    # Adjusts the weight, that is inside self.weight_matrix
    # Originally was a lambda func call, but more ops were needed
    new_w: np.ndarray = np.dot(hebbian_lr, correlation_term)
    new_w += w
    new_w *= weight_decay

    return (k1, k2, new_w)


class Network:
    def __init__(self,
                 n_neurons: int,
                 image_input: bool,
                 resolution: int,
                 lif_init: str = "default",
                 w_init: str = "default",
                 hist_lim: int = 10,
                 verbose_logging:bool = False) -> None:

        # Network Parameters
        self.hebbian_lr     = np.float64(0.001)
        self.weight_decay   = np.float64(0.0005)
        self.weight_penalty = np.float64(0.5)

        self.n_neurons      = n_neurons
        
        # Where all neuron's keys and their LIF objects (KEY : child object)
        self.LIFNeurons     = dict()
        
        # Boolean, change to False and there will be no input image
        # If False, change input_neurons: list in class method: "step()"
        # If False, remove "step_vision()" calls
        self.image_input    = image_input
        self.resolution     = resolution

        self.w_init         = w_init
        self.weight_log     = []
        
        # Essentially a backlog / to-do list for the network
        # If any neurons fired at step 0 completion:
        #       At next step, these neurons will be simulated first
        self.signal_cache   = dict()

        # Neuron parameters
        self.hist_lim       = hist_lim
        self.lif_init       = lif_init

        self.verbose_logging = verbose_logging
        self.step_debug_log = []

    def InitNetwork(self):
        # WARNING START
        # No need to touch this area, unless you absolutly want to modify it
        
        # Initialize data reduction system: See file ./Conv2D.py
        self.image_sensor = Conv2D.Conv2D(
            resolution = self.resolution, kernel_size = 3)
        self.n_inputs = self.image_sensor.n_sensors

        # Generating key : coordinate pairs for visualization
        coordinates = dict()
        coordinates_dump = dict()
        input_points = generate_grids(256, 256, 1024, s = 256, r = self.n_inputs)
        input_points = np.reshape(input_points, (input_points.shape[0] * input_points.shape[1], 3))
        for i in range(self.n_inputs):
            self.LIFNeurons[f"Alpha {i}"] = LIF(
                i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                verbose_log=self.verbose_logging)
            f = str(tuple(input_points[i])).replace("(", "").replace(")", "")
            coordinates[f"Alpha {i}"] = input_points[i]
            coordinates_dump[f"Alpha {i}"] = f
            

        neuron_points = random_coords(self.n_neurons, x_lim=1024, y_lim=1024, z_lim=512)
        for i in range(self.n_neurons):
            self.LIFNeurons[str(i)] = LIF(
                i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                verbose_log=self.verbose_logging)
            f = str(tuple(neuron_points[i])).replace("(", "").replace(")", "")
            coordinates[str(i)] = input_points[i]
            coordinates_dump[str(i)] = f

        self.n_neurons += self.image_sensor.n_sensors
        self.neuron_keys = list(self.LIFNeurons.keys())

        # WARNING END
        
        # See WeightMatrix class:
        self.weightsclass = WeightMatrix(self.neuron_keys, self.w_init)
        self.weightsclass.PrintMatrix()
        self.weight_matrix = self.weightsclass.matrix

        return coordinates, coordinates_dump

    def get_correlation_term(self, n1: str, n2: str):
        # Calculates the activity correlation between both
        # neurons.
        pre_ss = self.LIFNeurons[n1].spike_log
        post_ss = self.LIFNeurons[n2].spike_log
        if len(pre_ss) == len(post_ss) and len(pre_ss) >= 1:
            correlation_term = np.dot(pre_ss, post_ss)
        else:
            correlation_term = 0
        return correlation_term

    def adjust_weights(self, w, correlation_term):
        # Adjusts the weight, that is inside self.weight_matrix
        # Originally was a lambda func call, but more ops were needed
        new_w = np.dot(self.hebbian_lr, correlation_term)
        new_w += w
        new_w *= self.weight_decay

        return new_w

    # IDK how to explain this part, sorry!
    def hebbian_optimization(self, args):
        df, pair_list = args
        for row_index, col_name, func in pair_list:
            # Use vectorized operation to update elements efficiently
            c_term = self.get_correlation_term(row_index, col_name)
            df.at[row_index, col_name] = func(
                df.at[str(row_index), str(col_name)],
                correlation_term = c_term)

    def PrepSignals(self, fired_list:list):
        # The backlog handler
        # Called as a data formatter + handler
        # Handler stores to a backlog/todo list
        # Data is dumped to self.signal_cache
        # Data is then accessed at new step call
        cache_dict = dict()
        for fired_k in fired_list:
            for other_k in self.neuron_keys:
                if other_k != fired_k:
                    weight = self.weight_matrix[fired_k][other_k]
                    signal = self.LIFNeurons[fired_k].V_threshold
                    cache_dict[str(other_k)] = (signal*weight)

        return cache_dict

    def step(self, step_number = 0, input_current = np.float64(0.0000), input_neuron:str = "0", fired_input_keys = []):
        if self.verbose_logging:
            step_start = process_time()
            self.step_debug_log.append(f"\nStep {step_number} [START]")

        if input_current != np.float64(0.0000):
            if self.verbose_logging:
                self.step_debug_log.append("\n\tInput Current Detected!")
            input_current = input_current / np.pi
        else:
            if self.verbose_logging:
                self.step_debug_log.append("\tNo Input\n")
        
        fired_neuron_keys = fired_input_keys
        signal_keys = list(self.signal_cache.keys())
        if self.image_input:
            signal_keys = [s_k for s_k in signal_keys if "Alpha" not in s_k]
            filtered_keys = [n_k for n_k in self.neuron_keys if "Alpha" not in n_k]

        self.step_debug_log.append(f"\n\tAll Neuron Keys:\n\t\t{self.neuron_keys}")
        if len(signal_keys) > 0:
            if self.verbose_logging:
                self.step_debug_log.append(f"\n\tBacklogs detected!")
                self.step_debug_log.append(f"\n\tBacklog Signal Keys: {signal_keys}")
                self.step_debug_log.append(f"\n\tBacklog [START]")
            for receiver_neuron in signal_keys:
                r_k = receiver_neuron
                recieved_signal = self.signal_cache[str(r_k)]
                neu = self.LIFNeurons[r_k]
                if str(r_k) == str(input_neuron):
                    neu.update(input_current+recieved_signal)
                    if self.verbose_logging:
                        self.step_debug_log.append(f"\n\t\tUpdate: input neuron {r_k}")
                else:
                    if self.verbose_logging:
                        self.step_debug_log.append(f"\n\t\tUpdate: neuron {r_k}")
                    neu.update(np.float64(recieved_signal))

                if neu.spike_bool:
                    fired_neuron_keys.append(r_k)
                if self.verbose_logging:
                    self.step_debug_log.append(f"\n\t\t\tRemoved: {r_k} from {filtered_keys}")
                filtered_keys.remove(r_k)

            if self.verbose_logging:
                self.step_debug_log.append(f"\n\tBacklog [ENDED]")
        if self.verbose_logging:
            self.step_debug_log.append(f"\n\tNormal Step [START]")
        for k in filtered_keys:
            if self.verbose_logging:
                self.step_debug_log.append(f"")
            neu = self.LIFNeurons[k]
            if str(k) == str(input_neuron):
                if self.verbose_logging:
                    self.step_debug_log.append(f"\n\t\tUpdate: input neuron {k}")
                neu.update(input_current)
            else:
                if self.verbose_logging:
                    self.step_debug_log.append(f"\n\t\tUpdate: neuron {k}")
                neu.update(np.float64(0))

            if neu.spike_bool:
                fired_neuron_keys.append(k)

        if len(fired_neuron_keys) >= 1:
            if self.verbose_logging:
                self.step_debug_log.append(f"\n\tThese Neurons Fired: {fired_neuron_keys}")
            self.signal_cache = self.PrepSignals(fired_neuron_keys)
        else:
            if self.verbose_logging:
                self.step_debug_log.append(f"\n\t\tNO Neurons Fired")
        if self.verbose_logging:
            self.step_debug_log.append(f"\n\tNormal Step [ENDED]")
            self.step_debug_log.append(f"\n\tGLOBAL HEBBIAN WEIGHT OPT [START]")
            hebb_start = process_time()

        # Move correlation term here
        # Prepare data that can be passed to multiprocessing
        weight_update_key_pairs = []
        for k1 in self.neuron_keys:
            for k2 in self.neuron_keys:
                if k1 != k2:
                    weight_update_key_pairs.append((
                        k1, k2, 
                        self.weight_matrix.at[k1, k2],
                        self.get_correlation_term(k1, k2),
                        self.hebbian_lr,
                        self.weight_decay,
                    ))
        # Modify hebbian_optimization method applied to each
        # element to use multiprocessing
        with multiprocessing.Pool() as pool:
            for result in pool.starmap(adjust_weights_local, weight_update_key_pairs):
                k1, k2, new_w = result
                self.weight_matrix.at[k1, k2] = new_w

        if self.verbose_logging:
            hebb_end = process_time()
            self.step_debug_log.append(f"\n\tGLOBAL HEBBIAN WEIGHT OPT [ENDED]")
            self.step_debug_log.append(f"\n\t\tTOOK {hebb_end - hebb_start}")
            self.step_debug_log.append(f"\n\tWeight Normalization [START]")

        norm_start = process_time()
        # Normalize the weights to prevent uncontrolled growth
        self.weight_matrix /= np.max(np.abs(self.weight_matrix))
        # Scale the weights to keep it between 0.0 and 1.0
        self.weight_matrix = (self.weight_matrix-np.min(self.weight_matrix))/(np.max(self.weight_matrix)-np.min(self.weight_matrix))
        norm_end = process_time()
        if self.verbose_logging:
            self.step_debug_log.append(f"\n\tWeight Normalization [ENDED]")
            self.step_debug_log.append(f"\n\t\tTOOK {norm_end - norm_start}")

        # Copy weight matrix to a logger
        if self.verbose_logging:
            self.step_debug_log.append(f"\n\tWeight Logging [START]")
        self.weight_log.append(np.copy(self.weight_matrix.to_numpy()))
        if self.verbose_logging:
            self.step_debug_log.append(f"\n\tWeight Logging [ENDED]")

        # NOTE: WARNING START
        #
        # NOTE: When running on long periods of time!
        # NOTE: spawn a separate process to write the logs!
        #
        # NOTE: WARNING END
        del fired_neuron_keys
        del filtered_keys
        if self.verbose_logging:
            step_end = process_time()
            self.step_debug_log.append(f"\nStep {step_number} [END]\n")
            self.step_debug_log.append(f"\nStep {step_number} TOOK {step_end - step_start}\n")

    def step_vision(self, current_vector: list):
        fired_cache = []
        for input_id in range(self.image_sensor.n_sensors):
            pixel_index = input_id
            pixel = current_vector[pixel_index]
            input_id = str(input_id)
            neu = self.LIFNeurons[f"Alpha {input_id}"]
            neu.update(pixel)
            if neu.spike_bool:
                fired_cache.append(f"Alpha {input_id}")
        return fired_cache

    def RunVision(self, ticks):
        import cv2
        from glob import glob
        images = [f for f in glob("./SAMPLES/**")]

        img = cv2.imread("./SAMPLES/sample.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        
        current_data = utils.to_current(img)
        current_vector = utils.to_vector(current_data)
        img_ticker = 0
        for i in tqdm(range(ticks)):
            fired = []
            with multiprocessing.Pool(10) as pool:
                for fired_cache in pool.imap_unordered(self.step_vision, (current_vector, )):
                    fired += fired_cache
            input_data = np.float64(-55.0)
            self.step(i, input_current = input_data,
                     fired_input_keys=fired)
            img_ticker += 1

    def SaveWeightTables(self, mode = "npy"):
        if mode == "npy":
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))
            np.save("./logs/weight_logs.npy", self.weight_log)
        else:
            print("No save format was defined, saving in .npy format!")
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))
            np.save("./logsweight_logs.npy", self.weight_log)

    def SaveNeuronPotentials(self):
        format_cache = []
        for k in list(self.neuron_keys):
            format_cache.append(np.asarray(self.LIFNeurons[k].V))
        format_cache = np.asarray(format_cache)
        np.save("./logsneuron_V_logs.npy", format_cache)

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
        np.save("./logs/neuron_spike_logs.npy", format_cache)

def EuclidianDistance(coord_A, coord_B):
    dist = [(a - b)**2 for a, b in zip(coord_A, coord_B)]
    dist = np.sqrt(sum(dist))
    return dist

if __name__ == "__main__":
    snn = Network(
        n_neurons = 64,
        lif_init = "random",
        w_init="default",
        hist_lim=21,
        verbose_logging = True,
        image_input=True,
        resolution=256)

    coords_dict, coords_dump = snn.InitNetwork()
    # Dump ID : Coord pair to json
    with open("./logs/coordinates.json", "w") as outfile:
        json.dump(coords_dump, outfile)

    # Cross Calc
    for k1 in snn.neuron_keys:
        for k2 in snn.neuron_keys:
            if k2 != k1:
                dist = EuclidianDistance(coords_dict[k2], coords_dict[k1])
                snn.weight_matrix[k2][k1] = dist

    snn.weightsclass.postproccess()
    snn.weightsclass.PrintMatrix()

    snn.RunVision(1)
    snn.SaveWeightTables()

    # Dump cols and rows
    with open("./logs/ids.txt", "w") as outfile:
        outfile.write(",".join(snn.neuron_keys))
    with open("./logs/debug.log", "w") as outfile:
        for debug_msg in snn.step_debug_log:
            if isinstance(debug_msg, list):
                for process_msg in debug_msg:
                    outfile.writelines(process_msg)
            else:
                outfile.writelines(debug_msg)