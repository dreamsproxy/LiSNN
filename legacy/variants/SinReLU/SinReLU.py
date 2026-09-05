from activations import SineReLU
import numpy as np
from random import random
from scipy import stats
import tqdm
from w_init import glorot
from optimize import gradient_descent
import pandas as pd
from time import process_time as pt

def scale_data(data, input_float):
    """
    Scale the input data between (0, input_float].
    
    Parameters:
        data (numpy.ndarray): Input data to be scaled.
        input_float (float): The upper limit of the scale.
        
    Returns:
        numpy.ndarray: Scaled data.
    """
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    scaled_data *= input_float
    return scaled_data

class Node:
    def __init__(self) -> None:
        #self.decay = np.float32(float(1.0 - (random() * 0.01)))
        self.decay = 1.0
        self.activation = SineReLU(decay=self.decay)
        self.fire_state: bool = False
        self.fire_log: list = list()
        self.output: np.float32 = np.float32(0.0)

    def update(self, dt_data) -> tuple:
        output, fire = self.activation.activate(dt_data)
        self.output = output
        self.fire_state: bool = fire
        self.fire_log.append(self.output)
        return (output, fire)

    def params(self) -> tuple:
        activity_log = self.fire_log[-31:]
        return (self.decay, activity_log, self.fire_state, self.fire_log, self.output)

class Net:
    def __init__(self, n_inputs:int, n_hidden:int, n_outputs:int) -> None:
        self.n_inputs:int = n_inputs
        self.n_hidden:int = n_hidden
        self.n_outputs:int = n_outputs

        self.total_nodes:int = self.n_inputs + self.n_hidden + self.n_outputs
        self.input_keys = [str(i) for i in range(0, self.n_inputs)]
        start = int(self.total_nodes - self.n_outputs)
        self.output_keys = [str(i) for i in range(start, self.total_nodes)]
        #print(self.output_keys)
        #raise
        self.nodes:dict = dict()
        for i in range(self.total_nodes):
            self.nodes[str(i)] = Node()
        self.node_keys:list = [str(i) for i in list(self.nodes.keys())]

        self.w_loaded = False

        # Init weight mat with glorot (xavier) dist
        self.w_matrix:np.ndarray = glorot((self.total_nodes, self.total_nodes))
        
        #self.w_matrix = np.ones((self.total_nodes, self.total_nodes), dtype=np.float32)
        # Need to eliminate input and output connections:
        self.w_matrix[:self.n_inputs, :self.n_inputs] = np.nan
        self.w_matrix[-self.n_outputs:, -self.n_outputs:] = np.nan
        # Elinimate recurrent
        node_list:list = [i for i in range(self.total_nodes)]
        self.w_matrix[node_list, node_list] = np.nan

        self.backlog_cache:dict = dict()
        self.tick:float = 0.00
        self.dt:float = 0.10
        self.step_counter = 0

    def step(self, data) -> None:
        #s = pt()
        #print("\nSTEP START")
        #print(f"Batch {self.batch_step}")
        #print(f"PRE BACKLOG:\t{self.backlog_cache}")
        cache = []
        for k in self.node_keys:
            k = str(k)
            if k not in self.input_keys:
                cache.append(k)

        # Run Backlogs

        for i, k in enumerate(self.input_keys):
            sends, fire = self.nodes[str(k)].update(data[i])
            if len(self.backlog_cache.keys()) >= 1:
                sends += np.sum(list(self.backlog_cache.values()))

            if fire:
                for r_k in cache:
                    w = self.w_matrix[int(k)][int(r_k)]
                    r_k = str(r_k)
                    self.nodes[r_k].update(sends*w)

        self.backlog_cache.clear()
        for k in self.node_keys:
            out = self.nodes[str(k)].output
            fire = self.nodes[str(k)].fire_state
            if fire:
                self.backlog_cache[str(k)] = out
                #print(self.backlog_cache)
            #print(f"{k}\t{fire}\t{out}")
        #e = pt()
        #print(f"Took {e - s} ns!")
        #print(f"\nPOST BACKLOG:\t{self.backlog_cache}")
        #print("\nSTEP END\n")

    def weight_update(self, y_true: np.ndarray) -> np.float32:
        #if self.step_counter >= 2:
        #    bias_matrix = self.w_matrix
        #    for k1 in range(self.total_nodes):
        #        x:list = self.nodes[str(k1)].fire_log
        #        weight_pairs = list ()
        #        for k2 in range(self.total_nodes):
        #            y:list = self.nodes[str(k2)].fire_log
        #            weight_pairs.append(y)
        #        bias_matrix += np.multiply(stats.pearsonr(x, y).correlation, 0.99)
        #s = pt()
        y_preds = []
        for k in self.output_keys:
            y_preds.append(np.float32(np.mean(self.nodes[str(k)].fire_log)))

        y_preds = np.asarray(y_preds, dtype=np.float32)
        
        #y_preds = (y_preds - y_preds.min()) / (y_preds.max() - y_preds.min())
        
        self.w_matrix, error = gradient_descent.optimize_weights(y_preds, self.w_matrix, y_true, LR, num_iterations=1)
        #e = pt()
        #print(f"Took {e - s} seconds!")
        error = np.float32(error)
        #if self.step_counter >= 2:
        #    self.w_matrix += bias_matrix
        #self.w_matrix = (self.w_matrix-np.min(self.w_matrix))/(np.max(self.w_matrix)-np.min(self.w_matrix))

    def normalize_outputs(self) -> np.ndarray:
        # Normalize output of all OUTPUT nodes:
        # Scale: (-1, 1]
        # Method: Min max
        outputs = []
        for k in self.output_keys:
            outputs.append(self.nodes[str(k)].output)
        outputs = np.ndarray(outputs)
        outputs = (outputs-np.min(outputs))/(np.max(outputs)-np.min(outputs))
        return outputs

    def run(self, batched_data):
        try:
            #for batch in tqdm.tqdm(batched_data):
            for n, batch in enumerate(batched_data):
                print(f"\nEpoch: {n}/{len(batched_data)}")
                batch: pd.DataFrame
                # Run data through network
                for i in tqdm.trange(batch.shape[1]):
                    self.batch_step = 0
                    input_data = batch.iloc[i]
                    input_data = input_data.to_numpy()
                    # Repeat each row 8 times
                    for i in range(8):
                        self.step(input_data)
                        self.batch_step += 1
                error = self.weight_update(input_data)
                print(f"Error: {error}")
                self.step_counter += 1
        except KeyboardInterrupt:
            pass
        # END TRAIN & START SAVING
        np.save("./weights.npy", self.w_matrix)
        # Dump all NODE params as well
        node_params = dict()
        for k in self.node_keys:
            node_params[str(k)] = self.nodes[str(k)].params()

        import csv
        with open("node_params.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header row with keys
            writer.writerow(['Key'] + [f'Value_{i+1}' for i in range(len(next(iter(node_params.values()))))])
            # Write data rows
            for key, value in node_params.items():
                writer.writerow([key] + list(value))

    def load_weights(self):
        self.w_loaded = True
        self.w_matrix = np.load("./weights.npy")
    
    def test_model(self, test_batch) -> np.ndarray:
        for i, k in enumerate(self.input_keys):
            self.step(test_batch[i])
        y_preds = []
        for k in self.output_keys:
            y_preds.append(self.nodes[k].output)
        y_preds = np.asarray(y_preds, dtype=np.float32)
        return y_preds

if __name__ == "__main__":
    global LR
    LR = 0.0314
    train = pd.read_csv("./dataset/train.csv").drop(columns=['date'])
    #print(train['wind_speed'].min())
    #print(train['wind_speed'].max())
    #raise
    train=(train-train.min())/(train.max()-train.min())
    train *= np.e

    test = pd.read_csv("./dataset/test.csv").drop(columns=['date'])
    test=(test-test.min())/(test.max()-test.min())
    test *= np.e

    from preprocessing.batch import sliding_window
    train_batched = sliding_window.batch(train, 32, 29)
    test_batched: pd.DataFrame = sliding_window.batch(test, 64, 62)
    
    test_data = test_batched[0].to_numpy()

    net = Net(4, 32, 4)
    # Override Weights Matrix
    #net.load_weights()

    net.run(train_batched)
    net.test_model(test_data)