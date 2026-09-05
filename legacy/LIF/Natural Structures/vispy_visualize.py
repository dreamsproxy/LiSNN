import numpy as np
from vispy import scene, app, visuals
import pandas as pd
import json

def load_weights(path = "./logs/weight_logs.npy"):
    weight_logs = []
    weight_matrix = np.load(path)
    with open("./logs/ids.txt", "r") as infile:
        neuron_ids = infile.read().split(",")

    for tick_slice in weight_matrix:
        df_tick_slice = pd.DataFrame(tick_slice, columns=neuron_ids)
        df_tick_slice.insert(0, "ID", neuron_ids)
        df_tick_slice.set_index(['ID'], inplace= True)
        df_tick_slice.transpose(copy=False)
        weight_logs.append(df_tick_slice)
    return weight_logs

def to_dict(weight_logs):
    latest = weight_logs[-1]
    ids = list(latest.columns)
    
    weights_dict = dict()
    for k in ids:
        temp_dict = latest[k].to_dict()
        for nest_k in list(temp_dict.keys()):
            if temp_dict[nest_k] == np.nan:
                del temp_dict[nest_k]
            if "Input " in nest_k and "Input " in k:
                del temp_dict[nest_k]
            if "Output " in nest_k and "Output " in k:
                del temp_dict[nest_k]
            if "Input " in nest_k and "Output " in k:
                del temp_dict[nest_k]
            if "Output " in nest_k and "Input " in k:
                del temp_dict[nest_k]
        temp_dict = {str(ke):np.float64(v) for ke, v in temp_dict.items() if v != np.nan}
        try:
            del temp_dict[str(k)]
        except:
            continue
        weights_dict[str(k)] = temp_dict
    #print(weights_dict)
    return weights_dict

def load_coords(path = "./logs/coordinates.json"):
    with open(path, "r") as infile:
        temp_dict = json.load(infile)
    coordinates = dict()
    for id in list(temp_dict.keys()):
        formated = [np.float64(ax) for ax in temp_dict[id].split(", ")]
        coordinates[id] = tuple(formated)
    return coordinates

def adjust_coordinates(coordinates, nested_weights, ignore_z = False):
    nodes = list(coordinates.keys())
    node_positions = np.array(list(coordinates.values()))
    for source, targets in nested_weights.items():
        if "Input " not in source or "Output " not in source:
            if "Input " not in source and "Output " not in source:
                source_index = nodes.index(source)
                for target, weight in targets.items():
                    #if weight >= 0.99:
                        target_index = nodes.index(target)
                        # Maintain Z axis
                        if ignore_z:
                            z = node_positions[source_index][2]
                        #node_positions[source_index] += (node_positions[target_index] - node_positions[source_index]) * np.float64(np.float64(weight) * np.float64(0.5))
                        node_positions[source_index] += (node_positions[target_index] - node_positions[source_index]) * np.float64(weight)
                        if ignore_z:
                            node_positions[source_index][2] = z

    return dict(zip(nodes, node_positions))

coordinates = load_coords()
nested_weights = load_weights()
nested_weights = to_dict(nested_weights)

coordinates = adjust_coordinates(coordinates, nested_weights, ignore_z=False)

# Extract node positions and edges
nodes = list(coordinates.keys())

node_positions = np.multiply(np.array(list(coordinates.values())), 1.5)
node_positions = np.array(list(coordinates.values()))


edges = []
for source, targets in nested_weights.items():
    for target, weight in targets.items():
        edges.append((nodes.index(source), nodes.index(target)))
canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', size=(1080, 720), show=True)

def on_key_press(event):
    """Callback function for key presses."""
    if event.key == 'R':
        # Reset the view to the original positions
        view.camera.center = np.mean(node_positions, axis=0)
    elif event.key == 'Right':
        # Rotate the view to the right
        view.camera.azimuth+=2.0  # Adjust the rotation angle as needed
        canvas.update()
    elif event.key == 'Left':
        # Rotate the view to the left
        view.camera.azimuth-=2.0  # Adjust the rotation angle as needed
        canvas.update()
    elif event.key == 'Up':
        # Rotate the view to the left
        view.camera.elevation+=2.0  # Adjust the rotation angle as needed
        canvas.update()
    elif event.key == 'Down':
        # Rotate the view to the left
        view.camera.elevation-=2.0  # Adjust the rotation angle as needed
        canvas.update()
    #elif event.key == "X":
# Connect the key press event to the callback function
canvas.events.key_press.connect(on_key_press)

view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Set the center of the camera to a specific point
center_point = np.mean(node_positions, axis=0)  # Set center to the mean of node positions
view.camera.center = center_point

# Create nodes
scatter = scene.visuals.Markers()
scatter.set_data(node_positions, edge_color='black', face_color='black', size=12)
view.add(scatter)

# Create edges
lines = scene.visuals.Line(pos=np.array(node_positions), color=(0.0, 0.0, 0.0, 0.5))
lines.set_data(node_positions[edges], width=0.001)
view.add(lines)

if __name__ == '__main__':
    app.run()