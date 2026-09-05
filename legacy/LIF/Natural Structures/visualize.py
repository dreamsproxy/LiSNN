import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go

def WeightMatrix(path = "./logs/weight_logs.npy", trim_last: int = 0):
    weight_logs = []
    weight_matrix = np.load(path)
    with open("./logs/ids.txt", "r") as infile:
        neuron_ids = infile.read().split(",")
    if trim_last >= 1:
        weight_matrix = weight_matrix[trim_last:-1]
    for tick_slice in weight_matrix:
        df_tick_slice = pd.DataFrame(tick_slice, columns=neuron_ids)
        df_tick_slice.insert(0, "ID", neuron_ids)
        df_tick_slice.set_index(['ID'], inplace= True)
        for i in neuron_ids:
            if "Alpha" in i:
                df_tick_slice.drop([f"{i}"], inplace=True)
        df_tick_slice.transpose(copy=False)
        for i in neuron_ids:
            if "Alpha" in i:
                df_tick_slice.drop([f"{i}"], axis=1, inplace=True)
        weight_logs.append(df_tick_slice)
    frames = [
        go.Frame(
            data=go.Heatmap(
                z=frame.values,
                x=frame.columns,
                y=frame.index,
                colorscale="viridis"),
            name=i)
        for i, frame in enumerate(weight_logs)
    ]

    fig = go.Figure(data=frames[0].data, frames=frames).update_layout(
        updatemenus=[
            {
                "buttons": [{"args": [None, {"frame": {"duration": 5, "redraw": True}}],
                            "label": "Play", "method": "animate",},
                            {"args": [[None],{"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate", "transition": {"duration": 0},},],
                            "label": "Pause", "method": "animate",},],
                "type": "buttons",
            }
        ],
        # iterate over frames to generate steps... NB frame name...
        sliders=[{"steps": [{"args": [[f.name],{"frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate",},],
                            "label": f.name, "method": "animate",}
                            for f in frames],}],
        height=1000,
        width=1000,
        yaxis={"title": "Pre-Synaptic IDs", "tickangle":0},
        xaxis={"title": "Post-Synaptic IDs", "side": "top"},
        title_x=0.5,

    )
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.show()

def NeuronSpikes(file_type = ".npy"):
    import plotly.express as px
    spikes = np.load("./logs/neuron_spike_logs.npy")
    with open("./logs/ids.txt", "r") as infile:
        ids = infile.read().split(",")

    # Convert to dataframe
    spikes_df = pd.DataFrame(np.swapaxes(spikes, 0, 1), columns=ids)
    spikes_df.set_index(ids)

    fig = px.line(spikes_df, title="Neuron Spike")
    fig.show()

def NeuronPotentials():
    import plotly.express as px
    potentials = np.load("./logs/neuron_V_logs.npy")
    with open("./logs/ids.txt", "r") as infile:
        ids = infile.read().split(",")

    # Convert to dataframe
    potentials_df = pd.DataFrame(np.swapaxes(potentials, 0, 1), columns=ids)
    potentials_df.set_index(ids)
    print(potentials_df.head())

    fig = px.line(potentials_df, title="Neuron Potentials")
    fig.update_layout(
        yaxis={"title": "Voltage", "tickangle":90},
        xaxis={"title": "Ticks", "side": "bottom"},
        title_x=0.5,

    )
    fig.show()

def OutputActivity():
    import json
    from sklearn.preprocessing import minmax_scale

    with open("./output.json", "r") as jin:
        output_buffer = json.load(jin)
    
    keys = list(output_buffer.keys())
    outputs = []
    for k in keys:
        cache = output_buffer[k]
        cache = cache.split(", ")
        cache = [float(i) for i in cache]
        cache = [np.float64(i) for i in cache]
        outputs.append(cache)
    
    shortest = int(1024)
    
    for i in outputs:
        if len(i) < shortest:
            shortest = len(i)
    
    del cache

    for i, k in enumerate(outputs):
        outputs[i] = minmax_scale(k[:32], feature_range=(0.0, 255.0))
    outputs = np.array(outputs)

    #outputs = (outputs-np.min(outputs))/(np.max(outputs)-np.min(outputs))
    xy = int(np.sqrt(outputs.shape[0]))
    
    frames = []
    for tick_slice in range(len(outputs[0])):
        df_tick_slice = pd.DataFrame(np.reshape(np.copy(outputs[:, tick_slice]), (xy, xy)))

        #df_tick_slice.transpose(copy=False)
        
        frames.append(df_tick_slice)
    frames = [
        go.Frame(
            data=go.Heatmap(
                z=frame.values,
                x=frame.columns,
                y=frame.index,
                colorscale="gray",
                zmax=0,
                zmin=255),
                #zmax=-35,
                #zmin=-90),
            name=i)
        for i, frame in enumerate(frames)
    ]

    fig = go.Figure(data=frames[0].data, frames=frames).update_layout(
        updatemenus=[
            {
                "buttons": [{
                    "args": [None, {"frame": {"duration": 5, "redraw": True}}],
                    "label": "Play", "method": "animate",},{
                        
                    "args": [[None], {"frame": {"duration": 0,"redraw": False},
                        "mode": "immediate", "transition": {"duration": 0},},],
                    "label": "Pause", "method": "animate",},],
                
                "type": "buttons",
            }
        ],
        # iterate over frames to generate steps... NB frame name...
        sliders=[{"steps": [{"args": [[f.name],{"frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate",},],
                            "label": f.name, "method": "animate",}
                            for f in frames],}],
        height=1000,
        width=1000,
        yaxis={"title": "Output Layer Activity", "tickangle":0},
        #xaxis={"title": "Post-Synaptic IDs", "side": "top"},
        title_x=0.5,

    )
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.show()

if __name__ == "__main__":
    #WeightMatrix()
    OutputActivity()
    #NeuronPotentials()
    #NeuronSpikes()
    
