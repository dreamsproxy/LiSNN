import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go

def WeightMatrix(path = "./weight_logs.npy"):
    weight_logs = []
    weight_matrix = np.load(path)
    with open("ids.txt", "r") as infile:
        neuron_ids = infile.read().split(",")

    for tick_slice in weight_matrix:
        df_tick_slice = pd.DataFrame(tick_slice, columns=neuron_ids)
        df_tick_slice.insert(0, "ID", neuron_ids)
        print(neuron_ids)
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
    raise

def NeuronSpikes(file_type = ".npy"):
    import plotly.express as px
    spikes = np.load("./neuron_spike_logs.npy")
    with open("ids.txt", "r") as infile:
        ids = infile.read().split(",")

    # Convert to dataframe
    spikes_df = pd.DataFrame(np.swapaxes(spikes, 0, 1), columns=ids)
    spikes_df.set_index(ids)
    print(spikes_df.head())

    fig = px.line(spikes_df, title="Neuron Spike")
    fig.show()

def NeuronPotentials():
    import plotly.express as px
    potentials = np.load("./neuron_V_logs.npy")
    with open("ids.txt", "r") as infile:
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

def Network(path = "./weight_logs.npy"):
    import igraph as ig
    def GenerateCoords(cx, cy, cz, radius, n_nodes=360):
        phi = np.linspace(0, 2 * np.pi, n_nodes)
        theta = np.linspace(0, np.pi, n_nodes)

        #theta, phi = np.meshgrid(theta, phi)

        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        return x, y, z
        
    weight_matrix = np.load(path)
    
    neuron_ids = [i for i in range(weight_matrix.shape[1])]
    link_map = {}
    Edges = []
    
    for n1 in neuron_ids:
        link_pairs = []
        for n2 in neuron_ids:
            if n1 != n2:
                link_pairs.append(n2)
        link_map[n1] = link_pairs
    
    for i in neuron_ids:
        pairs = link_map[i]
        for target in pairs:
            Edges.append((i, target))

    G = ig.Graph(Edges, directed=False)
    layt=G.layout('sphere', dim=3)

    x, y, z = GenerateCoords(0.5, 0.5, 0.5, radius=0.3, n_nodes=len(neuron_ids))
    
    for i in range(len(neuron_ids)):
        layt[i][0] = x[i]
        layt[i][1] = y[i]
        layt[i][2] = z[i]

    link_coords = []

    for i in range(len(neuron_ids)):
        link_coords.append((x[i], y[i], z[i]))
    
        
    # TODO
    # Get only the link cords where the weights are more than 0.09
    node_points = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        name='actors',
        marker = go.Marker(
            symbol='circle',
            size=13,
            colorscale="viridis",
            opacity = 0.8),
        text=neuron_ids,
        hoverinfo='text'  
    )
    
    fig = go.Figure(data=node_points)
    for start in link_coords:
        for end in link_coords:
            fig.add_trace(
                go.Scatter3d(
                    x=[start[0], end[0]], 
                    y=[start[1], end[1]], 
                    z=[start[2], end[2]],
                    mode='lines',
                    showlegend=False,
                    hoverinfo='none',
                    name=""
                ),
            )

    fig.show()
if __name__ == "__main__":
    WeightMatrix()
    #NeuronSpikes()
    #NeuronPotentials()
    #Network()