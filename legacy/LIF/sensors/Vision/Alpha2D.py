import numpy as np
from sklearn.preprocessing import minmax_scale
import cv2
import math
import pandas as pd
from tqdm import tqdm
from sensor import Vision

def AlphaKernel(tiles_list: list, k_size: int, verbose: bool):
    tile_count = k_size*k_size
    tile_alpha_counts = list()
    for t in tiles_list:
        tile_alpha_counts.append((t >= np.mean(t)).sum()/tile_count)

    alphas = np.asarray(tile_alpha_counts)
    if verbose:
        results = {
            "K-Size" : k_size,
            "Mean" : np.mean(alphas),
            "Variance" : np.var(alphas),
            "STD" : np.std(alphas)
            }
        return results
    else:
        return alphas

def KernelStats():
    kernel_tests = [x for x in range(3, 64)]
    results = []
    for k_size in tqdm(kernel_tests):
        vis_sens = Vision(kernel_size=k_size)
        img = vis_sens.Load("./sample.png")
        tiles = vis_sens.MakeTiles(img)
        #results.append(BrightnessKernel(tiles, verbose=True))
        results.append(AlphaKernel(tiles, k_size, verbose=True))
    results = pd.DataFrame.from_dict(results)
    results.to_csv("./kernel_test_results.csv")
    import plotly as py
    import plotly.graph_objs as go
    import plotly

    pd.options.plotting.backend = "plotly"

    df = pd.read_csv("./kernel_test_results.csv", index_col=0)

    mean_trace = go.Bar(x = df["K-Size"], y = df["Mean"], name="Mean")
    variance_trace = go.Bar(x = df["K-Size"], y = df["Variance"], name="Variance")
    std_trace = go.Bar(x = df["K-Size"], y = df["STD"], name="STD")

    fig = plotly.tools.make_subplots(rows=3, cols=1)
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = df["K-Size"],
            ticktext = df["K-Size"]
        )
    )
    fig.append_trace(mean_trace, 1, 1)
    fig.append_trace(variance_trace, 2, 1)

    fig.append_trace(std_trace, 3, 1)


    fig.show()
