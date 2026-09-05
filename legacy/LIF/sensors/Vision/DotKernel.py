import numpy as np
import cv2
import plotly as py
import plotly.graph_objs as go
import plotly
from sensor import Vision
from sklearn.preprocessing import minmax_scale

def Kernel(tiles: list, kernel: tuple, bias: float):
    kernel_values = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 1.0, 1.0, 0.5],
        [0.5, 1.0, 1.0, 0.5],
        [0.0, 0.5, 0.5, 0.0]
    ])
    dot_results = []
    for t in tiles:
        r = minmax_scale(np.matmul(t, kernel_values), feature_range=(-1.0, 1.0))
        dot_results.append(np.mean(r))
    return dot_results

if __name__ == "__main__":
    vis_sens = Vision(kernel_size=4)
    img = vis_sens.Load("./sample.png")
    tiles = vis_sens.MakeTiles(img)
    results = minmax_scale(Kernel(tiles, 4, 0), feature_range=(-70., -55.))
    results = results.reshape((64, 64))
    print(results)
    raise
    fig = plotly.tools.make_subplots(rows=2, cols=2)
    
    img = cv2.imread("./sample.png", cv2.IMREAD_GRAYSCALE)
    original_img_trace = go.Heatmap(z=img, colorscale="gray")
    fig.append_trace(original_img_trace, 1, 1)
    
    new_img_3 = DotKernel(img, 3, 0.0)
    new_img_3_trace = go.Heatmap(z=new_img_3, colorscale="gray", name="K-Size 3")
    fig.append_trace(new_img_3_trace, 1, 2)
    
    new_img_5 = DotKernel(img, 5, 0.0)
    new_img_5_trace = go.Heatmap(z=new_img_5, colorscale="gray")
    fig.append_trace(new_img_5_trace, 2, 1)
    
    new_img_9 = DotKernel(img, 9, 0.0)
    new_img_9_trace = go.Heatmap(z=new_img_9, colorscale="gray")
    fig.append_trace(new_img_9_trace, 2, 2)
    
    fig.show()
    cv2.imshow("New Image", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()