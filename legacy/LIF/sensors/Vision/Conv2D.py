import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
import tqdm
from glob import glob
class Conv2D:
    def __init__(self, kernel_size) -> None:
        self.logs = []
        self.first_step = True
        self.kernel_size = kernel_size
        self.strides = 2

        self.kernel = np.random.rand(self.kernel_size[0], self.kernel_size[1])
        self.kernel = -2 + (2 - (-2)) * self.kernel

        self.bias = -2 + (2 - (-2)) * np.random.rand(1)

    def cross_correlation(self, img):
        img_height, img_width = img.shape
        kernel_height, kernel_width = self.kernel.shape
        output_height = (img_height - kernel_height) // self.strides + 1
        output_width = (img_width - kernel_width) // self.strides + 1
        output = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                patch = img[
                    i * self.strides : i * self.strides + kernel_height,
                    j * self.strides : j * self.strides + kernel_width,
                ]
                output[i, j] = np.sum(patch * self.kernel)
        return output

    def Call(self, input_image):
        img = np.pad(input_image, ((1, 1), (1, 1)), mode="constant")
        conv_layer = []
        conv_layer.append(
            self.cross_correlation(img)
        )

        # Perform intermediate sums
        intermediate_sum = [].append(np.sum(conv_layer, axis=0))

        # Add bias to each intermediate sums
        bias_sum = []
        for i in range(1): 
            bias_sum.append(intermediate_sum[i] + self.bias[i])

        bias_sum = np.asarray(bias_sum)

        # Turn list into np.array and transposed to match input structure
        out_img = np.transpose(np.array(bias_sum), (1, 2, 0))
        biased_img = np.reshape(out_img, (out_img.shape[0], out_img.shape[1]))

        if self.first_step == False:
            old_img = np.reshape(self.logs[-1], (self.logs[-1].shape[0], self.logs[-1].shape[1]))
            #old_img = minmax_scale(old_img, feature_range=(0, 1))
            
            delta = np.subtract(biased_img, old_img)
            delta = np.sum([old_img, delta], axis=0)
            self.logs.append(delta)
            #biased_img = np.mean([biased_img, old_img], axis=0)
        else:
            self.logs.append(biased_img)
        self.first_step = False
        return biased_img

    def Get(self, last_only: bool):
        if last_only:
            return self.logs[-1]
        else:
            return self.logs

    def GetLayer(self):
        return self.logs

def main(epochs):
    # Load imgs
    paths = glob("./samples/**")
    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        images.append(img)
    
    img = cv2.imread("./samples/sample.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    # Define Params
    kernel_size = (3, 3)
    
    # Initialize Layers as dict
    layers = {
        "0":Conv2D(kernel_size),
        "1":Conv2D(kernel_size),
        "2":Conv2D(kernel_size),
        "3":Conv2D(kernel_size)}
    
    counter = 0
    #for e in range(epochs):
    for e in tqdm.trange(epochs):
        output_cache = []
        if counter >= len(images):
            counter = 0
            img = images[0]
        else:
            img = images[counter]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        for i, k in enumerate(list(layers.keys())):
            if i == 0:
                output_cache = layers[k].Call(img)
            else:
                output_cache = layers[k].Call(output_cache)
        counter+=1
    results = []
    for k in list(layers.keys()):
        current_layer = layers[k]
        results.append(current_layer.Get(True))

    return results, layers["0"].GetLayer()

def vis_results(out_imgs, epochs):
    import seaborn as sns
    fig, axes = plt.subplots(1, len(out_imgs), figsize=(11.7, 8.27))
    fig.suptitle(f"{len(out_imgs)} Layer Conv2D as a Data Reducer.\n{epochs} epochs")
    #axes[0].set_title('Input Image')
    # Hide X and Y axes label marks
    #axes[0].axis('off')

    for i, trace in enumerate(out_imgs):
        sns.heatmap(ax=axes[i], data=trace, square=True, cmap="gray", cbar=False)
        axes[i].set_title(f'Layer {i}\nSize {trace.shape}')
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    epochs = 128
    out, layer1_logs = main(epochs)
    # Trim layer1logs
    layer1_logs = layer1_logs[0:3]
    for i, e in enumerate(layer1_logs):
        e = minmax_scale(e, feature_range=(0, 255))
        cv2.imwrite(f"Epoch {i}.png", e)
    vis_results(out, epochs)
