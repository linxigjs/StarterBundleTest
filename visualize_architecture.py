from pyimagesearch.nn.conv.lenet import LeNet
from keras.utils import plot_model

model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True, show_layer_names=True)