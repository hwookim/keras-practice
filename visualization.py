from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt

plt.show()

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))