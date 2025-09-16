import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


def generate_time_label_array_gray(label, image_size=(32, 64), font_scale=2):
    import matplotlib.pyplot as plt

    fontsize = int(image_size[0] * font_scale)

    fig, ax = plt.subplots(figsize=(image_size[1] / 10, image_size[0] / 10), dpi=10)
    ax.set_facecolor("white")

    ax.text(
        0.5,
        0.5,
        str(label),
        color="black",
        fontsize=fontsize,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.axis("off")

    canvas = FigureCanvas(fig)
    canvas.draw()
    image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    image_pil = Image.fromarray(image_array[:, :, :3]).convert("L")
    return np.array(image_pil)
