import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def show_max_projection(vol, title=""):
    mip = vol.max(axis=0)
    plt.imshow(mip, cmap="gray")
    plt.title(title)
    plt.axis("off")

def overlay_paths(img, paths, color="lime", lw=0.5):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    for zs, ys, xs in paths.values():
        ax.plot(xs, ys, color=color, lw=lw)
    ax.axis("off")
    return fig
