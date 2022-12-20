import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def get_colormaps():
    # define colors
    color1 = hsv_to_rgb([357.93 / 360, 84.06/100, 81.18/100]) # red
    color2 = hsv_to_rgb([226.73 / 360, 65/100, 64/100]) #blue
    color3 = hsv_to_rgb([246.16 / 360, 33 / 100, 74 / 100]) #bluish
    color4 = hsv_to_rgb([145.14 / 360, 93/100,63/100]) # green
    color5 = hsv_to_rgb([28.32 / 360, 87.35/100, 96.08/100]) # orange
    color6 = hsv_to_rgb([196.11 / 360, 78 / 100, 93 / 100]) #light blue
    color7 = hsv_to_rgb([305 / 360, 53.66 / 100, 64.31 / 100]) # magenta


    colors = [color1, color2, color3, color4, color5, color6, color7]
    white = (1, 1, 1, 1)

    # for i in range(7):
    #     plt.plot(np.arange(9)-1*i, color=eval(f"color{i+1}"), linewidth = 10)
    # plt.show()

    newcolors = np.zeros((256, 4))
    newcolors[:, -1] = 1
    newcolors[:128, 0] = np.linspace(color2[0], white[0], 128)
    newcolors[128:, 0] = np.linspace(white[0], color1[0], 128)
    newcolors[:128, 1] = np.linspace(color2[1], white[1], 128)
    newcolors[128:, 1] = np.linspace(white[1], color1[1], 128)
    newcolors[:128, 2] = np.linspace(color2[2], white[2], 128)
    newcolors[128:, 2] = np.linspace(white[2], color1[2], 128)
    cmp_light = ListedColormap(newcolors)
    return colors, cmp_light

if __name__ == '__main__':
    get_colormaps()