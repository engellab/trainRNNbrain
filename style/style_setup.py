import typing as T
import os
import matplotlib.colors as mcolors
import matplotlib.font_manager as mfonts
from trainRNNbrain.utils import get_project_root
from matplotlib import pyplot as plt

def setup_fonts(fontpaths, verbose: bool = False, **kwargs):
    for font in sorted(mfonts.findSystemFonts(fontpaths=fontpaths, **kwargs)):
        mfonts.fontManager.addfont(font)
        if verbose: print(f'font:\t{os.path.basename(font)}')


def setup_colors(themepaths, verbose: bool = False):
    for themepath in themepaths:
        with open(themepath, 'rt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('#'): continue
                names, value = line.split(':')
                value = value.strip()
                for name in names.split(','):
                    name = name.strip()
                    mcolors._colors_full_map[name] = value  # type: ignore
                    if verbose: print(f'color:\t{name:16s} {value:}')

setup_fonts([os.path.join(get_project_root(), "style", "assets", "fonts")], verbose=False)
setup_colors([os.path.join(get_project_root(), "style", "assets", "styles", "scientific.txt")], verbose=False)
plt.style.use([os.path.join(get_project_root(), 'style', 'assets', 'styles', 'scientific.mplstyle'), {'figure.dpi': 144}])