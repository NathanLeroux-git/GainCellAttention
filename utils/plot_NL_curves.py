import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import torch
import json
import sys, os
dir_name = os.getcwd()
sys.path.insert(0, dir_name)
from modules import model_gpt
font = {'size': 8}
rc('font', **font)
rcParams['mathtext.default'] = 'regular'  # Math subscripts and greek letters non-italic

file_out="./nonlinear_curves.json"

def return_ax_NL_fig(ax):
    linewidth = 1.5

    x = torch.tensor(0.9).unsqueeze(-1)
    y = torch.arange(0, 0.9, 0.01).unsqueeze(0)

    dot_product_methods = ["DRAM_MAC_temporal_encoding_surrogate_QK",
                        #    "x3_dot_product",
                        "x5_dot_product",
                        "sigmoid_dot_product",
                        "exponential_dot_product",
                        ]

    labels = ["gain cells",
            #   "x**3",
            "x**5",
            "sigmoid",
            "exponential",
            ]

    colors = ["darkgreen",
            "goldenrod",
            "lightslategray",
            "maroon",
            ]

    dot_product_methods.reverse()
    colors.reverse()

    dict = []
    
    for (dot_product_method, label, color) in zip(dot_product_methods, labels, colors):    
        function = getattr(model_gpt, dot_product_method).apply
        z = function(x, y, torch.tensor(1.))
        ax.plot(y.squeeze(), z.squeeze(), linewidth=linewidth, label=label, color=color)
        
        dict += {"function": label,
                 "x": x,
                 "y": y,
                 }
    
    with open(file_out, 'w') as f:
        json.dump(dict, f, indent=4)
    
    return ax

if __name__ == '__main__':
    centimeters = 1 / 2.54  # centimeters in inches  
    # width = 8 * centimeters
    # height = 6 * centimeters

    a = 1
    width = 2 * centimeters * a
    height = 1.5 * centimeters * a

    fig, ax = plt.subplots(figsize=(width, height))
    ax = return_ax_NL_fig(ax)
    ax.set_xlabel('Weight (a.u)')
    ax.set_ylabel('Output (a.u)')
    fig.tight_layout()
    file_out = './plots/NL_curves'
    for fmt in ['png', 'svg', 'pdf']:
        plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)