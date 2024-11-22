import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator, FormatStrFormatter, LogFormatter, LogLocator, FuncFormatter, ScalarFormatter, NullFormatter
import numpy as np
import wandb
import json
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_NL_curves import return_ax_NL_fig

entity = "user"
project = "owt"
out_file_root = "./plots/"
# rcParams['text.usetex'] = True
font = {'size': 8}
rc('font', **font)
rcParams['mathtext.default'] = 'regular'  # Math subscripts and greek letters non-italic

def import_run(run_id):
    api = wandb.Api(overrides={
                           "project": project,       
                           "entity": entity,
                            })
    run = api.run(entity+'/'+project+'/'+run_id)
    return run

def plot_metrics_multi_run(
                           ax=None,
                           data_dict={},
                           metric="val/loss",
                           max_axis=1000,
                           runs=[],
                           perplexity=False,
                           log_scale=False,
                           yaxis_limit=None,
                           moving_average_window=None,
                           hide_labels=False,
                           shade_lim=None,
                           legend_on=False,
                           ):
        for r, run in enumerate(runs):
            name, label, color = run['name'], run['label'], run['color']
            data = data_dict[name]
            iterations, values = data['iterations'], data[metric]
            # if name=="DRAM_surrogate_stop_13000_ft_from_gpt2-LinearDRAMAttention_tbs_1920_decay_factor_0.00016_stop3000":
            #     iterations = np.array(iterations) + 3000
            if perplexity:
                values = np.exp(np.array(values))
            if moving_average_window is not None:
                values = np.convolve(values, np.ones(moving_average_window) / moving_average_window, mode='valid')
                iterations = iterations[:len(values)]  
            ax.plot(iterations, values, '-', c=color, linewidth=linewidth, label=label)
            
        # yaxis
        if perplexity:
            if yaxis_limit is not None:
                ax.set_ylim(yaxis_limit)
            if not(hide_labels):
                ax.set_ylabel('Perplexity', fontsize=font['size'])  
            if log_scale:
                ax.set_yscale("log")         
                y_major = LogLocator(base = 10) 
                y_minor = LogLocator(base = 10, subs =np.arange(20, 40).tolist()) 
                ax.yaxis.set_major_locator(y_major)
                ax.yaxis.set_minor_locator(y_minor)            
                ax.set_yticks([20, 30, 40])                 
                ylabels = [f'{x:.0f}' for x in ax.get_yticks()]
                ax.set_yticklabels(ylabels)
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.xaxis.set_minor_locator(MultipleLocator(1000))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
        else:            
            if yaxis_limit is not None:
                ax.set_ylim(yaxis_limit)
            if not(hide_labels):
                ax.set_ylabel('Cross entropy loss', fontsize=font['size'])
            ax.tick_params(axis="y", direction='in')
            ax.tick_params(which="minor", direction='in')
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(2))
        
        # xaxis
        ax.set_xlim([0, max_axis])
        if shade_lim is not None:
            ax.fill_between(np.arange(max_axis), shade_lim[0], shade_lim[1], alpha=0.1, color='black', lw=0) 
        if legend_on:
            ax.legend(frameon=True, fontsize=font['size'])
        if not(hide_labels):
            ax.set_xlabel("Backpropagation iterations", fontsize=font['size'])            
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        # ax.xaxis.set_major_locator(MultipleLocator(2000))
        # ax.set_xticks([x for x in np.arange(0, max_axis, 2000)])
        ax.set_xticks([x for x in np.arange(0, max_axis, 1000)])
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.set_xticks([x for x in np.arange(0, max_axis, 4000)])
        xlabels = [f'{x:.0f}k' for x in ax.get_xticks()/1000]
        xlabels[0] = '0'
        ax.set_xticklabels(xlabels)
        return ax
        
def plot_calibration(ax=None,
                     saved_calibration=np.array([0.]),
                     experiment_names="all",
                     perplexity=False,
                     log_scale=True,
                     labels=[],
                     max_iter=100,
                     yaxis_limit=None,
                     colors=["blue"],
                     legend_on=False,
                     ):
    
    for exp_id, (exp, label, color) in enumerate(zip(data_dict_calibration, labels, colors)):
        if exp["name"] in experiment_names:
            saved_calibration = exp['loss_tensor']
            num_iters = min(max_iter, len(saved_calibration))
            iters = np.arange(0, num_iters)
            if perplexity:
                saved_calibration = np.exp(np.array(saved_calibration))
            ax.plot(iters, saved_calibration[:num_iters], '-o', linewidth=linewidth, ms=markersize, label=label, color=color) #, c='darkgreen'
            
    ax.set_xlabel("Adaptation iterations", fontsize=font['size'])
    if perplexity:
        ax.set_ylabel("Perplexity", fontsize=font['size'])
    else:
        ax.set_ylabel("Cross entropy loss", fontsize=font['size'])
        ax.tick_params(axis="y", direction='in')
        ax.tick_params(which="minor", direction='in')
        # ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        # ax.yaxis.set_major_locator(MultipleLocator(2))
        
    if yaxis_limit is not None:
        ax.set_ylim(yaxis_limit)
    
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(which="minor", direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(8))
    if log_scale:
        ax.set_yscale("log")    
        # formatter = LogFormatter(base=100, labelOnlyBase=False)
        # ax.get_yaxis().set_major_formatter(formatter) 
    
    if legend_on:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])  # Reverse the order
    return ax


linewidth = 2
markersize = 2
centimeters = 1 / 2.54  # centimeters in inches  
# width = 14 * centimeters
# height = 5 * centimeters
width = 12 * centimeters
height = 4 * centimeters  

legend_on = True

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))

# ax0 = inset_axes(ax1, width=width/4 * 0.5, height=height/4)
# ax0 = return_ax_NL_fig(ax0)
# ax0.set_xticks([0, 0.45, 0.9])                 
# xlabels = [f'{x:.1f}' for x in ax0.get_xticks()]
## ax0.set_xticklabels(xlabels)

# # Arxiv submission data
# saved_calibration = np.array([ 8.6550, 13.8243,  5.1499,  3.8073,  3.1518,  3.1054,  3.1018,  3.1047,
#          3.0987,  3.0997,  3.1004,  3.1003,  3.1003])
# NCS 4_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention_no_fine-tuning_bis.pt with decay factor improved -> LinearDRAMAttention trained on 13000 iterations
data_filename_calibration = './tests_divers/calibration_dram_scaling_parameters_results_multi_functions.json'
with open(data_filename_calibration, 'r') as f:
    data_dict_calibration = json.load(f)
data_dict_calibration = data_dict_calibration[1:]

attentions = [
              "DRAMAttention",
            #   "NLAttention_x3",
            #   "NLAttention_x5",
            #   "NLAttention_sigmoid",
            #   "NLAttention_exponential",
              ]

experiment_names = []
for attention in attentions:
    experiment_names += [f"{attention} no quant no decay adapted from gpt2-LinearDRAMAttention_tbs_1920_decay_factor_0.00016_stop13000_with_head_scaling.pt"]

labels = ["Nonlinear model",
          r'$x^3$',
          r'$x^5$',
          r"$Sigmoid$",
          r"$Exponential$",
          ]

colors = ["darkgreen",
          "",
          "goldenrod",
          "lightslategray",
          "maroon",
          ]

data_dict_calibration.reverse()
labels.reverse()
colors.reverse()

perplexity=True
ax1 = plot_calibration(ax=ax1,
                    saved_calibration=data_dict_calibration,
                    experiment_names=experiment_names,
                    perplexity=perplexity,
                    log_scale=True,
                    labels=labels,
                    # yaxis_limit=[19, 10e+4],
                    colors=colors,
                    legend_on=legend_on,
                    )

run_list_1 = [ 
            {"label": "Software model trained from scratch",
             "color": "black",
             "name": "gpt2-from-scratch_stop13000_bs_20_grad_ac_96",
              },    
                         
            {"label": "Nonlinear model trained from scratch",
             "color": "darkmagenta",
             "name": "DRAM_surrogate_stop_13000_ft_from_scratch",
             },
            
            # {"label": "Nonlinear model\nfine-tuned from GPT-2",
            #  "color": "brown",
            #  "name": "DRAM_surrogate_stop_13000_ft_from_gpt2",
            #  },         
            
            {"label": "Linear model fine-tuned from GPT-2",
             "color": "darkblue",
             "name": "gpt2-LinearDRAMAttention_tbs_1920_decay_factor_0.00016_stop13000",
             },
            
            {"label": "Nonlinear model fine-tuned from linear model",
             "color": "darkgreen",
             "name": "DRAM_surrogate_stop_13000_ft_from_gpt2-LinearDRAMAttention_tbs_1920_decay_factor_0.00016_stop3000",
             },  
            ]

data_filename = './utils/owt_training_results_ncs.json'
with open(data_filename, 'r') as f:
    data_dict = json.load(f)

perplexity=True # if False, Cross entropy loss
max_axis = 13001
moving_average_window = 5
yaxis_limit_inset = [20, 25]

ax2 = plot_metrics_multi_run(
            ax=ax2,
            data_dict=data_dict,
            max_axis=max_axis,
            metric='val/loss',
            runs=run_list_1,
            perplexity=perplexity,
            log_scale=False,
            yaxis_limit=[19, 40],
            moving_average_window=moving_average_window,
            shade_lim=yaxis_limit_inset,
            legend_on=legend_on,  
            )

# axins = inset_axes(ax2, width=width/4 * 2/3, height=height/4)
axins = inset_axes(ax2, width=width/4 * 0.5, height=height/4 * 1.1)

axins = plot_metrics_multi_run(
            ax=axins,
            data_dict=data_dict,
            max_axis=max_axis,
            metric='val/loss',
            runs=run_list_1,
            perplexity=perplexity,
            log_scale=False,
            yaxis_limit=yaxis_limit_inset,
            moving_average_window=moving_average_window,
            hide_labels=True,    
            shade_lim=yaxis_limit_inset,   
            legend_on=False,     
            )

# plot baseline
baseline = 3.10 # trained ChatGPT2
if perplexity:
    baseline = np.exp(baseline)    
for ax_ in [ax2]:
    ax_.plot(np.arange(max_axis), np.ones((max_axis))*baseline, '--', c="black", linewidth=linewidth, label="Baseline: public software GPT-2")
    # ax.fill_between(np.arange(max_axis), baseline-1.5, baseline+1.5, alpha=0.5, color='black') # 1.5 is the perplexity std
    if legend_on:
        ax_.legend()

fig.tight_layout()
# for ax_single in [ax0, ax1, ax2]:
for ax_single in [ax1, ax2]:
    ax_single.xaxis.labelpad = 5
    ax_single.yaxis.labelpad = 5
 
file_out = './plots/combined_dram_training_results'
if legend_on:
    file_out += "_labels"
for fmt in ['png', 'svg', 'pdf']:
    plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)

plt.show()