import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator, FormatStrFormatter
import numpy as np
import wandb
import json

data_filename = './utils/owt_training_results.json'

entity = "nleroux"
project = "owt"
out_file_root = "./plots/"

metrics_list = ["val/loss"]
ylabel_list = ["Cross entropy loss"]
# ylabel_list = ["Perplexity"]

font = {'size': 8}
rc('font', **font)
# change font
# rcParams['font.sans-serif'] = "Arial"
# rcParams['font.family'] = "sans-serif"
# rcParams['text.usetex'] = True
rcParams['mathtext.default'] = 'regular'  # Math subscripts and greek letters non-italic
linewidth = 1.5
markersize = 3
centimeters = 1 / 2.54  # centimeters in inches  
width = 15 * centimeters # 5.5
height = 4 * centimeters  

fig, ax = plt.subplots(1, 3, figsize=(width, height))

def import_run(run_id):
    api = wandb.Api(overrides={
                           "project": project,       
                           "entity": entity,
                            })
    run = api.run(entity+'/'+project+'/'+run_id)
    return run

def plot_metrics_multi_run(apply=False,
                           ax=None,
                           save_plot=False,
                           metrics="MAE (degrees) test",
                           end=-1,
                           max_axis=1000,
                           ylabel="Mean Absolute Error (°)",
                           runs=[],
                           file_out="",
                           from_wandb=False,
                           ):
    if apply:
        ax_idx = 1
        # plot baseline
        baseline_cel = 3.00 # trained ChatGPT2        
        ax[ax_idx].plot(np.arange(max_axis), np.ones((max_axis))*baseline_cel, '--', c="black", linewidth=linewidth, label="Baseline: pre-trained\nsoftware GPT-2")
        # baseline_ppl = np.exp(baseline_cel)
        # ax[ax_idx].plot(np.arange(max_axis), np.ones((max_axis))*baseline_ppl, '--', c="black", linewidth=linewidth, label="Baseline")
        
        # steps = np.arange(0, end)
        # iter = steps * 100
        max_iter = 0
        
        metric = np.zeros((len(runs), end))
        iter = np.zeros((len(runs), end))
        
        if from_wandb:
            data_dict = {}
        else:
            with open(data_filename, 'r') as f:
                data_dict = json.load(f)
             
        for r, run in enumerate(runs):
            if from_wandb:
                run_history = import_run(run['id']).history
                iter_iterrows, metric_iterrows = run_history(keys=['iter']).iterrows(), run_history(keys=[metrics]).iterrows()
                for i, (idx, data) in enumerate(zip(iter_iterrows, metric_iterrows)):
                    if i>=end:
                        break
                    print(f"iter {i}\t{metrics} = {data[1][metrics]:.4f}")
                    iter[r, i] = idx[1]['iter']
                    metric[r, i] = data[1][metrics]
                    # metric[r, i] = np.exp(data[1][metrics])
                    if idx[1]['iter'] >= max_axis:    
                        break
            else:
                data = data_dict[runs[r]['label']]
                iter[r, :len(data['iterations'])] = data['iterations']
                metric[r, :len(data[ylabel])] = data[ylabel]
                i = len(data['iterations'])
                
            ax[ax_idx].plot(iter[r, :i], metric[r, :i], '-', c=runs[r]['color'], linewidth=linewidth, label=runs[r]['label'])
            
            if from_wandb:
                data_dict.update({runs[r]['label']: {'iterations': iter[r, :i].tolist(), ylabel: metric[r, :i].tolist()}})
                max_iter = max(max_iter, idx[1]['iter'])
            else:            
                max_iter = max(iter[r, :-1])

            # ax[ax_idx].set_xlim([0,21])
            # ax[ax_idx].legend(frameon=True, fontsize=font['size'])
            ax[ax_idx].set_xlabel("Backpropagation iterations", fontsize=font['size'])
            ax[ax_idx].set_ylabel(ylabel, fontsize=font['size'])
            # ax[ax_idx].set_ylim([3, 4])
            # ax[ax_idx].set_yscale("log")
            # ax[ax_idx].set_xscale("log")
            # ax[ax_idx].set_title(''.join(["end value: "]+[f"{metric_end[r]:.4f}, " for r in range(len(run_ids))]))
            
            ax[ax_idx].tick_params(axis="x", direction='in')
            ax[ax_idx].tick_params(which="minor", direction='in')
            ax[ax_idx].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[ax_idx].xaxis.set_major_locator(MultipleLocator(2000))
            ax[ax_idx].set_xticks([x for x in np.arange(0, min(max_iter, max_axis), 2000)])
            xlabels = [f'{x:.0f}k' for x in ax[ax_idx].get_xticks()/1000]
            xlabels[0] = '0'
            ax[ax_idx].set_xticklabels(xlabels)
            
            ax[ax_idx].tick_params(axis="y", direction='in')
            ax[ax_idx].tick_params(which="minor", direction='in')
            ax[ax_idx].yaxis.set_minor_locator(AutoMinorLocator(5))
            ax[ax_idx].yaxis.set_major_locator(MultipleLocator(2))

            fig.tight_layout()
            ax[ax_idx].xaxis.labelpad = 5
            ax[ax_idx].yaxis.labelpad = 5
            
            if r == 3:
                ax_idx += 1               
                # plot baseline
                baseline_cel = 3.00 # trained ChatGPT2        
                ax[ax_idx].plot(np.arange(max_axis), np.ones((max_axis))*baseline_cel, '--', c="black", linewidth=linewidth, label="Baseline: pre-trained\nsoftware GPT-2")
                
        with open(data_filename, 'w') as f:
            json.dump(data_dict, f, indent=4)

        return ax
        
def plot_calibration(ax=None, save_plot=False, saved_calibration=np.array([0.]), end=100, file_out=None):
    ax_idx = 0
    max_iter = min(end, len(saved_calibration))
    iters = np.arange(0, max_iter)
    # saved_calibration = np.exp(saved_calibration)
    ax[ax_idx].plot(iters, saved_calibration[:max_iter], '-o', c='darkgreen', linewidth=linewidth, ms=markersize, label="Nonlinear model")
    ax[ax_idx].set_xlabel("Adaptation iterations", fontsize=font['size'])
    ax[ax_idx].set_ylabel("Cross entropy loss", fontsize=font['size'])
    # ax[ax_idx].set_ylabel("Perplexity", fontsize=font['size'])
    ax[ax_idx].tick_params(axis="x", direction='in')
    ax[ax_idx].tick_params(which="minor", direction='in')
    ax[ax_idx].xaxis.set_minor_locator(AutoMinorLocator(1))
    ax[ax_idx].xaxis.set_major_locator(MultipleLocator(2))
    ax[ax_idx].tick_params(axis="y", direction='in')
    ax[ax_idx].tick_params(which="minor", direction='in')
    ax[ax_idx].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[ax_idx].yaxis.set_major_locator(MultipleLocator(2))
    # ax[ax_idx].set_yscale("log")
    
    return ax

run_list = [
            
            # {
            # "label": "GPT-2 trained from scratch",
            #  "id": "nleroux/owt/nleroux/owt/tf6qb0s4",
            #  "color": "darkmagenta",
            #  "name": "gpt2-from-scratch",
            #  },
                
            {"label": "Nonlinear model trained\nfrom scratch",
             "id": "nleroux/owt/nleroux/owt/tijpdfhw",
             "color": "black",
             "name": "DRAM_ft_from_scratch_fixed_att_and_output_threshold_no_calib",
             },
            
            {"label": None,
             "id": "nleroux/owt/kgzz0vr2",
             "color": "black",
             "name": "DRAM_ft_from_scratch_fixed_att_and_output_threshold_no_calib-2nd_part",
             },
            
            {"label": "Nonlinear model\nfine-tuned from GPT-2",
             "id": "nleroux/owt/0czn51nf",
             "color": "brown",
             "name": "DRAM_ft_from_gpt2_fixed_att_and_output_threshold_no_calib",
             },
            
            {"label": None,
             "id": "nleroux/owt/d782o1vw",
             "color": "brown",
             "name": "DRAM_ft_from_gpt2_fixed_att_and_output_threshold_no_calib-2nd_part",
             },
            
            # {"label": "Quantized GPT-2",
            #  "id": "nleroux/owt/d2bylvrx",
            #  "color": "grey",
            #  "name": "gpt2_quantized_32bits_output",
            #  },
            
            {"label": "Linear model\nfine-tuned from GPT-2",
             "id": "nleroux/owt/jn1qm4g9",
             "color": "darkblue",
             "name": "LinearDRAMAttention_tilling_output_pulse_fixed_threhshold_atten_80µA_wa_40µA_3000iters_32bits_out",
             },
            
            {"label": None,
             "id": "nleroux/owt/1oy54bt9",
             "color": "darkblue",
             "name": "LinearDRAMAttention_tilling_output_pulse_fixed_threhshold_atten_80µA_wa_40µA_3000iters_32bits_out",
             },
            
            {"label": "Nonlinear model fine-tuned\nfrom linear model",
             "id": "nleroux/owt/nh41m8mj",
             "color": "darkgreen",
             "name": "DRAM_ft_from_LinearDRAMAttention_tilling_output_pulse_fixed_threhshold_atten_80µA_wa_40µA_2000iters_32bits_out",
             },      
            
            {"label": None,
             "id": "nleroux/owt/v5l6dacw",
             "color": "darkgreen",
             "name": "DRAM_ft_from_LinearDRAMAttention_tilling_output_pulse_fixed_threhshold_atten_80µA_wa_40µA_2000iters_32bits_out",
             },           
            ]

save_plot = True
for met, ylabel in zip(metrics_list, ylabel_list):
    ax = plot_metrics_multi_run(apply=True, 
                ax=ax,
                save_plot=save_plot,
                metrics=met,
                end=100,
                max_axis=10001,
                ylabel=ylabel,
                runs=run_list,
                file_out=out_file_root+"training_dram",
                from_wandb=True,
                )
    

saved_calibration = np.array([ 8.6550, 13.8243,  5.1499,  3.8073,  3.1518,  3.1054,  3.1018,  3.1047,
         3.0987,  3.0997,  3.1004,  3.1003,  3.1003])

ax = plot_calibration(ax=ax,
                 save_plot=save_plot,
                 saved_calibration=saved_calibration,
                 end=100,
                 file_out=out_file_root+"figure_calibration")

fig.tight_layout()
for ax_single in ax:
    ax_single.xaxis.labelpad = 5
    ax_single.yaxis.labelpad = 5
    # ax_single.legend(frameon=True, fontsize=font['size'])
 
file_out = './plots/combined_dram_training_results'
for fmt in ['png', 'svg', 'pdf']:
    plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)  

plt.show()