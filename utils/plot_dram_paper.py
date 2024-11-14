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
width = 4.5 * centimeters # 5.5
height = 4 * centimeters  

def import_run(run_id):
    api = wandb.Api(overrides={
                           "project": project,       
                           "entity": entity,
                            })
    run = api.run(entity+'/'+project+'/'+run_id)
    return run

def plot_metrics_multi_run(apply=False,
                           save_plot=False,
                           metrics="MAE (degrees) test",
                           end=-1,
                           max_axis=1000,
                           ylabel="Mean Absolute Error (°)",
                           runs=[],
                           file_out="",
                           from_wandb=True,
                           ):
    if apply:
        
        fig, ax = plt.subplots(1)
        fig.set_figwidth(width)
        fig.set_figheight(height)
        
        # plot baseline
        baseline_cel = 3.00 # trained ChatGPT2        
        ax.plot(np.arange(max_axis), np.ones((max_axis))*baseline_cel, '--', c="black", linewidth=linewidth, label="Baseline: pre-trained software gpt2")
        # baseline_ppl = np.exp(baseline_cel)
        # ax.plot(np.arange(max_axis), np.ones((max_axis))*baseline_ppl, '--', c="black", linewidth=linewidth, label="Baseline")
        
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
                
            ax.plot(iter[r, :i], metric[r, :i], '-', c=runs[r]['color'], linewidth=linewidth, label=runs[r]['label'])
            
            if from_wandb:
                data_dict.update({runs[r]['label']: {'iterations': iter[r, :i].tolist(), ylabel: metric[r, :i].tolist()}})
                max_iter = max(max_iter, idx[1]['iter'])
            else:            
                max_iter = max(iter[r, :-1])

            # ax.set_xlim([0,21])
            # ax.legend(frameon=False, fontsize=font['size'])
            ax.set_xlabel("Iterations", fontsize=font['size'])
            ax.set_ylabel(ylabel, fontsize=font['size'])
            # ax.set_ylim([3, 4])
            # ax.set_yscale("log")
            # ax.set_xscale("log")
            # ax.set_title(''.join(["end value: "]+[f"{metric_end[r]:.4f}, " for r in range(len(run_ids))]))
            
            ax.tick_params(axis="x", direction='in')
            ax.tick_params(which="minor", direction='in')
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_major_locator(MultipleLocator(2000))
            ax.set_xticks([x for x in np.arange(0, min(max_iter, max_axis), 2000)])
            xlabels = [f'{x:.0f}k' for x in ax.get_xticks()/1000]
            xlabels[0] = '0'
            ax.set_xticklabels(xlabels)
            
            ax.tick_params(axis="y", direction='in')
            ax.tick_params(which="minor", direction='in')
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(2))

            fig.tight_layout()
            ax.xaxis.labelpad = 5
            ax.yaxis.labelpad = 5
            
            if r == 3:
                if save_plot:    
                    for fmt in ['png', 'svg', 'pdf']:
                        plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)  
                file_out += "2nd_part"
                fig, ax = plt.subplots(1)
                fig.set_figwidth(width)
                fig.set_figheight(height)                
                # plot baseline
                baseline_cel = 3.00 # trained ChatGPT2        
                ax.plot(np.arange(max_axis), np.ones((max_axis))*baseline_cel, '--', c="black", linewidth=linewidth, label="Baseline")
                
        with open(data_filename, 'w') as f:
            json.dump(data_dict, f, indent=4)
                
        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)  
        plt.show()
        
def plot_calibration(save_plot=False, saved_calibration=np.array([0.]), end=100, file_out=None):
    fig, ax = plt.subplots(1)          
    fig.set_figwidth(width * centimeters)
    fig.set_figheight(height * centimeters)
    max_iter = min(end, len(saved_calibration))
    iters = np.arange(0, max_iter)
    # saved_calibration = np.exp(saved_calibration)
    ax.plot(iters, saved_calibration[:max_iter], '-o', c='darkblue', linewidth=linewidth, ms=markersize)
    ax.set_xlabel("Iterations", fontsize=font['size'])
    ax.set_ylabel("Cross entropy loss", fontsize=font['size'])
    # ax.set_ylabel("Perplexity", fontsize=font['size'])
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(which="minor", direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis="y", direction='in')
    ax.tick_params(which="minor", direction='in')
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    # ax.set_yscale("log")

    fig.tight_layout()
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5
    if save_plot:    
        for fmt in ['png', 'svg', 'pdf']:
            plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
    plt.show()

run_list = [
            
            # {
            # "label": "GPT-2 trained from scratch",
            #  "id": "nleroux/owt/nleroux/owt/tf6qb0s4",
            #  "color": "darkmagenta",
            #  "name": "gpt2-from-scratch",
            #  },
                
            {"label": "Hardware model trained from scratch",
             "id": "nleroux/owt/nleroux/owt/tijpdfhw",
             "color": "black",
             "name": "DRAM_ft_from_scratch_fixed_att_and_output_threshold_no_calib",
             },
            
            {"label": "Hardware model trained from scratch (2)",
             "id": "nleroux/owt/kgzz0vr2",
             "color": "black",
             "name": "DRAM_ft_from_scratch_fixed_att_and_output_threshold_no_calib-2nd_part",
             },
            
            {"label": "Fine-tuned from GPT-2",
             "id": "nleroux/owt/0czn51nf",
             "color": "brown",
             "name": "DRAM_ft_from_gpt2_fixed_att_and_output_threshold_no_calib",
             },
            
            {"label": "Fine-tuned from GPT-2 (2)",
             "id": "nleroux/owt/d782o1vw",
             "color": "brown",
             "name": "DRAM_ft_from_gpt2_fixed_att_and_output_threshold_no_calib-2nd_part",
             },
            
            # {"label": "Quantized GPT-2",
            #  "id": "nleroux/owt/d2bylvrx",
            #  "color": "grey",
            #  "name": "gpt2_quantized_32bits_output",
            #  },
            
            {"label": "Intermediate model fine-tuned from GPT-2",
             "id": "nleroux/owt/jn1qm4g9",
             "color": "darkblue",
             "name": "LinearDRAMAttention_tilling_output_pulse_fixed_threhshold_atten_80µA_wa_40µA_3000iters_32bits_out",
             },
            
            {"label": "Intermediate model fine-tuned from GPT-2 (2)",
             "id": "nleroux/owt/1oy54bt9",
             "color": "darkblue",
             "name": "LinearDRAMAttention_tilling_output_pulse_fixed_threhshold_atten_80µA_wa_40µA_3000iters_32bits_out",
             },
            
            {"label": "Fine-tuned from intermediate model",
             "id": "nleroux/owt/nh41m8mj",
             "color": "darkgreen",
             "name": "DRAM_ft_from_LinearDRAMAttention_tilling_output_pulse_fixed_threhshold_atten_80µA_wa_40µA_2000iters_32bits_out",
             },      
            
            {"label": "Fine-tuned from intermediate model (2)",
             "id": "nleroux/owt/v5l6dacw",
             "color": "darkgreen",
             "name": "DRAM_ft_from_LinearDRAMAttention_tilling_output_pulse_fixed_threhshold_atten_80µA_wa_40µA_2000iters_32bits_out",
             },           
            ]

save_plot = True
for met, ylabel in zip(metrics_list, ylabel_list):
    plot_metrics_multi_run(apply=True, 
                save_plot=save_plot,
                metrics=met,
                end=100,
                max_axis=10001,
                ylabel=ylabel,
                runs=run_list,
                file_out=out_file_root+"training_dram",
                from_wandb=False,
                )
    

saved_calibration = np.array([ 8.6550, 13.8243,  5.1499,  3.8073,  3.1518,  3.1054,  3.1018,  3.1047,
         3.0987,  3.0997,  3.1004,  3.1003,  3.1003])
plot_calibration(save_plot=save_plot,
                 saved_calibration=saved_calibration,
                 end=100,
                 file_out=out_file_root+"figure_calibration")