import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams['figure.figsize'] =(1.8,1.5) 
plt.rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#023d6bff', 'g', 'r', 'c', 'm', 'y', 'k'])
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = 'k'
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

f = open("./tests_divers/nvidia_4090_attention.json", 'r')
data = json.load(f)
mean_energy_per_token = data['mean_energy_per_token']
mean_latency_per_token = data['mean_latency_per_token']

energy_GPU = mean_energy_per_token
latency_GPU = mean_latency_per_token
# frequency_GPU = 1/latency_GPU

f = open("./tests_divers/nvidia_jetson_attention.json", 'r')
data = json.load(f)
mean_energy_per_token = data['mean_energy_per_token']
mean_latency_per_token = data['mean_latency_per_token']

energy_jetson = mean_energy_per_token
latency_jetson = mean_latency_per_token
# frequency_jetson = 1/latency_jetson

#enter the numbers here:

energy_IMC_comp = np.array([1120e-12,700e-12,4e-9,330e-12])
energy_IMC_comp_str = ['$\Phi(Q \cdot K^T)$',"$\Phi(S) \cdot V$", "Digital & routing", "DAC"]

energy_IMC = np.sum(energy_IMC_comp)
energy_IMC *= 12 # Because we want to compare to executing 12 attention heads on GPUs


# bar plot of energy consumption with log scale
# fig, ax = plt.subplots(figsize=(1.8, 2.1))
centimeters = 1 / 2.54  # centimeters in inches  
width = 10 * centimeters # 5.5
height = 5.5 * centimeters
# height = 2.1
fig, ax = plt.subplots(1, 2, figsize=(width, height))

bar_width = 0.7
bar1 = ax[1].bar(1, energy_GPU, bar_width, label='Nvidia RTX 4090', color="#023d6bff")
bar2 = ax[1].bar(2, energy_jetson, bar_width, label='Nvidia Jetson Nano', color="#0000ffff")
bar3 = ax[1].bar(3, energy_IMC, bar_width, label='This work', color="#aa0088ff")
ax[1].set_yscale('log')
ax[1].set_xticks([1, 2, 3])
ax[1].set_xticklabels([bar1.get_label(), bar2.get_label(), bar3.get_label()], rotation=45)
ax[1].set_ylabel('Energy (J)')

fig.tight_layout()
# for fmt in ['png', 'svg', 'pdf']:
#     plt.savefig('./plots/gpu_versus_dram_energy' + '.%s' % fmt, format=fmt, dpi=1200)  
# plt.show()

#enter the numbers here:
latency_IMC = 65e-9
# frequency_IMC = 1/latency_IMC

# bar plot of latency with log scale
# fig, ax = plt.subplots(figsize=(1.8, 2.1))
bar1 = ax[0].bar(1, latency_GPU, bar_width, label='Nvidia RTX 4090', color="#023d6bff")
bar2 = ax[0].bar(2, latency_jetson, bar_width, label='Nvidia Jetson Nano', color="#0000ffff")
bar3 = ax[0].bar(3, latency_IMC, bar_width, label='This work', color="#aa0088ff")
ax[0].set_yscale('log')
ax[0].set_xticks([1, 2, 3])
ax[0].set_xticklabels([bar1.get_label(), bar2.get_label(), bar3.get_label()], rotation=45)
ax[0].set_ylabel('Latency (s)')
#tild the x-axis labels
fig.tight_layout()



for fmt in ['png', 'svg', 'pdf']:
    plt.savefig('./plots/gpu_versus_dram_energy_latency' + '.%s' % fmt, format=fmt, dpi=1200)  
plt.show()


#pie chart showing the componetns of IMC 
fig, ax = plt.subplots()
ax.pie(energy_IMC_comp, labels=energy_IMC_comp_str, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

speedup_jestson = latency_jetson/latency_IMC
speedup_GPU = latency_GPU/latency_IMC
energy_improvement_jetson = energy_jetson/energy_IMC
energy_improvement_GPU = energy_GPU/energy_IMC

print(f"IMC improvements:\nspeedup_jestson: {speedup_jestson:.0f}\nspeedup_GPU: {speedup_GPU:.0f}\nenergy_improvement_jetson: {energy_improvement_jetson:.0f}\nenergy_improvement_GPU: {energy_improvement_GPU:.0f}")