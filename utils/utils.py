import wandb
import requests
import os
import torch
import pandas as pd
import sys
from collections import OrderedDict
import torch.distributed as dist
    
def get_params_grad(model):
    # names = [name for name, _ in model.state_dict().items()]
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name, 'gradient is None', 'requires_grad:', param.requires_grad)
        elif param.grad.sum()==0:
            print(name, 'gradient is Zero', 'requires_grad:', param.requires_grad)
        # else:
        #     print(name, param.grad.sum(), 'requires_grad:', param.requires_grad)
    print('\n')

def import_run(run_id, entity='', project=''):
    api = wandb.Api(overrides={
                           "project": project,       
                           "entity": entity,
                            })
    run = api.run(entity+'/'+project+'/'+run_id)
    return run

def del_previous_plots(epoch, run, status):
    if status == 'online':
        if epoch>1:        
            api = wandb.Api(overrides={
                            "project": run.project,       
                            "entity": run.entity,
                                })
            run_adress = f'{run.entity}/{run.project}/{run.id}'
            run_api = api.run(run_adress)        
            list_plot_media = []
            for i, file in enumerate(run_api.files()):
                if file.name[0:12]=='media/plotly':             
                    list_plot_media += [[i, file.name, file.updatedAt]]
                    
            list_plot_media.sort(key=lambda x: x[2], reverse=False)
            if len(list_plot_media)>2:
                run_api.files()[list_plot_media[0][0]].delete()   
                run_api.files()[list_plot_media[1][0]].delete() 
                
def load_model_from_file(args, target_model, saved_model_name, directory='../saved_models'):
    model_to_load = torch.load(directory+'/'+saved_model_name+'.pt',
                            #    map_location=f'cuda:{args.devices[0]}',
                               map_location='cpu',
                               ) if args.pre_trained_model_name is not None else None
    if 'model' in model_to_load:
        model_to_load = model_to_load['model']
    # state of LIF neurons must be removed of dictonary because we don't want to transfer them
    # for key in model_to_load:
    #     if key[-7:]=="k_cache" or key[-7:]=="v_cache" or key[-7:]=="state.S" or key[-7:]=="state.U" or key[-7:]=="state.I" or key[-8:]=="state.Ir":
    #         del model_to_load[key]    
    def condition(key):
        return key[-7:]!="k_cache" and key[-7:]!="v_cache" and key[-7:]!="state.S" and key[-7:]!="state.U" and key[-7:]!="state.I" and key[-8:]!="state.Ir"
    corrected_saved_model = OrderedDict((key, value) for key, value in model_to_load.items() if condition(key))
    if len(args.devices) < 2:
        target_model.load_state_dict(corrected_saved_model, strict=False)
    else:
        target_model.module.load_state_dict(corrected_saved_model, strict=False)
    return target_model
        
def transfer_weights(train_model, test_model):
    train_model_state_dict = train_model.state_dict()
    # state of LIF neurons must be removed of dictonary because we don't want to transfer them
    for key in train_model.state_dict():
        if key[-7:]=="k_cache" or key[-7:]=="v_cache" or key[-7:]=="state.S" or key[-7:]=="state.U" or key[-7:]=="state.I" or key[-8:]=="state.Ir":
            del train_model_state_dict[key]
    test_model.load_state_dict(train_model_state_dict, strict=False)   

def fix_LIF_states(model_to_load):
    if model_to_load is not None:
        for k, v in model_to_load.items():
            if k[-7:]=='state.U' or k[-7:]=='state.S' or k[-7:]=='state.I' or k[-8:]=='state.Ir':
                LIF_neurons_num = v.shape[-1]
                model_to_load[k] = v[0,:].reshape((1, LIF_neurons_num))
    return model_to_load

def download_dataset(name):
    url = 'http://ninapro.hevs.ch/system/files/DB8/'+name
    print(f'Downloading request on {url}, can take several minutes...')
    r = requests.get(url, allow_redirects=True)
    folder_name = '../datasets/ninapro8_dataset'
    os.makedirs(folder_name, exist_ok=True)
    open(folder_name+'/'+name, 'wb').write(r.content)
    
def add_dictonary_of_metrics(dict, dict_sum):
    if dict_sum is None:
        dict_sum = dict
    else:
        for (_, v1), (_, v2) in zip(dict.items(), dict_sum.items()):
            v2 += v1   
    return dict_sum

def mean_dictonary_of_metrics(num_elements, dict_sum):
    for _, v in dict_sum.items():
        v /= num_elements
    return dict_sum

def make_summary_csv(run, path):
    DataFrame = {}
    for k, v in run.summary.items():
        if not(k.startswith('_wandb')) and not(k.startswith('parameters')) and not(k.startswith('Train results')) and not(k.startswith('Test results')):            
            DataFrame.update({k: v})    
    DataFrame = pd.DataFrame(DataFrame, index=[0])
    DataFrame.to_csv(path + '_summary.csv')   
    
def make_config_csv(run, path):
    DataFrame = {}
    for k, v in run.config.items():
        DataFrame.update({k: str(v)}) 
    DataFrame = pd.DataFrame(DataFrame, index=[0])
    DataFrame.to_csv(path + '_config.csv') 
    
def get_wandb_run_local(path):
    summary = pd.read_csv(path + '_summary.csv')
    config = pd.read_csv(path + '_config.csv')
    return summary, config
    
def save_wand_run_local(group, sub, svs):
    entity = "user"
    project = "sEMG_DOA_regression_start_05_01_23"  
    api = wandb.Api(overrides={
        "project": project,       
        "entity": entity,
            })
    run = api.runs(path=entity+'/'+project,
                    filters={"$and": [
                                    {'group': group},
                                    {"config.subjects": int(sub)},
                                    {"config.stored_vector_size": int(svs)},
                                    ]
                            }
                    ) 
    run = run[0]
    group = '../results/' + group
    os.makedirs(group, exist_ok=True)  
    name = f'subject{sub:02d}svs{svs:03d}'
    path = group + '/' + name
    if not(os.path.isfile(path + '_summary.csv')):
        make_summary_csv(run, path)
    if not(os.path.isfile(path + '_config.csv')):
        make_config_csv(run, path)
    return path

def change_group_name(group_name, new_group_name):
    entity = "user"
    project = "dRAM_CAM"  
    api = wandb.Api(overrides={
                    "project": project,       
                    "entity": entity,
                        })
    runs = api.runs(path=entity+'/'+project,
                filters={"$and": [
                                {'group': group_name},
                                ]
                        }
                ) 
    for run in runs:
        run.group = new_group_name
        run.update()
    
# change_group_name('AttentionCAMOnly_CapaKVDecay_2e-2_mlo=100_svs=64_bs=64',
#                   'AttentionCAMOnly_CapaKVDecay_2e-2_m_lo=100_svs=64_bs=64')

def model_n_params(model):
    n_params = 0
    for params in model.parameters():    
        if params.requires_grad:
            n_params += params.numel()
    print(f'Num parameters: {n_params}')
    return n_params

def parse_command_line_to_dict(argv):
    kwargs = {}
    for i, arg in enumerate(argv):
        if i > 0:
            try:
                # For non-strings parameters
                exec(arg)
                key = arg[:(arg.find('='))]
                val = locals()[key]
                kwargs.update({key: val})
            except:
                # For strings parameters
                key = arg[:(arg.find('='))]
                val = arg[(arg.find('='))+1:]
                kwargs.update({key: val})
    return kwargs