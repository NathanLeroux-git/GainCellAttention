import wandb
import numpy as np

entity = "nleroux"
project = "sEMG_DOA_regression_start_05_01_23"

def get_runs(group="", subjects_list=None, svs_list=None):
    api = wandb.Api(overrides={
                "project": project,       
                "entity": entity,
                    })
    subjects_list = [{"config.subjects": int(sub)} for sub in subjects_list] if subjects_list is not None else {'group': group}
    svs_condition = [{"config.stored_vector_size": int(svs)} for svs in svs_list] if svs_list is not None else {'group': group}
    runs = api.runs(path=entity+'/'+project,
                    filters={"$and": [
                                    {'group': group},
                                    {"$or": subjects_list
                                    },
                                    {"$or": svs_condition
                                    }
                                    ]
                            }
                    )    
    return runs

subject_list = [0,1,2]
svs_list = [150]
group = 'binary embedding + LIF qkv and MLP ws=2000 ks=7 s=5 svs=150'
runs = get_runs(group=group, subjects_list=subject_list, svs_list=svs_list)

# Summary method: only access information from the end of the run
mae = []
for run in runs:
    mae += [run.summary['MAE (degrees) test']]
    
# History method: access all history
mae = np.zeros((3, 11))
for r, run in enumerate(runs):
    for i, run_epoch in enumerate(run.history(keys=['MAE (degrees) test']).iterrows()):
        mae[r, i] = run_epoch[1][1]

pass