import os, sys
import particle # Scikit HEP
import argparse
from utils import load_config
import torch_geometric as pyg
from torch_geometric.data import Data
import torch
from tqdm import tqdm # the best python package
from tqdm.contrib.concurrent import process_map
from functools import partial

CHARGE_DATAKEY = 'track_particle_charge'

def main():
    # Skip most of the datareader class from Acorn by adjusting the PyG data objects directly
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML file containing metadata about task')
    parser.add_argument('-w', '--workers', required=False, default=1, 
                        help='Number of workers that can be allocated to perform multiprocessing steps. Default is 1.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    config_filepath = args.config_file
    config = load_config(config_filepath if config_filepath != '' else 'config.yaml')
    
    assert 'input_dir' in config
    assert 'preprocess_output_dir' in config

    input_dir = config.get('input_dir')
    input_trainset_path = input_dir + os.sep + 'trainset'
    input_testset_path = input_dir + os.sep + 'testset'
    input_valset_path = input_dir + os.sep + 'valset'
    
    output_trainset_path = input_dir + os.sep + 'trainset'
    output_testset_path = input_dir + os.sep + 'testset'
    output_valset_path = input_dir + os.sep + 'valset'
    
    all_input_datasets = {"trainset": input_trainset_path, 
                    "testset": input_testset_path, 
                    "valset": input_valset_path}
    
    all_output_datasets = {"trainset": output_trainset_path, 
                    "testset": output_testset_path, 
                    "valset": output_valset_path}
    
        
    for dataset_name, dataset_dir in all_input_datasets.items():
        max_workers:int = int(args.workers)
        if max_workers != 1:
            print(f"INFO: Running process with {max_workers} workers")
            process_map(
                partial(
                    subroutine, 
                    input_dir=dataset_dir, 
                    output_dir=all_output_datasets[dataset_name],
                    config=config,
                    args=args
                ),
                (file for file in os.listdir(dataset_dir) if file.split('.')[-1] == 'pyg'),
                chunksize=1,
                max_workers=max_workers,
                desc=f"INFO: Processing {dataset_name} PyG Files"
            )
        else:
            print(f"INFO: Running process single-threaded")
            partial_subroutine = partial(
                subroutine, 
                input_dir=dataset_dir, 
                output_dir=all_output_datasets[dataset_name],
                config=config,
                args=args
            )
            for file in tqdm((file for file in os.listdir(dataset_dir) if file.split('.')[-1] == 'pyg'), 
                             desc=f"INFO: Processing {dataset_name} PyG Files"):
                if file.split('.')[-1] != 'pyg':
                    continue
                partial_subroutine(file)
                
            # Check that the extension of the file is '.pyg'

def subroutine(file, input_dir, output_dir, config, args):
    # Directly read PyG object (since most of the work is already done for us)
    assert file.split('.')[-1] == 'pyg', f'Not a pyg file, got "{file}"'
    data_object = torch.load(os.path.join(input_dir, file))
    if args.verbose:
        print(f"INFO: Processing event-{data_object.event_id}")
    if CHARGE_DATAKEY in data_object.keys():
        if args.verbose:
            print(f"INFO: Skipping event-{data_object.event_id}")
        return
    pdgids = data_object.get('track_particle_pdgId')
    if pdgids is None:
        raise KeyError(f"ERROR: PdgID not specified in PyG file (event-{data_object.event_id})")
    charges = [] # temporarily store in python object before converting to torch tensor
    for id in pdgids:
        charge = particle.pdgid.charge(particle.PDGID(id))
        charges.append(charge)
    charges_tensor = torch.Tensor(charges)
    data_object[CHARGE_DATAKEY] = charges_tensor
    save_pyg_data(data_object, os.path.join(config['preprocess_output_dir']), data_object.event_id)
            
def save_pyg_data( graph, output_dir, event_id):
        torch.save(graph, os.path.join(output_dir, f"event{event_id}-graph.pyg"))    

if __name__ == '__main__':
    main()