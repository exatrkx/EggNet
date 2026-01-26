import os, sys
import particle # Scikit HEP
import argparse
from utils import load_config, get_realpath
import torch_geometric as pyg
from torch_geometric.data import Data
import torch
from tqdm import tqdm # the best python package
from tqdm.contrib.concurrent import process_map
from functools import partial
import matplotlib.pyplot as plt
import enum
from eggnet.utils.mapping import map_tracks_to_nodes

### NOTE: To get the (x, y, z) or (r, phi, psi) components of the momentum or charge momentum ratio,
###       just add the coordinate after the datakey seperated by an underscore. 
# (TRACK_..._DATAKEY_x)
# (TRACK_..._DATAKEY_y)
# (TRACK_..._DATAKEY_z)

XYZ_COORDINATES = ('x', 'y', 'z')
CYLINDRICAL_COORDINATES = ('r', 'phi', 'psi') #TODO Future coordinate systems
# New event keys
TRACK_CHARGE_DATAKEY = 'track_particle_charge'
HIT_CHARGE_DATAKEY = 'hit_particle_charge'
TRACK_CHARGE_MOMENTUM_RATIO_DATAKEY = 'track_charge_pt_ratio' ### BUG: THIS _NEEDS_ TO START WITH 'track_' TO BE DETECTED BY THE DATASET CLASS
HIT_CHARGE_MOMENTUM_RATIO_DATAKEY = 'hit_charge_pt_ratio'

# Already in event keys
TRACK_PDGID_DATAKEY = 'track_particle_pdgId'
HIT_PDGID_DATAKEY = 'hit_particle_pdgId'
TRACK_MOMENTUM_DATAKEY = 'track_particle_pt'
HIT_MOMENTUM_DATAKEY = 'hit_particle_pt'

def main():
    # Skip most of the datareader class from Acorn by adjusting the PyG data objects directly
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML file containing metadata about task')
    parser.add_argument('-w', '--workers', required=False, default=1, 
                        help='Number of workers that can be allocated to perform multiprocessing steps. Default is 1.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite of existing files')
    global args
    args = parser.parse_args()
    config_filepath = args.config_file
    assert config_filepath.endswith('.yaml') or config_filepath.endswith('.yml')
    config = load_config(config_filepath)
    
    assert 'input_dir' in config
    assert 'output_dir' in config

    input_dir = config.get('input_dir')
    if not os.path.isdir(input_dir):
        raise NameError(f"Path {input_dir} does not exist")
    input_trainset_path = input_dir + os.sep + 'trainset'
    input_testset_path = input_dir + os.sep + 'testset'
    input_valset_path = input_dir + os.sep + 'valset'
    
    output_dir = config.get('output_dir')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, mode=0o661)
    output_trainset_path = output_dir + os.sep + 'trainset'
    output_testset_path = output_dir + os.sep + 'testset'
    output_valset_path = output_dir + os.sep + 'valset'
    
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
                desc=f"INFO: Processing {dataset_name} PyG Files",
                tqdm_class=tqdm,
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
    if not os.path.isdir(output_dir):
        try:
            # 0o666 -> rw-rw-rw-
            os.makedirs(output_dir, mode=0o777, exist_ok=True)
        except FileExistsError as e:
            print("DEBUG: File Exists Error raised")
    data_object = torch.load(os.path.join(input_dir, file))
    if args.verbose:
        print(f"INFO: Processing event-{data_object.event_id}")
    # if CHARGE_DATAKEY in data_object.keys() and not args.force:
    #     if args.verbose:
    #         print(f"INFO: Skipping event-{data_object.event_id}")
    # elif CHARGE_MOMENTUM_RATIO_DATAKEY in data_object.keys() and not args.force:
    #     if args.verbose:
    #         print(f"INFO: Skipping event-{data_object.event_id}")
    # else:
    # (Px, Py), Pz (or other coordinate system), Vx, Vy, Vz
    # Note (Px, Py) unresolvable without extra dimension
    # save_distribution_stats_figure(get_realpath(), charge_pt_ratio, "charge_pt_ratio")
    # save_distribution_stats_figure(get_realpath(), charges_tensor, "charge")
    # save_distribution_stats_figure(get_realpath(), momentum_tensor, "momentum")
    # exit()
    if config.get('add_charge_momentum_data'):
        def add_track_qpt_ratio(suffix=''):
            if TRACK_CHARGE_MOMENTUM_RATIO_DATAKEY+suffix in data_object.keys() and not args.force:
                if args.verbose:
                    print(f"INFO: Skipping event-{data_object.event_id} (track charge/momentum ratio already exists)")
            else:
                pdgids = data_object.get(TRACK_PDGID_DATAKEY)
                if pdgids is None:
                    raise KeyError(f"ERROR: PdgID not specified in PyG file (event-{data_object.event_id})")
                charges = [] # temporarily store in python object before converting to torch tensor
                for i, id in enumerate(pdgids):
                    charge = particle.pdgid.charge(particle.PDGID(id))
                    charges.append(charge)
                charges_tensor = torch.Tensor(charges)
                if config.get('add_charge_data'):
                    data_object[TRACK_CHARGE_DATAKEY] = charges_tensor
                # Compute charge/momentum ratio
                hit_momentum_tensor:torch.Tensor = data_object[TRACK_MOMENTUM_DATAKEY+suffix]
                # assert momentum_tensor.size() == charges_tensor.size()
                # assert momentum_tensor.size() == data_object[CHARGE_DATAKEY].size()
                charge_pt_ratio = charges_tensor / hit_momentum_tensor
                assert isinstance(charge_pt_ratio, torch.Tensor), f"{type(charge_pt_ratio)=}"
                assert charge_pt_ratio.size() == data_object["track_particle_pt"].size()
                data_object[TRACK_CHARGE_MOMENTUM_RATIO_DATAKEY+suffix] = charge_pt_ratio
        add_track_qpt_ratio()
        for coord in XYZ_COORDINATES:
            add_track_qpt_ratio('_' + coord)
            
    if config.get('add_hit_charge_momentum_data'):
        def add_hit_qpt_ratio(suffix=''):
            if HIT_CHARGE_MOMENTUM_RATIO_DATAKEY+suffix in data_object.keys() and not args.force:
                if args.verbose:
                    print(f"INFO: Skipping event-{data_object.event_id} (hit charge/momentum ratio already exists)")
            else:
                # ======= DIRECT CALCULATION OF HIT CHARGE/MOMENTUM RATIO =======
                hit_pdgids = data_object[HIT_PDGID_DATAKEY]
                if hit_pdgids is None:
                    raise KeyError(f"ERROR: PdgID not specified in PyG file (event-{data_object.event_id})")
                hit_charges = [] # temporarily store in python object before converting to torch tensor
                for i, id in enumerate(hit_pdgids):
                    id = id.item() if isinstance(id, torch.Tensor) else id
                    if id != id: # id is NaN => Fake hit -- must account for this
                        # We can assign a fake hit a charge of 0
                        charge = 0.0
                    else:
                        charge = particle.pdgid.charge(particle.PDGID(id))
                    hit_charges.append(charge)
                hit_charges_tensor = torch.Tensor(hit_charges)
                if config.get('add_charge_data'):
                    data_object[HIT_CHARGE_DATAKEY] = hit_charges_tensor
                # My attempt at projecting momentum into the right coordinate
                hit_momentum_tensor:torch.Tensor = data_object[HIT_MOMENTUM_DATAKEY] * data_object["hit_particle_v{}".format(suffix[1:])] 
                hit_charge_pt_ratio = hit_charges_tensor / hit_momentum_tensor
                assert hit_charge_pt_ratio.size() == data_object["hit_particle_pt"].size(), hit_charge_pt_ratio.size()

                # ======== IDX MAPPING TECHNIQUE (BUGGY) =======
                # idx_mapping = {v.item(): i for i, v in enumerate(data_object["track_particle_id"])}
                # idx_mapping[0] = 0 # The '0' particle ID represents either a fake or a non-accounted-for particle. For simplicity, let these be zero
                # indicies = torch.tensor(
                #     [round(idx_mapping.get(v.item(), 0)) for v in data_object["hit_particle_id"]]
                #     ) # BUG: The particle ID in hit_particle_id is not guaranteed to be in track_particle_id. Why?
                # hit_charge_pt_ratio = data_object.get(
                #         CHARGE_MOMENTUM_RATIO_DATAKEY
                #     )[indicies]
                
                # ======== EDGE MAPPING TECHINIQUE (BUGGY) =======
                # hit_charge_pt_ratio = map_tracks_to_nodes(
                #     charge_pt_ratio,
                #     data_object.get("track_edges"),
                #     num_nodes=data_object.num_nodes
                # ) # BUG: This is not guaranteed to be the same size as hit_particle_id (ex. size 301619 != 301626)
                assert hit_charge_pt_ratio is not None, f"Hit charge/momentum ratio not found in event-{data_object.event_id}"
                assert hit_charge_pt_ratio.size() == data_object.hit_particle_id.size(), f"{hit_charge_pt_ratio.size()} != {data_object.hit_particle_id.size()}"
                data_object[HIT_CHARGE_MOMENTUM_RATIO_DATAKEY+suffix] = hit_charge_pt_ratio
        add_hit_qpt_ratio() 
        if config.get('add_xyz_charge_momentum_data'):
            for coord in XYZ_COORDINATES:
                add_hit_qpt_ratio(suffix = '_' + coord)
    save_pyg_data(data_object, os.path.join(output_dir), data_object.event_id)
    validate_pyg_data(output_dir, data_object.event_id, config=config)

def save_distribution_stats_figure(directory, x:torch.Tensor, label:str):
    assert os.path.isdir(directory)
    standard_deviation, mean = torch.std_mean(x)
    print(f"{label} info".center(24, "="))
    print(f"{standard_deviation=} {mean=}")
    print()
    plt.figure()
    plt.hist(x, bins=100)
    plt.savefig(os.path.join(directory, label + "_histogram.png"))
            
def save_pyg_data( graph, output_dir, event_id):
    save_path = os.path.join(output_dir, f"event{event_id}-graph.pyg")
    if args.verbose:
        print(f"INFO: Saving graph data to {save_path}")
    torch.save(graph, save_path)    

def validate_pyg_data(output_dir, event_id, config={}):
    save_path = os.path.join(output_dir, f"event{event_id}-graph.pyg")
    # if args.verbose:
    #     print(f"INFO: Validating graph data at {save_path}")
    data_object = torch.load(save_path)
    assert TRACK_CHARGE_DATAKEY in data_object.keys()
    assert TRACK_CHARGE_MOMENTUM_RATIO_DATAKEY in data_object.keys()
    assert data_object[TRACK_CHARGE_DATAKEY].size() == data_object[TRACK_CHARGE_MOMENTUM_RATIO_DATAKEY].size()
    assert data_object[TRACK_CHARGE_DATAKEY].size() == data_object["track_particle_pt"].size()
    if config.get('add_xyz_charge_momentum_data'):  
        for suffix in XYZ_COORDINATES:
            assert TRACK_CHARGE_MOMENTUM_RATIO_DATAKEY + '_' + suffix in data_object.keys()
            assert data_object[TRACK_CHARGE_MOMENTUM_RATIO_DATAKEY + '_' + suffix].size() == data_object["track_particle_pt"].size()
        # charge.shape == charge_pt_ratio.shape == track_particle_id.shape

if __name__ == '__main__':
    main()