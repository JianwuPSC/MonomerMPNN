import argparse
import os.path
from pathlib import Path
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
from torch.utils import tensorboard 
from model.monomer_infer.monomer_infer_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, parse_PDB, parse_fasta
from model.monomer_infer.monomer_infer_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
from model.monomer_infer.pdbload import PDBDataset,PaddingCollate
from model.monomer_infer.utils import Logger,save_checkpoint,restore_checkpoint,recursive_to,random_mask_batch,S_to_seq
from model.monomer_infer.pssmload import pssm_load

#conda activate proteinsgm
#python proteinMPNN_infer_monomer.py --pdb_path /home/wuj/data/protein_design/SMURF_protein/example/train_pdb/template_rank.pdb --name=CNR --load_model_path training_monomer/time/checkpoints-meta/checkpoint.pth --model_name=checkpoint33 --output_dir=infer_monomer --batch_size=1 --num_seq_per_target=2 --device=cuda:0 --seed=37 --fixed_aa=1:5,10:15 --max_res_num=136

def get_args_parser():
    parser = argparse.ArgumentParser('proteinMPNN inference ', add_help=True)

    parser.add_argument("--pdb_path", required=True, type=str, help="pathway in train dataset")
    parser.add_argument("--name", required=True, type=str, help="train sample namses")
    parser.add_argument("--load_model_path", type=str, help="previous_checkpoint")
    parser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002 =version with 48 edges 0.10A noise")

    parser.add_argument("--output_dir", required=True, type=str, help="output/model and output")
    parser.add_argument("--ca_only", action="store_true", default=False, help="Parse CA-only structures and use CA-only models (default: false)")
    parser.add_argument("--save_score", type=int, default=1, help="0 for False, 1 for True; save score=-log_prob to npy files")
    parser.add_argument("--save_probs", type=int, default=1, help="0 for False, 1 for True; save MPNN predicted probabilites per position")

    parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
    parser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    parser.add_argument("--num_edges",type=int, default=48, help=" GNN edge")

    parser.add_argument("--score_only",type=int, default=0, help="0 for False, 1 for True; score only not in sample")
    parser.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; p(s_i given the rest of the sequence and backbone)")
    parser.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True;  p(s_i given backbone) in one forward pass")
    parser.add_argument("--min_res_num", type=int, default=48, help="min protein length")
    parser.add_argument("--max_res_num", type=int, default=256, help="max protein length")
    parser.add_argument("--ss_constraints",type=bool, default=True, help="ss backbone, 2D structure")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dim")
    parser.add_argument("--num_layers", type=int, default=3, help="encoder layers number")
    parser.add_argument("--backbone_noise", type=float, default=0.2, help="backbone noise")

    parser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--device", type=str, default='cuda:0', help="training with cuda:0")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=0, help="CUDA device ids")

    parser.add_argument("--seed", type=int, default=42, help="seed 42")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")

    parser.add_argument("--omits_aa", type=str, default=None, help="Specify omitted sequence 35:50,56:70.")
    parser.add_argument("--fixed_aa", type=str, default=None, help="Specify fixed sequence 43:45")
    parser.add_argument("--omit_AAs", default='X', help="default aa is X")

    parser.add_argument("--bias_aa",type=bool ,default=False, help="bias aa (adding global polar amino acid bias.)")
    parser.add_argument("--bias_position", type=str,default=None, help="Path to dictionary with per position bias.") 
    
    parser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    parser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    parser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    parser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    
    return parser

def main (args, mode = 'test'):
 
    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    # bias_AA_dict    
    bias_AA_dict = {'D': 1.39, 'E': 1.39, 'H': 1.39, 'K': 1.39, 'N': 1.39, 'Q': 1.39, 'R': 1.39, 'S': 1.39, 'T': 1.39, 'W': 1.39, 'Y': 1.39}
    # bias_AAs_np
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]
    if not args.bias_aa:
       bias_AAs_np, bias_AA_dict = np.zeros(len(alphabet)), None

    # bias position
    bias_position = None
    # fixed_list
    fixed_list = args.fixed_aa
    # omit_list
    omit_list = args.omits_aa

    temperatures = [float(item) for item in args.sampling_temp.split()] # 0.2,0.25,0.3
    alphabet_dict = dict(zip(alphabet, range(21)))

    omit_AAs_np = np.array([AA in args.omit_AAs for AA in alphabet]).astype(np.float32) # [0,0,0,...1]

    print("========================================== load pdb file and pssm")
    #pdb_dict_list = parse_PDB(args.pdb_path, ca_only=args.ca_only) # pdb_dict_list[my_dict[coords_chain_A,name,num_of_chains]]
    # list[0]  name:6MRR, num_of_chains:1, seq:68str, seq_chain_A:68str, coords_chain_A:{CA_chain_A:[68,3],N_chain_A:[68,3],C_chain_A:[68,3],O_chain_A:[68,3]} 
    #dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
    #all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]

    #if args.design_chains: # which pdb chain design
    #    designed_chain_list = [str(item) for item in args.design_chains.split()]
    #else:
    #    designed_chain_list = all_chain_list
    #fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
    #chain_id_dict = {}
    #chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

    # other methord read pdb
    pdb_dataset = PDBDataset(args.pdb_path, args.min_res_num, args.max_res_num, args.ss_constraints, fixed_list,omit_list)
    dataset_valid = torch.utils.data.DataLoader(pdb_dataset, batch_size=args.batch_size,
                                              collate_fn=PaddingCollate(args.max_res_num))
    if  args.pssm_multi == 1:
        pssm_coef, pssm_bias, pssm_odds = pssm_load(args.batch, args.pssm_path, args.length, args.max_res_num, args.device)
        pssm_log_odds_mask = (pssm_odds > args.pssm_threshold).float()
    else:
        pssm_coef, pssm_bias, pssm_odds = None,None,None
        pssm_log_odds_mask = None

    print("========================================== load model and updata")
    checkpoint = torch.load(args.load_model_path, map_location=args.device)    
    model = ProteinMPNN(ca_only=args.ca_only, num_letters=21, node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers, 
                        augment_eps=args.backbone_noise, k_neighbors=args.num_edges)

    model.to(args.device)
    model.load_state_dict(checkpoint['model'], strict=False) # model.state_dict()
    #
    model.eval()
    #
    print("========================================== Constract workdir and logdir")
    # Build paths for experiment
    workdir = Path(args.output_dir,'sampling')
    workdir.mkdir(exist_ok=True,parents=True)

    tb_dir = workdir.joinpath("Logger")
    tb_dir.mkdir(exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)
    logger = Logger(tb_dir)

    seqs_dir = workdir.joinpath("seqs")
    seqs_dir.mkdir(exist_ok=True)
    
    print("========================================== Sampling seqs and count scores")
    # Timing
    start_time = time.time()
    total_residues = 0
    protein_list = []
    total_step = 0
    # Validation epoch
    with torch.no_grad(): 
        test_sum, test_weights = 0., 0.
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            global_score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []

            valid_batch = recursive_to(protein,args.device)
            valid_batch = random_mask_batch(valid_batch,args.device,mask_max_len=0.35,mask_min_len=0.25,random_mask_prob=0.33,contiguous_mask_prob=0.33) # protein mask
            mask_position = (valid_batch['ss_mask']+valid_batch['mask_random']).bool().long() # (1-fixed_mask) * (omit_mask) -> valid_position
            mask_position = torch.ones(mask_position.shape).to(args.device)
 
            # chain_embed:[B,L], mask:[B,L], mask_pair:[B,L,L], coords:[B,L,4,3], coords_6d:[B,L,L], aa[B,L], aa_str[B,L], aa_index:[B,L], ss_mask:[B,L], ss_indices:[B,L],
            # fixed_mask:[B,L], omit_mask:[B,L] mask_random[B,L], mask_random_pair [B,L,L]
             
            # tied_featurize 20个 输出 chain_list_list:[A,B,C] visible_list:[A,B] masked_list[A,C] masked_chain_length_list[L,L,L]
            # chain_M_pos:[B,L] [0] fixed_position, omit_AA_mask:[B,L] [1] omit_position, residue_idx:[B,L] [1,2,3,101,102,103]
            # chain_M_pos:[B,L] [0] fixed_position, omit_AA_mask:[B,L] [1] omit_position, residue_idx:[B,L] [1,2,3,101,102,103]
            # dihedral_mask:[B,L,3] phi psi omega, pssm_coef:[B,L], pssm_bias:[B,L,21], pssm_log_odds_all:[B,L,21]
            # bias_by_res_all:[B,L], tied_pos_list_of_lists_list
            
            if args.score_only: # score_only loss nll 缺少采样过程 sampling 用来评估
                structure_sequence_score_file = str(seqs_dir) + '/score_only/' + args.name

                for j in range(args.num_seq_per_target*args.batch_size): # 1 
                    randn_1 = torch.randn(valid_batch['mask'].shape, device=args.device)
                    log_probs = model(valid_batch['coords'], valid_batch['aa'], valid_batch['mask'], mask_position*valid_batch['fixed_mask'], 
                                      valid_batch['aa_index'], valid_batch['chain_embed'], randn_1)

                    mask_for_loss = valid_batch['mask']*mask_position*valid_batch['fixed_mask']
                    scores = _scores(valid_batch['aa'], log_probs, mask_for_loss)
                    native_score = scores.cpu().data.numpy()
                    native_score_list.append(native_score)
                    global_scores = _scores(valid_batch['aa'], log_probs, valid_batch['mask'])
                    global_native_score = global_scores.cpu().data.numpy()
                    global_native_score_list.append(global_native_score)

                native_score = np.concatenate(native_score_list, 0)
                global_native_score = np.concatenate(global_native_score_list, 0)
                ns_mean = native_score.mean()
                ns_mean_print = np.format_float_positional(np.float32(ns_mean), unique=False, precision=4) # score mean 精确四位小数
                ns_std = native_score.std()
                ns_std_print = np.format_float_positional(np.float32(ns_std), unique=False, precision=4) # score std 精确四位小数

                global_ns_mean = global_native_score.mean()
                global_ns_mean_print = np.format_float_positional(np.float32(global_ns_mean), unique=False, precision=4) # score mean 精确四位小数
                global_ns_std = global_native_score.std()
                global_ns_std_print = np.format_float_positional(np.float32(global_ns_std), unique=False, precision=4) # score std 精确四位小数

                ns_sample_size = native_score.shape[0] # sample_size
                seq_str = _S_to_seq(valid_batch['aa'][0,], mask_position[0,]) # predict S_seq
                #seq_str = S_to_seq(valid_batch['aa'][0,], mask_position[0,]) # predict S_seq
                np.savez(structure_sequence_score_file, score=native_score, global_score=global_native_score, S=valid_batch['aa'][0,].cpu().numpy(), seq_str=seq_str)
                print(f'Score for {args.name} from PDB, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size},  global score, \
                        mean: {global_ns_mean_print}, std: {global_ns_std_print}, sample size: {ns_sample_size}')

            elif args.conditional_probs_only: # model.conditional_probs -> log_conditional_probs
                conditional_probs_only_file = str(seqs_dir) + '/conditional_probs_only/' + args.name
                log_conditional_probs_list = []
                for j in range(args.num_seq_per_target*args.batch_size):
                    randn_1 = torch.randn(valid_batch['mask'].shape, device=args.device)
                    log_conditional_probs = model.conditional_probs(valid_batch['coords'], valid_batch['aa'], valid_batch['mask'], mask_position*valid_batch['fixed_mask'], 
                                                                    valid_batch['aa_index'], valid_batch['chain_menbed'], randn_1, args.conditional_probs_only_backbone)
                    log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())
                concat_log_p = np.concatenate(log_conditional_probs_list, 0) #[B, L, 21]
                mask_out = (valid_batch['mask']*mask_position*valid_batch['fixed_mask'])[0,].cpu().numpy()
                np.savez(conditional_probs_only_file, log_p=concat_log_p, S=valid_batch['aa'][0,].cpu().numpy(), mask=valid_batch['mask'][0,].cpu().numpy(), design_mask=mask_out)

            elif args.unconditional_probs_only: # model.unconditional_probs -> log_unconditional_probs
                unconditional_probs_only_file = str(seqs_dir) + '/unconditional_probs_only/' + args.name
                log_unconditional_probs_list = []
                for j in range(args.num_seq_per_target*args.batch_size):
                    log_unconditional_probs = model.unconditional_probs(valid_batch['coords'], valid_batch['mask'], valid_batch['aa_index'], valid_batch['chain_embed'])
                    log_unconditional_probs_list.append(log_unconditional_probs.cpu().numpy())
                concat_log_p = np.concatenate(log_unconditional_probs_list, 0) #[B, L, 21]
                mask_out = (valid_batch['mask']*mask_position*valid_batch['fixed_mask'])[0,].cpu().numpy()
                np.savez(unconditional_probs_only_file, log_p=concat_log_p, S=valid_batch['aa'][0,].cpu().numpy(), mask=valid_batch['mask'][0,].cpu().numpy(), design_mask=mask_out)

            else: # more usefulway
                randn_1 = torch.randn(valid_batch['mask'].shape, device=args.device) # randn
                log_probs = model(valid_batch['coords'], valid_batch['aa'], valid_batch['mask'], mask_position*valid_batch['fixed_mask'], 
                                  valid_batch['aa_index'], valid_batch['chain_embed'], randn_1) # log_probs
                mask_for_loss = valid_batch['mask']*mask_position*valid_batch['fixed_mask']
                scores = _scores(valid_batch['aa'], log_probs, mask_for_loss) # score only the redesigned part
                native_score = scores.cpu().data.numpy()
                global_scores = _scores(valid_batch['aa'], log_probs, valid_batch['mask']) # score the whole structure-sequence
                global_native_score = global_scores.cpu().data.numpy()
                # Generate some sequences
                ali_file = str(seqs_dir) + '/' + args.name +'_sample' + '.fa'
                score_file = str(seqs_dir) + '/' + args.name +'_scores' + '.npz'
                probs_file = str(seqs_dir) + '/' + args.name +'_probs' + '.npz'
                
                t0 = time.time()
                with open(ali_file, 'w') as f:
                    for temp in temperatures: # 0.2,0.25,0.3
                        for j in range(args.num_seq_per_target*args.batch_size): # seq traget 生成几条序列
                            randn_2 = torch.randn(valid_batch['aa'].shape, device=args.device)

                            #sample_dict = model.sample(valid_batch['coords'], randn_2, valid_batch['aa'], mask_position, valid_batch['chain_embed'], valid_batch['aa_index'], 
                            #                           mask=valid_batch['mask'], temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                            #                           chain_M_pos=valid_batch['fixed_mask'], omit_AA_mask=valid_batch['omit_mask'],omit_AA_mask_flag=None, 
                            #                            pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), 
                            #                           pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_position)

                            sample_dict = model.sample(valid_batch['coords'], randn_2, valid_batch['aa'], mask_position, valid_batch['chain_embed'], valid_batch['aa_index'],
                                                       mask=valid_batch['mask'], temperature=temp,omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                                                       chain_M_pos=valid_batch['fixed_mask'], omit_AA_mask=valid_batch['omit_mask'],omit_AA_mask_flag=False)


                            S_sample = sample_dict["S"] # {"S": S, "probs": all_probs, "decoding_order": decoding_order}
                            # log_probs
                            log_probs = model(valid_batch['coords'], S_sample, valid_batch['mask'], mask_position*valid_batch['fixed_mask'], valid_batch['aa_index'], 
                                              valid_batch['chain_embed'], randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                            # score 与 global score 所用mask的范围
                            mask_for_loss = valid_batch['mask']*mask_position*valid_batch['fixed_mask'] # [B,L]
                            scores = _scores(S_sample, log_probs, mask_for_loss) # [B,L] [B,L,21] [B,L] -> [B,1]
                            scores = scores.cpu().data.numpy()
                            
                            global_scores = _scores(S_sample, log_probs, valid_batch['mask']) #score the whole structure-sequence # [B,L] [B,L,21] [B,L] -> [B,1]
                            global_scores = global_scores.cpu().data.numpy()
                            
                            all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                            all_log_probs_list.append(log_probs.cpu().data.numpy())
                            S_sample_list.append(S_sample.cpu().data.numpy())
							
                            for b_ix in range(args.batch_size):
                                # recovery
                                seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(valid_batch['aa'][b_ix], 21)*
                                                             torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)* mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])

                                # _S_to_seq
                                seq = _S_to_seq(S_sample[b_ix],mask_position[b_ix])
                                #seq = S_to_seq(S_sample[b_ix],mask_position[b_ix],args.length)
                                # score
                                score = scores[b_ix]
                                score_list.append(score)
                                global_score = global_scores[b_ix] 
                                global_score_list.append(global_score)
                                # native_seq
                                native_seq = _S_to_seq(valid_batch['aa'][b_ix], mask_position[b_ix])
                                #native_seq = S_to_seq(valid_batch['aa'][b_ix], mask_position[b_ix],args.length)

			        ###############
                                if b_ix == 0 and j==0 and temp==temperatures[0]:

                                    native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                                    global_native_score_print = np.format_float_positional(np.float32(global_native_score.mean()), unique=False, precision=4)

                                    if args.ca_only:
                                        print_model_name = 'CA_model_name'
                                    else:
                                        print_model_name = 'model_name'
                                    f.write('>{}, score={}, global_score={}, fixed_position={}, model_name={}, seed={}\n{}\n'.format(args.name, native_score_print, global_native_score_print, args.fixed_aa, args.model_name, seed, native_seq)) #write the native sequence
									
                                score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                                global_score_print = np.format_float_positional(np.float32(global_score), unique=False, precision=4)
                                seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
                                sample_number = j*args.batch_size+b_ix+1

                                f.write('>T={}, sample={}, score={}, global_score={}, seq_recovery={}\n{}\n'.format(temp,sample_number,score_print,
                                        global_score_print,seq_rec_print,seq))

                #write generated sequence
                if args.save_score:
                    np.savez(score_file, score=np.array(score_list, np.float32), global_score=np.array(global_score_list, np.float32))
                if args.save_probs:
                    all_probs_concat = np.concatenate(all_probs_list)
                    all_log_probs_concat = np.concatenate(all_log_probs_list)
                    S_sample_concat = np.concatenate(S_sample_list)
                    np.savez(probs_file, probs=np.array(all_probs_concat, np.float32), log_probs=np.array(all_log_probs_concat, np.float32), 
                             S=np.array(S_sample_concat, np.int32), mask=mask_for_loss.cpu().data.numpy())

                t1 = time.time()
                dt = round(float(t1-t0), 4)
                num_seqs = len(temperatures)*args.num_seq_per_target*args.batch_size*args.batch_size
                total_length = valid_batch['aa'].shape[1]
                print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')

if __name__ =='__main__':

    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mode = 'test' # train/evalu

    if mode == 'test':
     #   print('========  Train  =============================================================')
        main(args, mode=mode)
 
