import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
import queue
import os
import copy
import numpy as np
from torch.utils import tensorboard
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
from concurrent.futures import ProcessPoolExecutor    
from model.monomer_train.pdbload import ProteinDataset,PaddingCollate 
from model.monomer_train.utils import Logger,save_checkpoint,restore_checkpoint,recursive_to,random_mask_batch
from model.monomer_train.pdb_model import loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

scaler = torch.cuda.amp.GradScaler()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#conda activate proteinsgm
#python proteinMPNN_train_monomer.py --dataset_path pdb --batch_size=8 --sampler_number=200 --epochs=500 --output_dir training_monomer

def get_args_parser():
    parser = argparse.ArgumentParser('proteinMPNN training and inference ', add_help=True)

    parser.add_argument("--dataset_path", required=True, type=str, help="pathway in train dataset")
    #parser.add_argument("--sample_names", required=True, nargs='+', action='store', type=str, help="train sample namses")
    parser.add_argument("--previous_checkpoint", type=str, help="previous_checkpoint")

    parser.add_argument("--output_dir", required=True, type=str, help="output/model and output")
    parser.add_argument("--load_model_path", type=str, default=None, help="load model path")

    parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
    parser.add_argument("--sampler_number",type=int, default=200, help="sampler number Dataloader randomsamplering")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--steps", type=int, default=500, help="logger number")

    parser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")
    parser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")
    parser.add_argument("--min_res_num", type=int, default=50, help="min protein length")
    parser.add_argument("--max_res_num", type=int, default=800, help="max protein length")
    parser.add_argument("--ss_constraints",type=bool, default=True, help="ss backbone, 2D structure")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dim")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="encoder layers number")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--device", type=str, default='cuda:0', help="training with cuda:0")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=0, help="CUDA device ids")

    parser.add_argument("--save_checkpoint", type=bool, default=True, help="Save checkpoint: true or false")
    parser.add_argument("--save_prediction", type=bool, default=True, help="Save prediction: true or false")
    parser.add_argument("--mixed_precision",type=bool, default=False, help="mixed precision:scalar true or false")
    parser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")

    parser.add_argument("--seed", type=int, default=42, help="seed 42")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--rescut", type=int, default=3.5, help="PDB resolution cutoff")
    parser.add_argument("--dropout",type=int, default=0.1, help="droup out")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    return parser

def main (args, mode = 'train'):

    if mode == 'train':

        print("===============================Dataloading") ############### Dataloading

        scaler = torch.cuda.amp.GradScaler()

        # Dataset
        dataset = ProteinDataset(args.dataset_path, args.min_res_num,
                             args.max_res_num, args.ss_constraints)
        # coords:[118,4,3]  coords_6d:[8,118,118]  aa:[118]  mask_pair:[118,118] true/false  ss_indices:'6:15,21:34,49:61,98:115'

        train_size = int(0.95 * len(dataset))
        eval_size = len(dataset) - train_size
        # DataLoader
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size],
                                                      generator=torch.Generator().manual_seed(args.seed))

        train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=True,
                                                       num_samples=args.sampler_number*args.batch_size)
        train_dataload = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,batch_size=args.batch_size,
                                              collate_fn=PaddingCollate(args.max_res_num))

        eval_sampler = torch.utils.data.RandomSampler(eval_dataset,replacement=True,
                                                      num_samples=args.sampler_number*args.batch_size)
        eval_dataload = torch.utils.data.DataLoader(eval_dataset,sampler=eval_sampler,batch_size=args.batch_size,
                                              collate_fn=PaddingCollate(args.max_res_num))

        print(f"Train Dataset {args.dataset_path} loading")

        # iter
        train_iter = iter(train_dataload)
        eval_iter = iter(eval_dataload)
        print(f"{args.sampler_number*args.batch_size} samples in traing/test iter and batch_size is {args.batch_size}")

        print("==============================Initialize model") ########### Initialize model
        # model
        model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)

        model.to(args.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_std_opt(model.parameters(), args.hidden_dim, 0) # model.parameters() model_size 根据step更新lr的优化器
        state = dict(optimizer=optimizer, model=model, step=0)

        #if PATH:
        #    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("============================== Constract workdir and logdir") ########## Constract workdir and logdir
        workdir = Path(args.output_dir,'time')
        #workdir = Path(args.output_dir, Path(args.config).stem, time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
        workdir.mkdir(exist_ok=True,parents=True)

        tb_dir = workdir.joinpath("model_weights")
        tb_dir.mkdir(exist_ok=True)
        writer = tensorboard.SummaryWriter(tb_dir)
        logger = Logger(tb_dir)

        checkpoint_dir = workdir.joinpath("checkpoints")
        checkpoint_meta_dir = workdir.joinpath("checkpoints-meta", "checkpoint.pth") # resume
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_meta_dir.parent.mkdir(exist_ok=True)

        if checkpoint_meta_dir.is_file():
            state = restore_checkpoint(checkpoint_meta_dir, state, args.device) # utils.restore_checkpoint
            initial_step = int(state['step'])
            #model.load_state_dict(state['model'],False)
            model=state['model']
            #optimizer.optimizer.load_state_dict(state['optimizer'])
            optimizer=state['optimizer']
        else:
            initial_step = 0

        print("============================== Model running") ########## Model run

        # model training
        for step in range(initial_step, args.epochs + 1):
            train_batch = recursive_to(next(train_iter),args.device)  # 递归，将数据集按所需类型整合
            train_batch = random_mask_batch(train_batch,args.device,mask_max_len=0.35,mask_min_len=0.25,random_mask_prob=0.25,contiguous_mask_prob=0.25) # 30% protein mask [0]
            train_sum, train_weights, train_acc = 0., 0., 0.
            # batch
            # chain:[B,L], mask:[B,L] [0], mask_pair:[B,L,L], coords:[B,L,4,3], coords_6d:[B,L,L], aa[B,L], aa_str[B,L], aa_index:[B,L], ss_mask:[B,L] [0], ss_indices:[B,L],
            #  mask_random[B,L], mask_random [B,L,L]
            mask_position = (train_batch['ss_mask']+train_batch['mask_random']).bool().long() # (1-mask_random) + (1-ss_mask) -> valid_position

            optimizer.zero_grad()
            mask_for_loss = train_batch['mask']*mask_position # total mask
            #(self, X, S, mask, ss_mask, aa_index, chain_encoding_all)
            # mixed_precision scalar
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs = model(train_batch['coords'], train_batch['aa'], train_batch['mask'], mask_position,
                                      train_batch['aa_index'], train_batch['chain_embed'])  # log_probs[B,L,21]
                    _, loss_av_smoothed_train = loss_smoothed(train_batch['aa'], log_probs, mask_for_loss) # loss [B]  loss_av_smoothed[B]
           
                scaler.scale(loss_av_smoothed_train).backward()
                     
                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm) # 缩放梯度，防止梯度过大爆炸

                scaler.step(optimizer)
                scaler.update()

            # direct step
            else:

                log_probs = model(train_batch['coords'], train_batch['aa'], train_batch['mask'], mask_position,
                                  train_batch['aa_index'], train_batch['chain_embed'])  # log_probs[B,L,21]
                _, loss_av_smoothed_train = loss_smoothed(train_batch['aa'], log_probs, mask_for_loss) # loss [B]  loss_av_smoothed[B]

                loss_av_smoothed_train.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm) # 缩放梯度，防止梯度过大爆炸

                optimizer.step()

                # loss,true_false
                loss, loss_av, true_false = loss_nll(train_batch['aa'], log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy() #loss * mask_for_loss
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy() #true_false * mask_for_loss
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy() # weight sum(mask_for_loss)
             
            # logout/checkpoint
            if step % int(args.epochs/args.steps) == 0:
                writer.add_scalar("training_loss",loss_av_smoothed_train, step)

            # Save a temporary checkpoint to resume training after pre-emption periodically
            if step != 0 and step % int(args.epochs/(args.steps/10)) == 0:
                state['step'] = step
                save_checkpoint(checkpoint_meta_dir, state)

            #============================================== model evaluate

            # model evaluate
            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights, validation_acc = 0., 0., 0.

            # valid batch
            valid_batch = recursive_to(next(eval_iter),args.device)
            valid_batch = random_mask_batch(valid_batch,args.device,mask_max_len=0.35,mask_min_len=0.25,random_mask_prob=0.33,contiguous_mask_prob=0.33) # protein mask
            mask_position = (valid_batch['ss_mask']+valid_batch['mask_random']).bool().long()

            mask_for_loss = valid_batch['mask']*mask_position # mask_loss

            log_probs = model(valid_batch['coords'], valid_batch['aa'], valid_batch['mask'], mask_position,
                              valid_batch['aa_index'], valid_batch['chain_embed'])  # log_probs[B,L,21]

            #model_MPNN (model_feature(GNN(E:posi+chain V:posi+chain)), encoder(GNN message_pass), decoder(GNN message_passing + S -> predict S) loss)
            _, loss_av_smoothed_val = loss_smoothed(valid_batch['aa'], log_probs, mask_for_loss)
            loss, loss_av, true_false = loss_nll(valid_batch['aa'], log_probs, mask_for_loss)
                    
            validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            # Report the loss on an evaluation dataset periodically
            if step % int(args.epochs/args.steps) == 0: # 整数倍 % 除法取余数
                writer.add_scalar("eval_loss", loss_av_smoothed_val.item(), step)


            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3) # 三位小数
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)

            #============================================== Save checkpoint and logging
 
            # Save a checkpoint periodically
            if step != 0 and step % int(args.epochs/(args.steps/10)) == 0 or step == args.epochs:
                # Save the checkpoint.
                save_step = step // int(args.epochs/(args.steps/10)) # // 除法取整数
                state['step'] = step
                save_checkpoint(checkpoint_dir.joinpath(f'checkpoint_{save_step}.pth'), state)

            # logging updata
            if step % int(args.epochs/args.steps) == 0:
                logger.write('step:{},loss_smooth_train:{},loss_smooth_val:{},loss_nll_perplexity_train:{},loss_nll_perplexity_val:{},acc_tain:{},acc_val:{}\n'.format(
                            step,loss_av_smoothed_train.item(),loss_av_smoothed_val.item(),
                            train_perplexity_,validation_perplexity_,train_accuracy_,validation_accuracy_))


if __name__ =='__main__':

    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mode = 'train' # train/evalu

    if mode == 'train':
        main(args, mode=mode)
    #else:
    #    main(args, mode=mode)
