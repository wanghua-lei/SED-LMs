#!/usr/bin/env python3.9
import argparse
import time
from pprint import PrettyPrinter
import torch
import os
from loguru import logger
from ruamel import yaml
from tqdm import tqdm
import numpy as np
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from models.bart_captioning import BartCaptionModel
from models.bert_captioning import BertCaptionModel
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import setup_seed, set_logger, AverageMeter, decode_output
from tools.evaluate_psds import evaluate_psds

def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()

    epoch_loss = AverageMeter()
    start_time = time.time()  
    # batch accumulation parameter
    accum_iter = 4
    for batch_id, (audio, text, audio_name, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):

        step = len(dataloader) * (epoch - 1) + batch_id
        if scheduler is not None:
            scheduler(step)
        audio = audio.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        loss = model(audio, text) #use contrast
        
        loss.backward()
        if ((batch_id + 1) % accum_iter == 0):
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss.update(loss.cpu().item())
    elapsed_time = time.time() - start_time
    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


@torch.no_grad()
def validate(data_loader, model, device, log_dir, epoch, beam_size, eval=False):
    val_logger = logger.bind(indent=1)
    model.eval()
    with torch.no_grad():
        y_hat_all = [] 
        y_top_p = []
        ref_captions_dict = []
        file_names_all = []
        start_time = time.time()

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, caption_dict, audio_names, audio_ids = batch_data
            # move data to GPU
            audios = audios.to(device)

            output = model.generate(samples = audios, num_beams = beam_size)
            #output_top_p = model.generate(samples = audios, use_nucleus_sampling = True) #module
                                    
            y_hat_all.extend(output)
            #y_top_p.extend(output_top_p)

            ref_captions_dict.extend(caption_dict)
            file_names_all.extend(audio_names)
        #write to file
        #captions_pred_top, captions_gt = decode_output(y_top_p, ref_captions_dict, file_names_all,log_dir, epoch, beam_size = 0)                 
        captions_pred_greedy, captions_gt = decode_output(y_hat_all, ref_captions_dict, file_names_all,log_dir, epoch, beam_size = beam_size)

        #metrics = evaluate_metrics(captions_pred, captions_gt)
        #metrics_F1_top = evaluate_psds(captions_pred_top, eval = eval)
        metrics_F1_greedy = evaluate_psds(captions_pred_greedy, eval = eval)

        eval_time = time.time() - start_time
        val_logger.info(f'epoch {epoch}, eval time: {eval_time:.1f}')
        #val_logger.info(f'strategy top event F1 & segement F1: {metrics_F1_top}')
        val_logger.info(f'strategy beam event F1 & segement F1: {metrics_F1_greedy}')
        #val_logger.info(f'similarity: {metrics_sim:.3f}')
        return metrics_F1_greedy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="settings/pretrain.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-n", "--exp_name", default="Pretrain", type=str,
                        help="name of this experiment.")
    parser.add_argument('-l', '--lr', default=1e-05, type=float,
                        help='Learning rate.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed.')

    args = parser.parse_args()
    exp_name = args.exp_name

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["exp_name"] = args.exp_name
    config["seed"] = args.seed
    config["optim_args"]["lr"] = args.lr
    seed = config["seed"]
    setup_seed(seed)

    device = torch.device(config["device"])

    exp_name = exp_name + f"lr_{config['optim_args']['lr']}_seed_{seed}"

    model_output_dir, log_output_dir = set_logger(exp_name)
    main_logger = logger.bind(indent=1)

    dataloader = pretrain_dataloader(config,
                                     is_distributed=False,
                                     num_tasks=1,
                                     global_rank=0)

    if "bart" in config["text_decoder_args"]["name"]:
        model = BartCaptionModel(config)
    elif "bert" in config["text_decoder_args"]["name"]:
        model = BertCaptionModel(config)
    main_logger.info(f"Decoder model:{config['text_decoder_args']['name']}")
    model = model.to(device)

    # setup optim utils
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              weight_decay=config["optim_args"]["weight_decay"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])

    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')
    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')

    # load evaluation datamodule
    dcase_datamodule = AudioCaptionDataModule(config, "DCASE")
    dcase_val_loader = dcase_datamodule.val_dataloader()

    loss_stats = []
    dcase_eventF1 = []

    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')
        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)
        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')
        # evaluate on DCASE
        if epoch >= 10:
            main_logger.info('Evaluating on DCASE...')
            for i in range(1, 2):
                metrics_F1 = validate(dcase_val_loader,
                                        model,
                                        device=device,
                                        log_dir=log_output_dir,
                                        epoch=epoch,
                                        beam_size=i)
                event_F1 = metrics_F1[0]
                dcase_eventF1.append(event_F1)
                if event_F1 >= max(dcase_eventF1):
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": i,
                        "epoch": epoch,
                        "config": config,
                    }, str(model_output_dir) + '/dcase_best_model.pt')

    main_logger.info('Training done.')

if __name__ == '__main__':
    main()
