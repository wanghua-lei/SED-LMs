# We train the model here.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
from pprint import PrettyPrinter
import torch
import platform
import ruamel.yaml as yaml
from loguru import logger
from data_handling.datamodule import AudioCaptionDataModule
from models.bart_captioning import BartCaptionModel
from models.bert_captioning import BertCaptionModel
from models.t5_captioning import T5CaptionModel
# from torch import distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
from pretrain import validate, train
from tools.optim_utils import get_optimizer, cosine_lr, step_lr
from tools.utils import setup_seed, set_logger


def main():
    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('--local-rank', type=int, help="local gpu id", default = -1)
    # parser.add_argument('-n', '--exp_name', default='groundcap', type=str,
    #                     help='Name of the experiment.')
    parser.add_argument('-c', '--config', default='settings/settings.yaml', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-l', '--lr', default=1e-04, type=float,
                        help='Learning rate.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed.')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed
    config["optim_args"]["lr"] = args.lr
    setup_seed(config["seed"])
    folder_name = '16k_urban'

    model_output_dir, log_output_dir = set_logger(folder_name)

    main_logger = logger.bind(indent=1)

    # set up model
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())
    main_logger.info(f'Process on {device_name}')

    #Distributed training
    # LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    # dist.init_process_group(backend='nccl', init_method='env://')
    # torch.cuda.set_device(args.local_rank)
    # global_rank = dist.get_rank()
    # num_tasks = dist.get_world_size()

    # data loading
    datamodule = AudioCaptionDataModule(config, config["data_args"]["dataset"])
    train_loader = datamodule.train_dataloader(is_distributed=False)
    #train_loader = datamodule.train_dataloader(is_distributed=True, global_rank=global_rank, num_tasks= num_tasks)
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    #set up model
    if "bart" in config["text_decoder_args"]["name"]:
        model = BartCaptionModel(config)
    elif "bert" in config["text_decoder_args"]["name"]:
        model = BertCaptionModel(config)
    elif "t5" in config["text_decoder_args"]["name"]:
        model = T5CaptionModel(config)
    model = model.to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)   
    #model = DDP(model, device_ids=[args.local_rank], output_device = args.local_rank, find_unused_parameters=True)
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'f'{printer.pformat(config)}')
    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')

    if config["pretrain"]:
        pretrain_checkpoint = torch.load(config["pretrain_path"])
        model.load_state_dict(pretrain_checkpoint["model"],False)
        
        main_logger.info(f"Loaded weights from {config['pretrain_path']}")

    # set up optimizer and loss
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              weight_decay=config["optim_args"]["weight_decay"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    # scheduler = None
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(train_loader),
                          steps=len(train_loader) * config["training"]["epochs"])
    
    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    # training loop
    loss_stats = []
    eventF1 = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        main_logger.info(f'Training for epoch [{epoch}]')
        # scheduler_warmup.step()
        train_statics = train(model, train_loader, optimizer, scheduler, device, epoch)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')
        if epoch >= 10:
        # validation loop, validation after each epoch
            main_logger.info("Validating...")
            for i in range(1, 3):
                metrics_F1 = validate(val_loader,
                                model,
                                device=device,
                                log_dir=log_output_dir,
                                epoch=epoch,
                                beam_size=i)
                
                event_F1 = metrics_F1[0]
                eventF1.append(event_F1)
                if event_F1 >= max(eventF1):
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": i,
                        "epoch": epoch,
                        "config": config,
                    }, str(model_output_dir) + '/best_model.pt'.format(epoch))

    #Training done, evaluate on evaluation set
    main_logger.info('Training done. Start evaluating.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pt')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint occurred in {best_epoch} th epoch.')
    for i in range(1, 3):
        metrics_F1 = validate(test_loader, model,
                          device=device,
                          log_dir=log_output_dir,
                          epoch=0,
                          beam_size=i,
                          eval= True
                          )
    main_logger.info('Evaluation done.')


if __name__ == '__main__':
    main()