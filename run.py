import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color
from recbole.model.general_recommender import LightGCN, NCL
from recbole.trainer import Trainer, NCLTrainer

#from models.lightgcn import LightGCN
#from models.ncl import NCL
from models.scl import SCL
#from trainers.lightgcn_trainer import LightGCNTrainer
#from trainers.ncl_trainer import NCLTrainer
from trainers.scl_trainer import SCLTrainer

import time
import json
import os

def run_single_model(args):
    if args.model == 'lightgcn':
        model_class = LightGCN
        trainer_class = Trainer
    elif args.model == 'ncl':
        model_class = NCL
        trainer_class = NCLTrainer
    elif args.model == 'scl':
        model_class = SCL
        trainer_class = SCLTrainer
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Config files
    args.config_file_list = ['configs/overall.yaml']

    if args.model == 'lightgcn':
        args.config_file_list.append('configs/lightgcn.yaml')
    elif args.model == 'ncl':
        args.config_file_list.append('configs/ncl.yaml')
    elif args.model == 'scl':
        args.config_file_list.append('configs/scl.yaml')
    
    args.config_file_list.append(f'configs/{args.dataset}.yaml')

    # optional extra config
    if args.config:
        args.config_file_list.append(args.config)
    
    config = Config(
        model=model_class,
        dataset=args.dataset,
        config_file_list=args.config_file_list)

    if args.seed is not None:
        config['seed'] = args.seed
    
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(set_color('trainable parameters', 'yellow') + f': {num_parameters}')

    # trainer loading and initialization
    trainer = trainer_class(config, model)

    # model training
    train_start_time = time.time()
    
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    train_end_time = time.time()
    training_time = train_end_time - train_start_time

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    os.makedirs('results', exist_ok=True)

    config_tag = "default"
    if args.exp_name:
        config_tag = args.exp_name
    elif args.config:
        config_tag = os.path.splitext(os.path.basename(args.config))[0]
    
    result_record = {
        'model': args.model,
        'config_tag': config_tag,
        'dataset': args.dataset,
        'seed': config['seed'],
        'training_time': training_time,
        'num_parameters': num_parameters,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

    result_path = f"results/{args.model}_{config_tag}_seed{config['seed']}.json"
    
    with open(result_path, 'w') as f:
        json.dump(result_record, f, indent=2)
    
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    logger.info(set_color('training time', 'yellow') + f': {training_time:.4f}s')
    logger.info(set_color('result file', 'yellow') + f': {result_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='scl', choices=['lightgcn', 'ncl', 'scl'], help='Model to run: lightgcn, ncl or scl')
    parser.add_argument('--dataset', type=str, default='travel')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name for result filename.')
    args, _ = parser.parse_known_args()

    run_single_model(args)
