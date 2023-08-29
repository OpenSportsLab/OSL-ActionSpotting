from snspotting.apis import init_spotter, inference_spotter
# import mmcv

config_file = 'configs/learnablepooling/soccernet_netvlad++_resnetpca512.py'
checkpoint_file = 'output/learnablepooling/soccernet_netvlad++_resnetpca512/model.pth.tar'

# build the model from a config file and a checkpoint file
model = init_spotter(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
game = "/home/giancos/git/sn-spotting-pip/path/to/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
results = inference_spotter(config_file, model, input=game)
print(f"Found {len(results['predictions'])} actions!")

feature = "/home/giancos/git/sn-spotting-pip/path/to/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_ResNET_TF2_PCA512.npy"
results = inference_spotter(config_file, model, input=feature)
print(f"Found {len(results['predictions'])} actions!")

feature = "/home/giancos/git/sn-spotting-pip/path/to/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/2_ResNET_TF2_PCA512.npy"
results = inference_spotter(config_file, model, input=feature)
print(f"Found {len(results['predictions'])} actions!")


# # visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)







# import os
# import logging
# from datetime import datetime
# import time
# import numpy as np
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# import torch
# import mmengine
# from mmengine.config import Config, DictAction


# import json


# from snspotting.datasets import build_dataset, build_dataloader
# from snspotting.models import build_model
# from snspotting.loss import build_criterion
# from snspotting.core import build_optimizer, build_scheduler

# from snspotting.core.inference import infer_features, infer_game, infer_video
# from snspotting.core.evaluation import evaluate_Spotting #testClassication, testSpotting


# def parse_args():

#     parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
#     parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
#     parser.add_argument("--game", type=str, help="path of game")
#     parser.add_argument("--video", type=str, help="path of video")
#     parser.add_argument("--features", type=str, help="path of features")

#     parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the results")
#     parser.add_argument("--confidence_threshold", type=float, default=0.5, help="confidence threshold for results")

#     # not that important
#     parser.add_argument("--seed", type=int, default=42, help="random seed")
#     # parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
#     # parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
#     # parser.add_argument("--ema", action="store_true", help="whether to use model EMA")
#     # parser.add_argument("--wandb", action="store_true", help="whether to use wandb to log everything")
#     # parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
#     # parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
#     # parser.add_argument("--static_graph", action="store_true", help="set static_graph==True in DDP")
#     parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")

#     # # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
#     parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

#     # read args
#     args = parser.parse_args()
#     return args


# def main():

#     args = parse_args()

#     # Read Config
#     cfg = Config.fromfile(args.config)
    
#     # overwrite cfg from args
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
    
#     # for reproducibility
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)

#     # Create Work directory
#     os.makedirs(cfg.work_dir, exist_ok=True)

#     # Define logging
#     numeric_level = getattr(logging, args.loglevel.upper(), None)
#     if not isinstance(numeric_level, int):
#         raise ValueError('Invalid log level: %s' % args.loglevel)

#     # Define output folder
#     log_path = os.path.join(cfg.work_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
#     logging.basicConfig(
#         level=numeric_level,
#         format=
#         "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
#         handlers=[
#             logging.FileHandler(log_path),
#             logging.StreamHandler()
#         ])

#     # define GPUs
#     if cfg.training.GPU >= 0:
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.GPU)

#     # Dump configuration file
#     cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
#     logging.info(cfg)

#     # Start Timing
#     start=time.time()
#     logging.info('Starting main function')


#     # Ensure weights are not None
#     if cfg.model.load_weights is None:
#         cfg.model.load_weights = os.path.join(cfg.work_dir, "model.pth.tar")
    
#     # Build Model
#     model = build_model(cfg.model).cuda()


#     # Infer features
#     if args.features is not None:
#         output_file = os.path.join(os.path.dirname(args.features), "results_spotting.json")
        
#         # Prevent overwriting existing results
#         if os.path.exists(output_file) and not args.overwrite:           
#             logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
#         else:
#             json_results = infer_features(cfg, args.features, model, 
#                 confidence_threshold=args.confidence_threshold, 
#                 overwrite=args.overwrite)            
#             with open(output_file, 'w') as f:
#                 json.dump(json_results, f, indent=4)
        
        
            

#     # Infer Game folder
#     if args.game is not None:
#         output_file = os.path.join(args.game, "results_spotting.json")
        
#         # Prevent overwriting existing results
#         if os.path.exists(output_file) and not args.overwrite:           
#             logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
#         else:
#             json_results = infer_game(cfg, args.game, model, 
#                 confidence_threshold=args.confidence_threshold, 
#                 overwrite=args.overwrite)
#             with open(output_file, 'w') as f:
#                 json.dump(json_results, f, indent=4)
        
        

#     logging.info(f'Total Execution Time is {time.time()-start} seconds')

#     return 


# if __name__ == '__main__':
#     main()