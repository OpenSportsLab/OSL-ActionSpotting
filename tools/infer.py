import os 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from mmengine.config import Config, DictAction
from snspotting.apis import init_spotter, inference_spotter


def parse_args():

    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--input", type=str, help="path of game/features/video")
    parser.add_argument("--checkpoint", type=str, help="path of model checkpoint")

    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the results")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="confidence threshold for results")

    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")

    # read args
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # Read Config
    cfg = Config.fromfile(args.config)
    
    # overwrite cfg from args
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # define GPUs
    if cfg.training.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.GPU)

    # build the model from a config file and a checkpoint file
    model = init_spotter(cfg, args.checkpoint, device='cuda:0')

    # test a single image and show the results
    results = inference_spotter(cfg, model, 
                            input=args.input,
                            confidence_threshold=args.confidence_threshold)

    # print results
    print(f"Found {len(results['predictions'])} actions!")

    return 


if __name__ == '__main__':
    main()