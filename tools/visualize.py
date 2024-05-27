from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
from mmengine.config import Config, DictAction
from oslactionspotting.core.utils.io import check_config
from oslactionspotting.apis.visualize import build_visualizer


def parse_args():

    parser = ArgumentParser(
        description="Visualizer tool",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")

    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )

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

    # Define logging
    numeric_level = getattr(logging, cfg.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % cfg.log_level)

    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
    )
    # Check configs files
    logging.info("Checking configs files")
    check_config(cfg)

    logging.info(cfg)

    logging.info("Starting main function")

    logging.info("Build visualizer")

    visualizer = build_visualizer(cfg)

    logging.info("Start visualizing")

    visualizer.visualize()

    return


if __name__ == "__main__":
    main()
