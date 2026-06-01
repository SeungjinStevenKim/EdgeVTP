#!/usr/bin/env python3
"""Run EdgeVTP trajectory prediction on a real video."""

from utils.video.cli import load_config, parse_args
from utils.video.pipeline import VideoPredictor


def main():
    args = parse_args()
    config = load_config(args.config)
    VideoPredictor(args, config).run()


if __name__ == "__main__":
    main()
