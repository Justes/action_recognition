import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument(
        "--root", type=str, default="../datasets", help="root path to data directory"
    )
    parser.add_argument(
        "-s",
        "--source-names",
        type=str,
        required=True,
        nargs="+",
        help="source dataset for training(delimited by space)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        help="number of data loading workers (tips: 4 or 8 times number of gpus)",
    )
    parser.add_argument(
        "-n",
        "--num-classes",
        default=101,
        type=int,
        help="number of class)",
    )
    parser.add_argument("--height", type=int, default=128, help="height of an image")
    parser.add_argument("--width", type=int, default=256, help="width of an image")
    parser.add_argument("--frame", type=int, default=16, help="frame")


    # ************************************************************
    # Data augmentation
    # ************************************************************
    parser.add_argument(
        "--random-erase",
        action="store_true",
        help="use random erasing for data augmentation",
    )
    parser.add_argument(
        "--color-jitter",
        action="store_true",
        help="randomly change the brightness, contrast and saturation",
    )
    parser.add_argument(
        "--color-aug",
        action="store_true",
        help="randomly alter the intensities of RGB channels",
    )
    parser.add_argument(
        "--vertical-flip",
        action="store_true",
        help="randomly vertical flip",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="randomly rotation degree",
    )

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="optimization algorithm (see optimizers.py)",
    )
    parser.add_argument(
        "--lr", default=0.001, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--weight-decay", default=5e-04, type=float, help="weight decay"
    )
    # sgd
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum factor for sgd and rmsprop",
    )
    # rmsprop
    parser.add_argument(
        "--rmsprop-alpha", default=0.99, type=float, help="rmsprop's smoothing constant"
    )
    # adam/amsgrad
    parser.add_argument(
        "--adam-beta1",
        default=0.9,
        type=float,
        help="exponential decay rate for adam's first moment",
    )
    parser.add_argument(
        "--adam-beta2",
        default=0.999,
        type=float,
        help="exponential decay rate for adam's second moment",
    )

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument(
        "--max-epoch", default=100, type=int, help="maximum epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        help="manual epoch number (useful when restart)",
    )

    parser.add_argument(
        "--train-batch-size", default=32, type=int, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", default=32, type=int, help="test batch size"
    )
    parser.add_argument(
        "--dropout", default=0, type=float, help="drop out probability"
    )

    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="multi_step",
        help="learning rate scheduler (see lr_schedulers.py)",
    )
    parser.add_argument(
        "--stepsize",
        default=10,
        nargs="+",
        type=int,
        help="stepsize to decay learning rate",
    )
    parser.add_argument("--gamma", default=0.1, type=float, help="learning rate decay")
    parser.add_argument("--warmup", default=5, type=float, help="warm up epoch")
    parser.add_argument("--training-step", default=100, type=float, help="training step")
    parser.add_argument("--sche", default="", type=str, help="scheduler")


    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument("-a", "--arch", type=str, default="c3d")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="do not load pretrained weights"
    )
    parser.add_argument(
        "--pretrained-model", type=str, default="", help="load pretrained model"
    )

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument(
        "--load-weights",
        type=str,
        default="",
        help="load pretrained weights but ignore layers that don't match in size",
    )
    parser.add_argument("--evaluate", action="store_true", help="evaluate only")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=-1,
        help="evaluation frequency (set to -1 to test only in the end)",
    )
    parser.add_argument(
        "--start-eval",
        type=int,
        default=0,
        help="start to evaluate after a specific epoch",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=800,
        help="test-size for vehicleID dataset, choices=[800,1600,2400]",
    )
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--seed", type=int, default=1, help="manual seed")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        metavar="PATH",
        help="resume from a checkpoint",
    )
    parser.add_argument(
        "--save-dir", type=str, default="log", help="path to save log and model weights"
    )
    parser.add_argument("--use-cpu", action="store_true", help="use cpu")

    return parser


def dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        "source_names": parsed_args.source_names,
        "target_names": parsed_args.target_names,
        "root": parsed_args.root,
        "height": parsed_args.height,
        "width": parsed_args.width,
        "train_batch_size": parsed_args.train_batch_size,
        "test_batch_size": parsed_args.test_batch_size,
        "workers": parsed_args.workers,
        "random_erase": parsed_args.random_erase,
        "color_jitter": parsed_args.color_jitter,
        "color_aug": parsed_args.color_aug,
        "vertical_flip": parsed_args.vertical_flip,
        "rotation": parsed_args.rotation,
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        "optim": parsed_args.optim,
        "lr": parsed_args.lr,
        "weight_decay": parsed_args.weight_decay,
        "momentum": parsed_args.momentum,
        "rmsprop_alpha": parsed_args.rmsprop_alpha,
        "adam_beta1": parsed_args.adam_beta1,
        "adam_beta2": parsed_args.adam_beta2,
        "dropout": parsed_args.dropout
    }


def lr_scheduler_kwargs(parsed_args):
    """
    Build kwargs for lr_scheduler in lr_schedulers.py from
    the parsed command-line arguments.
    """
    return {
        "lr_scheduler": parsed_args.lr_scheduler,
        "stepsize": parsed_args.stepsize,
        "gamma": parsed_args.gamma,
    }
