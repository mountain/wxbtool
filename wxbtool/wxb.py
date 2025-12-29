from arghandler import ArgumentHandler, subcmd

from wxbtool.data.download import main as dnmain
from wxbtool.data.dsserver import main as dsmain
from wxbtool.data.aggregator import main as damain
from wxbtool.core.phase.eval import main as eval_main
from wxbtool.core.phase.infer import main as infer_main
from wxbtool.core.phase.infer import main_gan as inferg_main
from wxbtool.core.phase.test import main as ttmain
from wxbtool.core.phase.train import main as tnmain
from wxbtool.core.config import add_device_arguments


@subcmd
def help(parser, context, args):
    pass


@subcmd("train", help="start training")
def train(parser, context, args):
    add_device_arguments(parser)
    parser.add_argument("-g", "--gpu", type=str, default="", help="indexes of gpu")
    parser.add_argument(
        "-c",
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=0,
        help="current epoch to start training from",
    )
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        default="wxbtool.zoo.unet.t850d3",
        help="module of the metrological model to load",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default="",
        help="dump file of the metrological model to load",
    )
    parser.add_argument(
        "-k", "--check", type=str, default="", help="checkpoint file to load"
    )
    parser.add_argument("-r", "--rate", type=str, default="0.001", help="learning rate")
    parser.add_argument(
        "-w", "--weightdecay", type=float, default=0.0, help="weight decay"
    )
    parser.add_argument(
        "-R",
        "--ratio",
        type=str,
        default="10",
        help="the ratio of the two learning rates between generator and discriminator in GAN",
    )
    parser.add_argument(
        "-A",
        "--alpha",
        type=float,
        default=0.1,
        help="a 0 to 1 weight to control the loss calculation in GAN",
    )
    parser.add_argument(
        "-B", "--balance", type=float, default=0.9, help="exit balance for GAN training"
    )
    parser.add_argument(
        "-T",
        "--tolerance",
        type=float,
        default=0.05,
        help="exit balance tolerance for GAN training",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="",
        help="http url of the dataset server or binding unix socket (unix:/path/to/your.sock)",
    )
    parser.add_argument(
        "-G",
        "--gan",
        type=str,
        default="false",
        help="training GAN or not, default is false",
    )
    parser.add_argument(
        "-t", "--test", type=str, default="false", help="setting for test"
    )
    parser.add_argument(
        "-O", "--optimize", action="store_true", help="use optimized training for CI"
    )
    parser.add_argument(
        "-p", "--plot", type=str, default="false", help="plot training result"
    )

    opt = parser.parse_args(args)

    tnmain(context, opt)


@subcmd("test", help="start testing")
def test(parser, context, args):
    add_device_arguments(parser)
    parser.add_argument("-g", "--gpu", type=str, default="0", help="index of gpu")
    parser.add_argument(
        "-c",
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        default="wxbtool.zoo.unet.t850d3",
        help="module of the metrological model to load",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default="",
        help="dump file of the metrological model to load",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="",
        help="http url of the dataset server or binding unix socket (unix:/path/to/your.sock)",
    )
    parser.add_argument(
        "-G",
        "--gan",
        type=str,
        default="false",
        help="training GAN or not, default is false",
    )
    parser.add_argument(
        "-t", "--test", type=str, default="false", help="setting for test"
    )
    parser.add_argument(
        "-O", "--optimize", action="store_true", help="use optimized testing for CI"
    )
    parser.add_argument(
        "-p", "--plot", type=str, default="true", help="plot testing result"
    )
    opt = parser.parse_args(args)

    ttmain(context, opt)


@subcmd("forecast", help="forecast (deterministic or GAN ensemble)")
def forecast(parser, context, args):
    add_device_arguments(parser)
    parser.add_argument("-g", "--gpu", type=str, default="0", help="index of gpu")
    parser.add_argument(
        "-c",
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        default="wxbtool.zoo.unet.t850d3",
        help="module of the metrological model to load",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default="",
        help="dump file of the metrological model to load",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="",
        help="http url of the dataset server or binding unix socket (unix:/path/to/your.sock)",
    )
    parser.add_argument(
        "-t",
        "--datetime",
        type=str,
        required=True,
        help="datetime for forecast: use %Y-%m-%d for deterministic; use YYYY-MM-DDTHH:MM:SS for GAN",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output file format, either png or nc",
    )
    parser.add_argument(
        "-G",
        "--gan",
        type=str,
        default="false",
        help="set to true to run GAN ensemble mode",
    )
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=0,
        help="number of samples for GAN ensemble (required when --gan true)",
    )
    opt = parser.parse_args(args)

    # Route to deterministic or GAN forecast
    gan_enabled = str(getattr(opt, "gan", "false")).lower() in {"1", "true", "yes", "y"}
    if gan_enabled or getattr(opt, "samples", 0) > 0:
        if getattr(opt, "samples", 0) <= 0:
            raise SystemExit("error: --samples must be > 0 when --gan true")
        inferg_main(context, opt)
    else:
        infer_main(context, opt)


@subcmd("backtest", help="Backtesting model performance")
def backtest(parser, context, args):
    # Mirror arguments from eval
    add_device_arguments(parser)
    parser.add_argument("-g", "--gpu", type=str, default="0", help="index of gpu")
    parser.add_argument(
        "-c",
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        default="wxbtool.zoo.unet.t850d3",
        help="module of the metrological model to load",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default="",
        help="dump file of the metrological model to load",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="",
        help="http url of the dataset server or binding unix socket (unix:/path/to/your.sock)",
    )
    parser.add_argument(
        "-t",
        "--datetime",
        type=str,
        required=True,
        help="specific datetime for backtesting in the format %Y-%m-%d, e.g. 2025-01-01",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output file format, either png or nc",
    )
    parser.add_argument(
        "-p", "--plot", type=str, default="true", help="plot testing result"
    )
    opt = parser.parse_args(args)

    eval_main(context, opt)


@subcmd("data-serve", help="start the dataset server")
def data_serve(parser, context, args):
    parser.add_argument(
        "-b",
        "--bind",
        type=str,
        default=None,
        help="binding address (ip:port or unix:/path/to/your.sock)",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8088, help="the port of the dataset server"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4, help="the number of workers"
    )
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        default="wxbtool.specs.res5_625.t850weyn",
        help="module of a metrological model to load",
    )
    parser.add_argument(
        "-s",
        "--setting",
        type=str,
        default="Setting",
        help="setting for a metrological model spec",
    )
    parser.add_argument(
        "-t", "--test", type=str, default="false", help="setting for test"
    )
    opt = parser.parse_args(args)

    dsmain(context, opt)


@subcmd("data-download", help="download the latest hourly ERA5 data from ECMWF")
def data_download(parser, context, args):
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        default="wxbtool.zoo.unet.t850d3",
        help="module of the metrological model to load",
    )
    parser.add_argument(
        "-G",
        "--gan",
        type=str,
        default="false",
        help="model is GAN or not, default is false",
    )
    parser.add_argument(
        "--coverage",
        type=str,
        default="weekly",
        help="period of data coverage: daily, weekly, monthly or an integer in days",
    )
    opt = parser.parse_args(args)

    dnmain(context, opt)


@subcmd("data-agg", help="aggregate high-frequency data to lower-frequency")
def data_agg(parser, context, args):
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        default="wxbtool.specs.res5_625.t850weyn",
        help="module of the metrological model to load (for Setting)",
    )
    parser.add_argument(
        "-s",
        "--setting",
        type=str,
        default="Setting",
        help="name of the setting class defining the DESTINATION layout",
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="source data root directory",
    )
    parser.add_argument(
        "--window",
        type=int,
        required=True,
        help="window size in hours",
    )
    parser.add_argument(
        "--align",
        type=str,
        default="backward",
        choices=["backward", "forward", "center"],
        help="window alignment",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="number of worker processes",
    )
    parser.add_argument(
        "--lat",
        type=int,
        default=32,
        help="target latitude resolution (32 or 33)",
    )
    parser.add_argument(
        "--lon",
        type=str,
        default="0-360",
        choices=["0-360", "-180-180"],
        help="target longitude convention",
    )
    opt = parser.parse_args(args)

    damain(context, opt)


def main():
    import sys
    import os

    sys.path.insert(0, os.getcwd())

    handler = ArgumentHandler()
    if len(sys.argv) < 2:
        handler.run(["help"])
    else:
        handler.run(sys.argv[1:])


if __name__ == "__main__":
    main()
