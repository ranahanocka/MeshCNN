from . import base_options
from .base_options import time_s


class TestOptions(base_options.BaseOptions):
    def initialize(self):
        base_options.BaseOptions.initialize(self)
        self.parser.add_argument(
            "--results_dir", type=str, default="./results/", help="saves results here."
        )
        self.parser.add_argument(
            "--phase", type=str, default="test", help="train, val, test, etc"
        )  # todo delete.
        self.parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use latest cached model",
        )
        self.parser.add_argument(
            "--num_aug", type=int, default=1, help="# of augmentation files"
        )
        self.parser.add_argument(
            "--timestamp", type=str, default=time_s, help="model id to load if set"
        )
