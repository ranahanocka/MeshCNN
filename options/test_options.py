from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
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
        self.is_train = False
