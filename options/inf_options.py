from argparse import Namespace

from options.base_options import BaseOptions


class InferenceOptions(BaseOptions):
    def __init__(self, opt_path):
        super(InferenceOptions, self).__init__()
        self.opt_path = opt_path
        self.opt = Namespace()
        self.initialized = False
        self.is_train = False
        self.parser = None

    def initialize(self):
        for line in open(self.opt_path, "r"):
            if line.startswith("----"):
                continue
            key, value = line.strip().split(":")
            setattr(self.opt, key.strip(), value.strip())
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt.is_train = self.is_train
