from abc import abstractmethod, ABC

import numpy as np
import torch
import torch.nn as nn


class AbstractPointEncoding(nn.Module, ABC):
    def __init__(self, opt):
        super(AbstractPointEncoding, self).__init__()
        self.opt = opt

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class PositionalEncoding3D(AbstractPointEncoding):

    id = "positional_encoding_3d"

    def __init__(self, opt):
        self.max_freq_log2 = opt.max_freq_log2 if hasattr(opt, "max_freq_log2") else 5
        self.num_freqs = opt.num_freqs if hasattr(opt, "num_freqs") else 2
        self.log_sampling = opt.log_sampling if hasattr(opt, "log_sampling") else True
        super(PositionalEncoding3D, self).__init__()

        self.embed_fns = []
        out_dim = 0

        for dim in range(3):  # 3D coordinates
            freq_bands = (
                torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** self.max_freq_log2,
                    self.num_freqs,
                    dtype=torch.float64,
                )
                if not self.max_freq_log2
                else 2.0 ** torch.linspace(0.0, self.max_freq_log2, self.num_freqs)
            )

            for freq in freq_bands:
                self.embed_fns.append(lambda x: torch.sin(x[:, dim] * freq))
                self.embed_fns.append(lambda x: torch.cos(x[:, dim] * freq))
                out_dim += 2

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(x) for fn in self.embed_fns], -1)


class NoPointEncoding(AbstractPointEncoding):
    id = "no_encode"

    def __init__(self, opt):
        super(NoPointEncoding, self).__init__(opt)

    def forward(self, x):
        # return torch.unsqueeze(x, 0) do this for array using numpy not torch
        return np.expand_dims(x, 0)


def point_encoder_fabric(opt) -> AbstractPointEncoding:
    if opt.point_encode == PositionalEncoding3D.id:
        return PositionalEncoding3D(opt)
    elif opt.point_encode == NoPointEncoding.id:
        return NoPointEncoding(opt)
    else:
        raise ValueError("Unknown point encoder id: {}".format(id))


# Example usage:
# positional_encoding = PositionalEncoding3D(max_freq_log2=5, num_freqs=10, log_sampling=True)

# Assuming your point tensor has shape (batch_size, 3) for x, y, z coordinates
# point_coordinates = torch.randn((64, 3))

# Apply positional encoding
# positional_encoded_point = positional_encoding(point_coordinates)
