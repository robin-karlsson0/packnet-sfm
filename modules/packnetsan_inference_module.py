import argparse

import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from packnet_sfm.networks.depth.PackNetSAN01 import PackNetSAN01
from packnet_sfm.utils.config import parse_test_file

RES_FACTOR = 32


class PackNetSANWrapper():
    """
    How to use:

        # 1. Initialize model
        ckpt_file = PATH-TO-CHECKPOINT-FILE
        model = PackNetSANWrapper(ckpt_file)

        # 1. Load input data
        img = Image.open(path)
        sparse_depth_map = np.load(path) <-- torch.Tensor

        # 2. Inference
        with torch.no_grad():
            inv_depths = model.predict(rgb, input_depth)

    NOTE: PackNet uses linear normalizer [0, 255] --> [0., 1.] for images.

    """
    def __init__(self, ckpt_file: str, use_cuda: bool = True):
        """
        Args:
            ckpt_file: Path to PackNetSAN checkpoint file.
        """
        config, state_dict = parse_test_file(ckpt_file, None)

        # Modify keys in state_dict for compatibility
        state_dict_mod = {}
        for key in state_dict.keys():
            val = state_dict[key]
            new_key = key
            new_key = new_key.replace('model.depth_net.', '')
            new_key = new_key.replace('conv3.0.weight', 'conv3.weight')
            new_key = new_key.replace('conv3.0.bias', 'conv3.bias')
            state_dict_mod[new_key] = val

        # Initialize model
        ver = config.model.depth_net.version
        self.model = PackNetSAN01(version=ver)
        self.model.training = False
        self.model.load_state_dict(state_dict_mod)

        self.model.eval()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        # NOTE config.datasets.augmentation.image_shape
        self.input_shape = (352, 1216)

    def predict(self,
                rgb: Image,
                input_depth: torch.Tensor = None,
                with_flip: bool = True) -> torch.Tensor:
        """
        Args:
            rgb: Un-normalized RGB PIL.Image
            input_depth:
            with_flip: Add flipped version of input to improve prediction acc.
        """
        H, W = rgb.size

        if H % RES_FACTOR != 0 and W % RES_FACTOR != 0:
            raise ValueError(f"Image size must be divisible by {RES_FACTOR}")

        # rgb = self.resize_image(rgb)
        # input_depth = self.resize_image(input_depth, Image.NONE)
        rgb = self.to_tensor(rgb)
        rgb = rgb.unsqueeze(0)

        if self.use_cuda:
            rgb = rgb.cuda()
            if input_depth is not None:
                input_depth = input_depth.cuda()

        # Takes first element from returned list --> (B,C,H,W) tensor
        inv_depth = self.model(rgb, input_depth)['inv_depths'][0]

        if with_flip:
            rgb = self.flip_lr(rgb)
            if input_depth is not None:
                input_depth = self.flip_lr(input_depth)
            inv_depth_flipped = self.model(rgb, input_depth)['inv_depths'][0]
            inv_depth = self.post_process_inv_depth(inv_depth,
                                                    inv_depth_flipped,
                                                    method='mean')
        depth = self.inv2depth(inv_depth)

        return depth

    def resize_image(self,
                     image: Image,
                     interpolation=Image.ANTIALIAS) -> Image:
        """Resizes input image."""
        transform = transforms.Resize(self.input_shape,
                                      interpolation=interpolation)
        return transform(image)

    @staticmethod
    def to_tensor(image: Image,
                  tensor_type='torch.FloatTensor') -> torch.Tensor:
        """Casts an image to a torch.Tensor."""
        transform = transforms.ToTensor()
        return transform(image).type(tensor_type)

    @staticmethod
    def flip_lr(image: torch.Tensor) -> torch.Tensor:
        """Flip image horizontally."""
        assert image.dim(
        ) == 4, 'You need to provide a [B,C,H,W] image to flip'
        return torch.flip(image, [3])

    @staticmethod
    def is_tuple(data):
        """Checks if data is a tuple."""
        return isinstance(data, tuple)

    @staticmethod
    def is_list(data):
        """Checks if data is a list."""
        return isinstance(data, list)

    def is_seq(self, data):
        """Checks if data is a list or tuple."""
        return self.is_tuple(data) or self.is_list(data)

    def inv2depth(self, inv_depth):
        """Invert an inverse depth map to produce a depth map.

        Args:
            inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W].

        Returns:
            depth : torch.Tensor or list of torch.Tensor [B,1,H,W].
        """
        if self.is_seq(inv_depth):
            return [self.inv2depth(item) for item in inv_depth]
        else:
            return 1. / inv_depth.clamp(min=1e-6)

    def fuse_inv_depth(self,
                       inv_depth: torch.Tensor,
                       inv_depth_hat: torch.Tensor,
                       method: str = 'mean') -> torch.Tensor:
        """Fuse inverse depth and flipped inverse depth maps.

        Args:
            inv_depth : Inverse depth map [B,1,H,W].
            inv_depth_hat : Flipped inverse depth map produced from a flipped
                            image [B,1,H,W].
            method : Method that will be used to fuse the inverse depth maps.

        Returns
            fused_inv_depth : Fused inverse depth map [B,1,H,W].
        """
        if method == 'mean':
            return 0.5 * (inv_depth + inv_depth_hat)
        elif method == 'max':
            return torch.max(inv_depth, inv_depth_hat)
        elif method == 'min':
            return torch.min(inv_depth, inv_depth_hat)
        else:
            raise ValueError('Unknown post-process method {}'.format(method))

    def post_process_inv_depth(self,
                               inv_depth: torch.Tensor,
                               inv_depth_flipped: torch.Tensor,
                               method: str = 'mean') -> torch.Tensor:
        """ Post-process an inverse and flipped inverse depth map.

        Args:
            inv_depth : Inverse depth map [B,1,H,W].
            inv_depth_flipped : torch.Tensor [B,1,H,W].
            Inverse depth map produced from a flipped image.
            method : Method that will be used to fuse the inverse depth maps.

        Returns:
            inv_depth_pp : Post-processed inverse depth map [B,1,H,W]
        """
        B, C, H, W = inv_depth.shape
        inv_depth_hat = self.flip_lr(inv_depth_flipped)
        inv_depth_fused = self.fuse_inv_depth(inv_depth,
                                              inv_depth_hat,
                                              method=method)
        xs = torch.linspace(0.,
                            1.,
                            W,
                            device=inv_depth.device,
                            dtype=inv_depth.dtype).repeat(B, C, H, 1)
        mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
        mask_hat = self.flip_lr(mask)
        return mask_hat * inv_depth + mask * inv_depth_hat + (
            1.0 - mask - mask_hat) * inv_depth_fused


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(
        description='PackNet-SfM evaluation script')
    parser.add_argument('checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('img_path', type=str, help='Image file (.png)')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    return args


if __name__ == '__main__':

    args = parse_args()

    import matplotlib.pyplot as plt

    model = PackNetSANWrapper(args.checkpoint)

    img = Image.open(args.img_path)
    img = img.resize((640, 405))

    H, W = img.size
    H_crop = H - H % RES_FACTOR
    W_crop = W - W % RES_FACTOR

    img = img.crop((0, 0, H_crop, W_crop))

    with torch.no_grad():
        depth = model.predict(img, None, with_flip=True)

    plt.imshow(depth[0, 0].cpu().numpy())
    plt.show()
