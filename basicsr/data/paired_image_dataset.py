from os import path as osp

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
    【 NICOLE 2026 】
            dataroot_face_weight (str, optional): Data root path for offline face weight maps.
    【 NICOLE 2026 】
                The weight map should have the same relative path as the corresponding GT image.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.io_backend_type = self.io_backend_opt['type']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
# 【 NICOLE 2026 】
        self.face_weight_folder = opt.get('dataroot_face_weight', None)
        if self.face_weight_folder is not None and self.io_backend_type != 'disk':
            raise ValueError('dataroot_face_weight currently supports only disk io_backend.')
#【 NICOLE 2026 】
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        # ？？
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl, self.task)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;

        img_face_weight = None
        face_weight_path = None

        if self.task == 'CAR':
            # image range: [0, 255], int., H W 1

            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.
    
        elif self.task == 'denoising_gray': # Matlab + OpenCV version
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            # OpenCV version, following "Deep Convolutional Dictionary Learning for Image Denoising"
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            # # Matlab version (using this version may have 0.6dB improvement, which is unfair for comparison)
            # img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
            # if img_gt.ndim != 2:
            #     img_gt = rgb2ycbcr(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB), y_only=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise/255., img_gt.shape)
            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)

        elif self.task == 'denoising_color':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise/255., img_gt.shape)

        else:
            # image range: [0, 1], float32., H W 3
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
#  【 NICOLE 2026】
        if self.face_weight_folder is not None:
            face_weight_path = self._get_face_weight_path(gt_path)
            img_bytes = self.file_client.get(face_weight_path, 'face_weight')
            img_face_weight = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_face_weight = np.expand_dims(img_face_weight, axis=2).astype(np.float32)
            if img_face_weight.shape[0:2] != img_gt.shape[0:2]:
                raise ValueError(
                    f'Face weight map shape {img_face_weight.shape[0:2]} does not match GT shape '
                    f'{img_gt.shape[0:2]} for GT {gt_path}.')
#  【 NICOLE 2026】

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
#  【 NICOLE 2026】
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if img_face_weight is not None:
                img_gts, img_lq = paired_random_crop([img_gt, img_face_weight], img_lq, gt_size, scale, gt_path)
                img_gt, img_face_weight = img_gts
            else:
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation
            #img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            if img_face_weight is not None:
                img_gt, img_lq, img_face_weight = augment([img_gt, img_lq, img_face_weight], self.opt['use_hflip'],
                                                          self.opt['use_rot'])
            else:
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
#  【 NICOLE 2026】

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
            #  【 NICOLE 2026】
            if img_face_weight is not None:
                img_face_weight = img_face_weight[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
            #  【 NICOLE 2026】

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        #  【 NICOLE 2026】
        if img_face_weight is not None:
            img_face_weight = img2tensor(img_face_weight, bgr2rgb=False, float32=True)
        #  【 NICOLE 2026】
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        #return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}


    #  【 NICOLE 2026】
        results = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        if img_face_weight is not None:
            results['face_weight'] = img_face_weight
            results['face_weight_path'] = face_weight_path
        return results

    def _get_face_weight_path(self, gt_path):
        rel_gt_path = osp.relpath(gt_path, self.gt_folder)
        face_weight_path = osp.join(self.face_weight_folder, rel_gt_path)
        if not osp.isfile(face_weight_path):
            png_path = osp.splitext(face_weight_path)[0] + '.png'
            if osp.isfile(png_path):
                face_weight_path = png_path
            else:
                raise FileNotFoundError(
                    f'Face weight map not found. Expected {face_weight_path} or {png_path} for GT {gt_path}.')
        return face_weight_path

    #  【 NICOLE 2026】

    def __len__(self):
        return len(self.paths)
