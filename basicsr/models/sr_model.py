import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
# 【NICOLE 2026】
        # if self.cri_pix is None and self.cri_perceptual is None:
        #     raise ValueError('Both pixel and perceptual losses are None.')
        if train_opt.get('face_region_opt'):
            self.cri_face_region = build_loss(train_opt['face_region_opt']).to(self.device)
        else:
            self.cri_face_region = None

        # 【NICOLE 2026】
        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None
        # 【NICOLE 2026】

        # 【NICOLE2026】频域损失（全局 FocalFrequencyLoss 或 FaceRegionFocalFrequencyLoss，由 yaml type 决定）
        if train_opt.get('freq_opt'):
            self.cri_freq = build_loss(train_opt['freq_opt']).to(self.device)
        else:
            self.cri_freq = None
        # 【NICOLE2026】

        # 【NICOLE2026】原检查（已注释，新检查加入 cri_freq）
        # if self.cri_pix is None and self.cri_perceptual is None and self.cri_face_region is None and self.cri_edge is None:
        #     raise ValueError('Pixel, perceptual, face-region and edge losses are all None.')
        if (self.cri_pix is None and self.cri_perceptual is None and self.cri_face_region is None
                and self.cri_edge is None and self.cri_freq is None):
            raise ValueError('Pixel, perceptual, face-region, edge and frequency losses are all None.')
        # 【NICOLE2026】
# 【NICOLE 2026】

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
# 【NICOLE 2026】
        if 'face_weight' in data:
            self.face_weight = data['face_weight'].to(self.device)
        elif hasattr(self, 'face_weight'):
            del self.face_weight
# 【NICOLE 2026】


    def optimize_parameters(self, current_iter):
        # 【NICOLE2026】梯度累积：train.accum_iter > 1 时每 accum_iter 步才执行一次
        # optimizer.step()，等效 batch = batch_size_per_gpu * accum_iter，显存不增加。
        # 注意：固定 total_iter 下，优化器更新次数变为 total_iter / accum_iter。
        # 原实现（已注释）：
        # self.optimizer_g.zero_grad()
        accum_iter = max(int(self.opt['train'].get('accum_iter', 1)), 1)
        if (current_iter - 1) % accum_iter == 0:  # 累积窗口的第一步清零梯度
            self.optimizer_g.zero_grad()
        # 【NICOLE2026】
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
# 【NICOLE 2026】
        # face-region weighted reconstruction loss
        if self.cri_face_region:
            if not hasattr(self, 'face_weight'):
                raise ValueError(
                    'face_region_opt is enabled, but the current batch does not contain face_weight. '
                    'Please set dataroot_face_weight in the training dataset config.')
            l_face = self.cri_face_region(self.output, self.gt, self.face_weight)
            l_total += l_face
            loss_dict['l_face'] = l_face
        # 【NICOLE 2026】
        # face-region weighted RGB Sobel edge loss
        if self.cri_edge:
            if not hasattr(self, 'face_weight'):
                raise ValueError(
                    'edge_opt is enabled, but the current batch does not contain face_weight. '
                    'Please set dataroot_face_weight in the training dataset config.')
            l_edge = self.cri_edge(self.output, self.gt, self.face_weight)
            l_total += l_edge
            loss_dict['l_edge'] = l_edge
        # 【NICOLE 2026】
        # 【NICOLE2026】频域损失：全局 FFL 忽略 face_weight，face-region FFL 需要 face_weight
        if self.cri_freq:
            l_freq = self.cri_freq(self.output, self.gt, face_weight=getattr(self, 'face_weight', None))
            l_total += l_freq
            loss_dict['l_freq'] = l_freq
        # 【NICOLE2026】
# 【NICOLE 2026】
        loss_dict['l_total'] = l_total
        # 【NICOLE2026】梯度累积：损失除以 accum_iter 使累积梯度等于等效 batch 的均值；
        # 仅在累积窗口最后一步执行 optimizer.step() 与 EMA 更新。日志记录的是未缩放损失。
        # 原实现（已注释）：
        # l_total.backward()
        # self.optimizer_g.step()
        (l_total / accum_iter).backward()
        do_step = (current_iter % accum_iter == 0)
        if do_step:
            self.optimizer_g.step()
        # 【NICOLE2026】

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 【NICOLE2026】原实现（已注释）：EMA 只应在参数实际更新后执行
        # if self.ema_decay > 0:
        #     self.model_ema(decay=self.ema_decay)
        if self.ema_decay > 0 and do_step:
            self.model_ema(decay=self.ema_decay)
        # 【NICOLE2026】

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        # 【NICOLE 2026】
        face_metric_types = {'calculate_face_psnr', 'calculate_face_ssim'}
        need_face_weight_metric = False
        if with_metrics:
            need_face_weight_metric = any(
                metric_opt.get('type') in face_metric_types for metric_opt in self.opt['val']['metrics'].values())
        # 【NICOLE 2026】

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt
            # 【NICOLE 2026】
            if 'face_weight' in val_data:
                face_weight = val_data['face_weight'][0].detach().float().cpu().numpy()
                if face_weight.ndim == 3 and face_weight.shape[0] == 1:
                    face_weight = face_weight[0]
                metric_data['face_weight'] = face_weight
            elif need_face_weight_metric:
                raise ValueError(
                    'Face-region validation metrics are enabled, but the validation batch does not contain '
                    'face_weight. Please set dataroot_face_weight in the validation dataset config.')
            else:
                metric_data.pop('face_weight', None)
            # 【NICOLE 2026】

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
