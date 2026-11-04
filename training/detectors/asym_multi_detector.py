# -*- coding: gbk -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time

# ==========================================
# Core Shield: Monkey-patch Python's linecache
# ==========================================
import linecache
original_updatecache = linecache.updatecache
def safe_updatecache(filename, module_globals=None):
    try:
        return original_updatecache(filename, module_globals)
    except UnicodeDecodeError:
        return [] 
linecache.updatecache = safe_updatecache
# ==========================================

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from metrics.base_metrics_class import calculate_metrics_for_train

# ---------------------------
# 1. Unpaired Real Dataset
# ---------------------------
class UnpairedRealDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        
        cache_file = os.path.join(root_dir, 'dataset_cache.txt')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if not os.path.exists(cache_file):
            if local_rank == 0:
                print(f"[*] Rank 0 scanning directory {root_dir}...")
                temp_paths = []
                for dirpath, _, filenames in os.walk(root_dir):
                    for f in filenames:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            temp_paths.append(os.path.join(dirpath, f))
                
                with open(cache_file, 'w') as f:
                    for p in temp_paths:
                        f.write(p + '\n')
                print(f"[*] Cache saved to {cache_file}")
            else:
                while not os.path.exists(cache_file):
                    time.sleep(1)
        
        with open(cache_file, 'r') as f:
            self.paths = [line.replace('\x00', '').strip() for line in f.readlines() if line.strip()]
                
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# ---------------------------
# 2. Main Detector (Cosine Version)
# ---------------------------
@DETECTOR.register_module(module_name="asym_multi")
class Asym_multiDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
        feat_dim = config.get('feat_dim', 1024)
        
        # =====================================
        # ??????? 1????? Learnable Centers ??? EMA
        # =====================================
        self.num_centers = int(config.get('num_real_prototypes', 8))
        self.centers = nn.Parameter(torch.randn(self.num_centers, feat_dim))
        nn.init.xavier_uniform_(self.centers)
        
        # ??????е????????????? forward ????????????? Normalize
        self.fc = nn.Linear(feat_dim, 1)
        self.logit_scale = float(config.get('logit_scale', 16.0))
        # ????????
        self.lambda_bce = float(config.get('lambda2', 1.0))

        # ?? metric ??????????? real pull + fake repel ????????
        self.lambda_metric = float(config.get('lambda_metric', 1.0))

        # ???/????????????real pull ?? fake repel ??????
        # ???????? real ????????????? hard fake ????? real manifold ??????
        self.lambda_real_metric = float(config.get('lambda_real_metric', 0.5))
        self.lambda_fake_metric = float(config.get('lambda_fake_metric', 2.0))

        # fake repel ?????? real centers ?? top-k ??Υ???????? easy centers ???
        self.fake_topk = int(config.get('fake_topk', 3))

        self.lambda_center_div = float(config.get('lambda_center_div', 0.05))  # ?????? center ????
        self.lambda_balance = float(config.get('lambda_balance', 0.01))        # ??????? real ?????? center
        self.lambda_intra = float(config.get('lambda_intra', 0.05))
        self.lambda_conf = float(config.get('lambda4', 0.05))
        self.lambda_reg = float(config.get('lambda_reg', 0.0005))
        
        
        # -------------------------------
        # Configurable switches for ablation / paper experiments
        # -------------------------------
        self.use_unpaired_real = bool(config.get('use_unpaired_real', True))
        self.use_unpaired_real_for_confidence = bool(config.get('use_unpaired_real_for_confidence', True))
        self.use_unpaired_real_for_manifold = bool(config.get('use_unpaired_real_for_manifold', True))

        self.use_real_pull = bool(config.get('use_real_pull', True))
        self.use_center_div = bool(config.get('use_center_div', True))
        self.use_center_balance = bool(config.get('use_center_balance', True))
        self.use_intra_loss = bool(config.get('use_intra_loss', True))

        self.use_fake_repel = bool(config.get('use_fake_repel', True))
        self.use_patch_repel = bool(config.get('use_patch_repel', True))
        self.use_local_similarity_dropout = bool(config.get('use_local_similarity_dropout', True))
        self.use_boundary_mining = bool(config.get('use_boundary_mining', True))
        self.use_hard_fake_weighting = bool(config.get('use_hard_fake_weighting', True))
        self.detach_centers_for_fake = bool(config.get('detach_centers_for_fake', True))

        self.use_dynamic_margin = bool(config.get('use_dynamic_margin', True))
        self.use_fake_weight_warmup = bool(config.get('use_fake_weight_warmup', True))
        self.use_distance_fusion = bool(config.get('use_distance_fusion', False))

        # Cosine metric / curriculum boundary settings
        self.margin = float(config.get('margin', -0.2))
        self.margin_start = float(config.get('margin_start', self.margin))
        self.margin_end = float(config.get('margin_end', -0.6))
        self.margin_start_epoch = float(config.get('margin_start_epoch', 4.0))
        self.margin_end_epoch = float(config.get('margin_end_epoch', 8.0))
        self.steps_per_epoch = int(config.get('steps_per_epoch', 959))
        self.fake_warmup_epoch = float(config.get('fake_warmup_epoch', 3.0))

        self.local_dropout_prob = float(config.get('local_dropout_prob', 0.3))

        # Hard weighting thresholds. Keep old defaults to preserve old behavior.
        self.real_mid_dist_low = float(config.get('real_mid_dist_low', 0.2))
        self.real_mid_dist_high = float(config.get('real_mid_dist_high', 0.5))
        self.real_mid_weight = float(config.get('real_mid_weight', 3.0))
        self.real_extreme_weight = float(config.get('real_extreme_weight', 0.0))

        self.fake_mid_threshold_low = float(config.get('fake_mid_threshold_low', 0.1))
        self.fake_mid_threshold_high = float(config.get('fake_mid_threshold_high', 0.8))
        self.fake_mid_weight = float(config.get('fake_mid_weight', 5.0))
        self.fake_extreme_weight = float(config.get('fake_extreme_weight', 0.0))

        self.center_balance_temp = float(config.get('center_balance_temp', 0.1))
        # Soft real assignment temperature. Larger value reduces winner-takes-all center collapse.
        self.real_assign_temp = float(config.get('real_assign_temp', 0.5))
        self.pos_weight = float(config.get('pos_weight', 2.0))
        
        self.distance_scale = float(config.get('distance_scale', 1.0))
        self.alpha_mode = config.get('alpha_mode', 'fixed')
        self.alpha_min = float(config.get('alpha_min', 0.0))
        self.alpha_max = float(config.get('alpha_max', 1.0))
        alpha_init = float(config.get('alpha', 0.0))
        
        if self.alpha_mode == 'learnable':
            alpha_norm = (alpha_init - self.alpha_min) / (self.alpha_max - self.alpha_min + 1e-6)
            alpha_norm = min(max(alpha_norm, 1e-6), 1 - 1e-6)
            alpha_logit = math.log(alpha_norm / (1 - alpha_norm))
            self.alpha_param = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
            self.register_buffer('alpha_calibrated', torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer('alpha_param', torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer('alpha_calibrated', torch.tensor(alpha_init, dtype=torch.float32))
        
        self.unpaired_iter = self._build_unpaired_iter() if self.use_unpaired_real else None

        # Use a buffer so curriculum state can be saved/restored in checkpoints.
        self.register_buffer('global_step_tensor', torch.tensor(0, dtype=torch.long))

    def get_alpha(self):
        if self.alpha_mode == 'learnable':
            alpha = torch.sigmoid(self.alpha_param)
            return self.alpha_min + (self.alpha_max - self.alpha_min) * alpha
        return self.alpha_calibrated

    @torch.no_grad()
    def set_calibrated_alpha(self, alpha_value):
        alpha_value = float(alpha_value)
        alpha_value = min(max(alpha_value, self.alpha_min), self.alpha_max)
        self.alpha_calibrated.fill_(alpha_value)

    def _current_step(self):
        return int(self.global_step_tensor.item())

    def _current_margin(self):
        if not self.use_dynamic_margin:
            return self.margin_start

        start_step = int(self.margin_start_epoch * self.steps_per_epoch)
        end_step = int(self.margin_end_epoch * self.steps_per_epoch)
        step = self._current_step()

        if step <= start_step:
            return self.margin_start
        if step >= end_step:
            return self.margin_end

        progress = (step - start_step) / max(end_step - start_step, 1)
        return self.margin_start + progress * (self.margin_end - self.margin_start)

    def _current_fake_weight(self):
        if not self.use_fake_weight_warmup:
            return self.lambda_fake_metric

        warmup_steps = int(self.fake_warmup_epoch * self.steps_per_epoch)
        if warmup_steps <= 0:
            return self.lambda_fake_metric

        step = self._current_step() + 1
        if step < warmup_steps:
            return self.lambda_fake_metric * (step / warmup_steps)
        return self.lambda_fake_metric

    def _apply_local_similarity_dropout(self, sim_map, device):
        # This is not image-level dropout. It masks entries in the local-token-to-prototype similarity map.
        if (not self.use_local_similarity_dropout) or self.local_dropout_prob <= 0:
            return sim_map
        mask = torch.rand(sim_map.size(0), sim_map.size(1), 1, device=device) > self.local_dropout_prob
        return sim_map.masked_fill(~mask, -1e4)

    def _build_unpaired_iter(self):
        unpaired_bs = self.config.get('unpaired_batch_size', 16) 
        res = self.config.get('resolution', 224)
        mean = self.config.get('mean', [0.48145466, 0.4578275, 0.40821073])
        std = self.config.get('std', [0.26862954, 0.26130258, 0.27577711])
        
        transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        unpaired_real_dir = self.config.get('unpaired_real_dir', '/data2/Mining/real-candidates/')
        dataset = UnpairedRealDataset(unpaired_real_dir, transform=transform)
        num_workers = int(self.config.get('unpaired_workers', 1))
        
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=unpaired_bs,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        
        loader = DataLoader(**loader_kwargs)
        
        def infinite_generator():
            while True:
                for batch in loader:
                    yield batch
                    
        return infinite_generator()

    def build_backbone(self, config):
        if config['backbone_name'] == 'vit':
            from transformers import CLIPModel
            pretrained_path = config.get('pretrained', '') or config.get('backbone_config', {}).get('pretrained', '') or '/data2/Mining/active_learning/clip-model/'
            clip_model = CLIPModel.from_pretrained(pretrained_path)
            is_testing = bool(config.get('weights_path', ''))
            feat_dim = config.get('feat_dim', 1024)
            clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=feat_dim - 1, skip_svd=is_testing)
            
            class CLIPWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def features(self, x):
                    return self.model(x).last_hidden_state
            return CLIPWrapper(clip_model.vision_model)

        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        
        pretrained_path = config.get('pretrained', '')
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path)
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, False)
        return backbone

    def build_loss(self, config):
        return nn.BCELoss()

    def _extract_and_pool(self, x):
        feat = self.backbone.features(x)
        if len(feat.shape) == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.view(feat.size(0), -1)
        elif len(feat.shape) == 3:
            feat = feat[:, 0, :]
        return feat
        
    def _extract_both(self, x):
        feat = self.backbone.features(x)
        if len(feat.shape) == 4:
            b, c, h, w = feat.shape
            feat_global = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, c)
            feat_patches = feat.view(b, c, h * w).permute(0, 2, 1) # [B, num_patches, C]
        elif len(feat.shape) == 3:
            feat_global = feat[:, 0, :]
            feat_patches = feat[:, 1:, :] # [B, num_patches, C]
        return feat_global, feat_patches
        
    def features(self, data_dict: dict) -> torch.tensor:
        """????????????? L2 ????? (Cosine Metric ???)"""
        x = data_dict['image']
        feat_raw = self._extract_and_pool(x)
        feat_norm = F.normalize(feat_raw, p=2, dim=1)
        return feat_norm

    def _cosine_logits(self, features: torch.tensor) -> torch.tensor:
        """
        Cosine classifier with logit scale.
        features ????????? L2 normalize??
        """
        weight_norm = F.normalize(self.fc.weight, p=2, dim=1)
    
        logits = self.logit_scale * F.linear(
            features,
            weight_norm,
            None
        )
    
        if self.fc.bias is not None:
            logits = logits + self.fc.bias
    
        return logits


    def classifier(self, features: torch.tensor) -> torch.tensor:
        """???? scale ?? cosine classifier"""
        return self._cosine_logits(features)
    
    def forward(self, data_dict: dict, inference=False) -> dict:
      x = data_dict['image']
  
      # =====================================
      # centers and regularization tensors
      # =====================================
      centers_norm = F.normalize(self.centers, p=2, dim=1)
  
      orthogonal_losses = []
      for module in self.backbone.modules():
          if (
              isinstance(module, SVDResidualLinear)
              and hasattr(module, 'U_residual')
              and module.U_residual is not None
          ):
              orthogonal_losses.append(module.compute_orthogonal_loss())
  
      reg_term = sum(orthogonal_losses) if orthogonal_losses else torch.tensor(0.0, device=x.device)
  
      if hasattr(self, 'alpha_param') and self.alpha_param.requires_grad:
          reg_term = reg_term + 0.0 * self.alpha_param.sum()
  
      weight_norm = F.normalize(self.fc.weight, p=2, dim=1)
  
      # =====================================
      # inference branch
      # =====================================
      if inference:
          feat_raw = self._extract_and_pool(x)
          feat_norm = F.normalize(feat_raw, p=2, dim=1)
  
          logits = self._cosine_logits(feat_norm)
          p_cls = torch.sigmoid(logits).squeeze(-1)
  
          sim_to_all = torch.matmul(feat_norm, centers_norm.T)
          max_sim, _ = sim_to_all.max(dim=1)
          dist_to_center = (1.0 - max_sim) / self.distance_scale
  
          alpha = self.get_alpha().to(feat_norm.device, dtype=feat_norm.dtype)
          if self.use_distance_fusion:
              # The fused output is an anomaly score rather than a calibrated probability.
              prob = alpha * dist_to_center + (1 - alpha) * p_cls
          else:
              prob = p_cls
  
          return {
              'prob': prob.unsqueeze(1),
              'cls': p_cls.unsqueeze(1),
              'feat': feat_norm,
              'dist': dist_to_center.unsqueeze(1),
              'alpha': alpha.expand_as(p_cls).unsqueeze(1),
          }
  
      # =====================================
      # training branch
      # =====================================
      y_paired = data_dict['label']
      mask_real = (y_paired == 0)
      mask_fake = (y_paired == 1)
  
      x_rp = x[mask_real]
      x_f = x[mask_fake]
      if self.use_unpaired_real:
          x_ru = next(self.unpaired_iter).to(device=x.device, dtype=x.dtype)
      else:
          x_ru = x.new_empty((0, *x.shape[1:]))
  
      num_ru = x_ru.size(0)
      num_rp = x_rp.size(0)
      num_f = x_f.size(0)
  
      x_all = torch.cat([x_ru, x_rp, x_f], dim=0)
  
      # ???????????????? _extract_both ???????????????
      z_all_raw, z_patches_raw = self._extract_both(x_all)
      
      z_all = F.normalize(z_all_raw, p=2, dim=1)
      z_patches = F.normalize(z_patches_raw, p=2, dim=2)
  
      logits_all = self._cosine_logits(z_all)
      p_all = torch.sigmoid(logits_all).squeeze(-1)
  
      z_ru = z_all[:num_ru]
      z_rp = z_all[num_ru:num_ru + num_rp]
      
      z_f = z_all[num_ru + num_rp:]
      z_f_p = z_patches[num_ru + num_rp:] # ??????????????????????
  
      p_ru = p_all[:num_ru]
      p_rp = p_all[num_ru:num_ru + num_rp]
      p_f = p_all[num_ru + num_rp:]
  
      p_paired = torch.zeros_like(y_paired, dtype=p_all.dtype)
      if num_rp > 0: p_paired[mask_real] = p_rp
      if num_f > 0:  p_paired[mask_fake] = p_f
  
      return {
          'z_ru': z_ru, 'z_rp': z_rp, 'z_f': z_f, 
          'z_f_p': z_f_p, # ???????????????????????? get_losses
          'p_ru': p_ru, 'p_rp': p_rp, 'p_f': p_f,
          'cls': p_paired.unsqueeze(1), 'prob': p_paired.unsqueeze(1),
          'centers_norm': centers_norm, 'reg_term': reg_term
      }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        z_ru, z_rp, z_f = pred_dict['z_ru'], pred_dict['z_rp'], pred_dict['z_f']
        p_ru, p_rp, p_f = pred_dict['p_ru'], pred_dict['p_rp'], pred_dict['p_f']
        
        if z_ru.numel() > 0:
            device = z_ru.device
        elif z_rp.numel() > 0:
            device = z_rp.device
        else:
            device = z_f.device
        
        # ??????? forward ????????????????????????? DDP ???????????
        centers_norm = pred_dict['centers_norm']
        reg_loss_val = self.lambda_reg * pred_dict['reg_term']

        loss_real = torch.tensor(0.0, device=device)
        loss_fake = torch.tensor(0.0, device=device)
        loss_center_div = torch.tensor(0.0, device=device)
        loss_balance = torch.tensor(0.0, device=device)
        loss_intra = torch.tensor(0.0, device=device)
        current_margin = self._current_margin()
        dynamic_fake_weight = self._current_fake_weight()
        mean_fake_violation = torch.tensor(0.0, device=device)
        max_fake_sim_mean = torch.tensor(0.0, device=device)
        center_usage_mean = torch.tensor(0.0, device=device)
        # =========================================================
        # 1. Center Diversity Loss
        # =========================================================
        num_centers = centers_norm.size(0)
        if self.use_center_div and num_centers > 1:
            center_sim = torch.matmul(centers_norm, centers_norm.T)  # [K, K]
            off_diag_mask = ~torch.eye(num_centers, dtype=torch.bool, device=device)
            # [???????? 2] ?????? Bug?????? F.relu
            loss_center_div = F.relu(center_sim[off_diag_mask]).mean()

        # =========================================================
        # 2. ?????????￡??Pull real samples to nearest real center
        # =========================================================
        real_feats = []
        if self.use_unpaired_real_for_manifold and z_ru.numel() > 0:
            real_feats.append(z_ru)
        if z_rp.numel() > 0:
            real_feats.append(z_rp)
        
        if self.use_real_pull and real_feats:
            real_feats = torch.cat(real_feats, dim=0)
            real_sim = torch.matmul(real_feats, centers_norm.T)  # [N_real, K]

            real_temp = max(float(self.real_assign_temp), 1e-6)
            real_assign_prob = F.softmax(real_sim / real_temp, dim=1)  # [N_real, K]
            real_dist = (real_assign_prob.detach() * (1.0 - real_sim)).sum(dim=1)
            
            # ?????????? 1.0
            hard_weight_real = torch.ones_like(real_dist)
            
            # 1. ????е??????????? 0.2 ?? 0.5 ????????????????磩?????? 3 ?????????
            med_real_mask = (real_dist > self.real_mid_dist_low) & (real_dist <= self.real_mid_dist_high)
            hard_weight_real[med_real_mask] = self.real_mid_weight
            
            # 2. ????????????????????????????????????????????
            extreme_real_mask = (real_dist > self.real_mid_dist_high)
            hard_weight_real[extreme_real_mask] = self.real_extreme_weight
            
            loss_real = (real_dist * hard_weight_real).mean()

            temp = max(float(self.center_balance_temp), 1e-6)
            assign_prob = F.softmax(real_sim / temp, dim=1)      # [N_real, K]
            center_usage = assign_prob.mean(dim=0)               # [K]
            center_usage_mean = center_usage.max().detach()
            uniform_usage = torch.full_like(center_usage, 1.0 / num_centers)
            if self.use_center_balance:
                loss_balance = (center_usage - uniform_usage).pow(2).sum()
            assign_idx = real_sim.detach().argmax(dim=1)
      
            if self.use_intra_loss:
                intra_losses = []
            
                for k in range(num_centers):
                    idx_k = (assign_idx == k)
            
                    if idx_k.sum().item() > 1:
                        feat_k = real_feats[idx_k]
            
                        mean_k = feat_k.mean(dim=0, keepdim=True)
                        mean_k = F.normalize(mean_k, p=2, dim=1)
            
                        sim_k = torch.matmul(feat_k, mean_k.T).squeeze(1)
            
                        intra_losses.append((1.0 - sim_k).mean())
            
                if len(intra_losses) > 0:
                    loss_intra = torch.stack(intra_losses).mean()

        # =========================================================
        # 3. ????????????Global fake repel from all real centers
        # =========================================================
        # =========================================================
        # 3. ???????????? (???? Patch Dropout ?? ??????)
        # =========================================================
        if self.use_fake_repel and z_f.numel() > 0:
            centers_for_fake = centers_norm.detach() if self.detach_centers_for_fake else centers_norm

            if self.use_patch_repel:
                # Local intrusion: [N_fake, Num_Patches, K]
                z_f_p = pred_dict['z_f_p']
                fake_sim_p = torch.matmul(z_f_p, centers_for_fake.T)
                fake_sim_p = self._apply_local_similarity_dropout(fake_sim_p, device)
                max_fake_sim, _ = fake_sim_p.max(dim=1)  # [N_fake, K]
            else:
                # Global intrusion: [N_fake, K]
                max_fake_sim = torch.matmul(z_f, centers_for_fake.T)

            max_fake_sim_mean = max_fake_sim.mean().detach()

            # Dynamic or fixed curriculum boundary.
            fake_violation = F.relu(max_fake_sim - current_margin)

            if self.use_boundary_mining:
                topk = min(max(int(self.fake_topk), 1), fake_violation.size(1))
                hard_fake_violation = torch.topk(
                    fake_violation,
                    k=topk,
                    dim=1,
                    largest=True
                ).values
            else:
                hard_fake_violation = fake_violation

            if self.use_hard_fake_weighting:
                hard_weight_fake = torch.ones_like(hard_fake_violation)
                med_fake_mask = (
                    (hard_fake_violation > self.fake_mid_threshold_low) &
                    (hard_fake_violation <= self.fake_mid_threshold_high)
                )
                hard_weight_fake[med_fake_mask] = self.fake_mid_weight

                extreme_fake_mask = (hard_fake_violation > self.fake_mid_threshold_high)
                hard_weight_fake[extreme_fake_mask] = self.fake_extreme_weight
                loss_fake = (hard_fake_violation.pow(2) * hard_weight_fake).mean()
            else:
                loss_fake = hard_fake_violation.pow(2).mean()

            mean_fake_violation = hard_fake_violation.mean().detach()
            
        # =========================================================
        # ????????????real/fake metric ??????
        # =========================================================
        
        # Curriculum fake weight is computed from the checkpointable global step buffer.
        self.global_step_tensor += 1
            
        metric_loss = (
            self.lambda_real_metric * loss_real +
            dynamic_fake_weight * loss_fake
        )

        # =========================================================
        # 4. BCE ?? Confidence Loss ????????
        # =========================================================
        label_rp = torch.zeros_like(p_rp)
        label_f = torch.ones_like(p_f)
        
        loss_bce = torch.tensor(0.0, device=device)
        if len(p_rp) > 0:
            loss_bce = loss_bce + self.loss_func(p_rp, label_rp)
        if len(p_f) > 0:
            loss_bce = loss_bce + self.pos_weight * self.loss_func(p_f, label_f)
            
        loss_conf = -torch.log(1 - p_ru + 1e-6).mean() if (self.use_unpaired_real_for_confidence and len(p_ru) > 0) else torch.tensor(0.0, device=device)
            
        # =========================================================
        # 5. Overall Loss
        # =========================================================
        total_loss = (
            self.lambda_bce * loss_bce +
            self.lambda_metric * metric_loss +
            self.lambda_center_div * loss_center_div +
            self.lambda_balance * loss_balance +
            self.lambda_intra * loss_intra +
            self.lambda_conf * loss_conf +
            reg_loss_val
        )

        return {
            'loss_real': loss_real,
            'loss_fake': loss_fake,
            'loss_center_div': loss_center_div,
            'loss_balance': loss_balance,
            'loss_intra': loss_intra,
            'loss_bce': loss_bce,
            'loss_conf': loss_conf,
            'loss_reg': reg_loss_val,
            'metric_loss': metric_loss,
            'current_margin': torch.tensor(current_margin, device=device),
            'dynamic_fake_weight': torch.tensor(dynamic_fake_weight, device=device),
            'mean_fake_violation': mean_fake_violation,
            'max_fake_sim': max_fake_sim_mean,
            'center_usage_max': center_usage_mean,
            'global_step': self.global_step_tensor.detach().float(),
            'overall': total_loss
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

# ==============================================================================
# SVD PEFT ??????? (???????)
# ==============================================================================
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r 
        
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)
    
    def compute_orthogonal_loss(self):
        UUT_residual = self.U_residual @ self.U_residual.t()
        VVT_residual = self.V_residual @ self.V_residual.t()
        
        UUT_residual_identity = torch.eye(UUT_residual.size(0), device=UUT_residual.device)
        VVT_residual_identity = torch.eye(VVT_residual.size(0), device=VVT_residual.device)
        
        loss = 0.5 * torch.norm(UUT_residual - UUT_residual_identity, p='fro') + 0.5 * torch.norm(VVT_residual - VVT_residual_identity, p='fro')
        return loss
        
def apply_svd_residual_to_self_attn(model, r, skip_svd=False):
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r, skip_svd))
        else:
            apply_svd_residual_to_self_attn(module, r, skip_svd)
            
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def replace_with_svd_residual(module, r, skip_svd=False):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone() if not skip_svd else None)

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        if skip_svd:
            min_dim = min(in_features, out_features)
            r_actual = min(r, min_dim)
            new_module.weight_main.data.zero_()
            
            if min_dim > r_actual:
                new_module.S_residual = nn.Parameter(torch.zeros(min_dim - r_actual))
                new_module.U_residual = nn.Parameter(torch.zeros(out_features, min_dim - r_actual))
                new_module.V_residual = nn.Parameter(torch.zeros(min_dim - r_actual, in_features))
            else:
                new_module.S_residual = None
                new_module.U_residual = None
                new_module.V_residual = None
            return new_module

        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
        r = min(r, len(S))  

        U_r = U[:, :r]      
        S_r = S[:r]         
        Vh_r = Vh[:r, :]    

        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        new_module.weight_main.data.copy_(weight_main)

        U_residual = U[:, r:]    
        S_residual = S[r:]       
        Vh_residual = Vh[r:, :]  

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
        return new_module
    else:
        return module