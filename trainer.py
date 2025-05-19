from OCR.ocr_manager import OCRManager
from torch.nn import CrossEntropyLoss
import torch
from OCR.ocr_utils import LM_ind_to_str
import numpy as np
from torch.cuda.amp import autocast
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from basic.custom_conv import Conv2d
import torch.optim as optim
import matplotlib.pyplot as plt
class Manager(OCRManager):

    def __init__(self, params):
        super(Manager, self).__init__(params)

    def load_save_info(self, info_dict):
        if "curriculum_config" in info_dict.keys():
            if self.dataset.train_dataset is not None:
                self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict

    def get_init_hidden(self, batch_size):
        num_layers = 1
        hidden_size = self.params["model_params"]["enc_dim"]
        return torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)

    def apply_teacher_forcing(self, y, y_len, error_rate):
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate and y[b][i] != self.dataset.tokens["pad"]:
                    y_error[b][i] = np.random.randint(0, len(self.dataset.charset)+2)
        return y_error, y_len

    def train_batch(self, batch_data, metric_names):
        loss_func = CrossEntropyLoss(ignore_index=self.dataset.tokens["pad"])

        x = batch_data["imgs"].to(self.device)

        x_masked, mask = self.random_masking(x, mask_ratio=0.75)

        meta_model = self.models["encoder"]
        meta_auxi_model = l2l.algorithms.MAML(meta_model, lr=1e-3, first_order=True)
        # Select one image for adaptation
        x_support = x_masked[0:1]  # Use the first image for adaptation
        x_query = x_masked[1:]
        sum_loss = 0

        y = batch_data["labels"].to(self.device)
        y_query = y[1:]
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        reduced_size_query = reduced_size[1:]
        y_len = batch_data["labels_len"]
        y_len_query = y_len[1:]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred, y_len_query = self.apply_teacher_forcing(y_query, y_len_query, error_rate)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred, y_len_query = self.apply_teacher_forcing(y_query, y_len_query, error_rate)
        else:
            simulated_y_pred = y_query

        with autocast(enabled=self.params["training_params"]["use_amp"]):
            hidden_predict = None
            cache = None

            num_adaptation_steps = 1
            learner = meta_auxi_model.clone().requires_grad_(True)
            for _ in range(num_adaptation_steps):
                #self.freeze_params(learner)
                #self.make_padding_trainable(learner)
                _ , reconstructed = learner(x_support)
                reconstructed = reconstructed.to(device=self.device, dtype=torch.float16)
                reconstructed = F.interpolate(reconstructed, size=(x_support.size(2), x_support.size(3)), mode='bilinear', align_corners=False)
                inner_loss = self.ssim(x_support, reconstructed, max_val=1.0)
                learner.adapt(inner_loss, allow_unused=True)

            raw_features = learner(x_query)

            raw_features = raw_features[0]
            features_size = raw_features.size()
            #print(features_size)
            if len(features_size) != 4:
                raise ValueError(f"Expected 4 dimensions (b, c, h, w), but got {len(features_size)} dimensions: {features_size}")
            b, c, h, w = features_size

            
            # primary tesk decoder
            pos_features = self.models["decoder"].features_updater.get_pos_features(raw_features)#raw_features


            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            output, pred, hidden_predict, cache, weights = self.models["decoder"](features, enhanced_features,
                                                                            simulated_y_pred[:, :-1],
                                                                            reduced_size_query,
                                                                            [max(y_len_query) for _ in range(b)],
                                                                            features_size,
                                                                            start=0,
                                                                            hidden_predict=hidden_predict,
                                                                            cache=cache,
                                                                            keep_all_weights=True)
            

            loss_ce = loss_func(pred, y_query[:, 1:])
            sum_loss += loss_ce
            with autocast(enabled=False):
                self.zero_optimizers()
                self.backward_loss(sum_loss)              
                self.step_optimizers()

            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len_query[i]] for i in range(b)]
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "loss": sum_loss.item(),
            "loss_ce": loss_ce.item(),
            "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"]["synthetic_data"] else 0,
        }
        return values

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        max_chars = self.params["training_params"]["max_char_prediction"]

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b, ), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
            predicted_tokens_len = torch.ones((b, ), dtype=torch.int, device=self.device)

            whole_output = list()
            confidence_scores = list()
            cache = None
            hidden_predict = None
            if b > 1:
                features_list = list()
                for i in range(b):
                    pos = batch_data["imgs_position"]
                    output = self.models["encoder"](x[i:i+1, :, pos[i][0][0]:pos[i][0][1], pos[i][1][0]:pos[i][1][1]])
                    output = output[0].unsqueeze(0) if output[0].dim() == 3 else output[0]

                    features_list.append(output)
                max_height = max([f.size(2) if f.dim() > 2 else 0 for f in features_list])  # Ensure dimension exists
                max_width = max([f.size(3) if f.dim() > 3 else 0 for f in features_list])  # Use a default value if not

                features = torch.zeros((b, features_list[0].size(1), max_height, max_width), device=self.device, dtype=features_list[0].dtype)
                for i in range(b):
                    f_height, f_width = features_list[i].size(2), features_list[i].size(3)
                    if features_list[i].dim() == 4:  # Confirming that we have a 4D tensor
                        features[i, :, :f_height, :f_width] = features_list[i]
                    else:
                        print(f"Unexpected tensor shape encountered in features_list at index {i}: {features_list[i].shape}")
            else:
                encoder_output = self.models["encoder"](x)
                features = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
            features_size = features.size()
            coverage_vector = torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device)
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            for i in range(0, max_chars):
                output, pred, hidden_predict, cache, weights = self.models["decoder"](features, enhanced_features, predicted_tokens, reduced_size, predicted_tokens_len, features_size, start=0, hidden_predict=hidden_predict, cache=cache, num_pred=1)
                whole_output.append(output)
                confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                coverage_vector = torch.clamp(coverage_vector + weights, 0, 1)
                predicted_tokens = torch.cat([predicted_tokens, torch.argmax(pred[:, :, -1], dim=1, keepdim=True)], dim=1)
                reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.dataset.tokens["end"]))
                predicted_tokens_len += 1

                prediction_len[reached_end == False] = i + 1
                if torch.all(reached_end):
                    break

            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            predicted_tokens = predicted_tokens[:, 1:]
            prediction_len[torch.eq(reached_end, False)] = max_chars - 1
            predicted_tokens = [predicted_tokens[i, :prediction_len[i]] for i in range(b)]
            confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        process_time = time.time() - start_time

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
        }
        return values

    def test_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        x_masked, mask = self.random_masking(x, mask_ratio=0.75)
        meta_model = self.models["encoder"]

        meta_auxi_model = l2l.algorithms.MAML(meta_model, lr=1e-3)

        # Select one image for adaptation
        x_adapt = x_masked[0:1]  # Use the first image for adaptation
        x_test = x[1:]  # The rest are for testing

        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        max_chars = self.params["training_params"]["max_char_prediction"]

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x_test.size(0)
            reached_end = torch.zeros((b, ), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
            predicted_tokens_len = torch.ones((b, ), dtype=torch.int, device=self.device)

            whole_output = list()
            confidence_scores = list()
            cache = None
            hidden_predict = None
            
            ####inner loop
            num_adaptation_steps = 1
            learner = meta_auxi_model.clone().requires_grad_(True)
            
            for _ in range(num_adaptation_steps):
                self.freeze_params(learner)
                self.make_padding_trainable(learner)
                _ , reconstructed = learner(x_adapt)
                reconstructed = reconstructed.to(device=self.device, dtype=torch.float16)
                reconstructed = F.interpolate(reconstructed, size=(x_adapt.size(2), x_adapt.size(3)), mode='bilinear', align_corners=False)
                inner_loss = self.ssim(x_adapt, reconstructed, max_val=1.0)
                self.manually_adapt_learner(learner, inner_loss, lr=1e-3)
                #learner.adapt(inner_loss,allow_unused=True)
            learner.eval()
            
            with torch.no_grad():
                if b > 1:
                    features_list = list()
                    for i in range(b):
                        pos = batch_data["imgs_position"]
                        output = learner(x_test[i:i+1, :, pos[i][0][0]:pos[i][0][1], pos[i][1][0]:pos[i][1][1]])
                        features_list.append(output[0])
                    max_height = max([f.size(2) for f in features_list])
                    max_width = max([f.size(3) for f in features_list])
                    features = torch.zeros((b, features_list[0].size(1), max_height, max_width), device=self.device, dtype=features_list[0].dtype)
                    for i in range(b):
                        features[i, :, :features_list[i].size(2), :features_list[i].size(3)] = features_list[i]
                else:
                    encoder_output = learner(x_test)
                    features = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output

                torch.cuda.empty_cache()
                #outer loop
                features_size = features.size()
                coverage_vector = torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device)
                pos_features = self.models["decoder"].features_updater.get_pos_features(features)
                features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
                enhanced_features = pos_features
                enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)

                for i in range(0, max_chars):
                    output, pred, hidden_predict, cache, weights = self.models["decoder"](features, enhanced_features, predicted_tokens, reduced_size, predicted_tokens_len, features_size, start=0, hidden_predict=hidden_predict, cache=cache, num_pred=1)
                    whole_output.append(output)
                    confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                    coverage_vector = torch.clamp(coverage_vector + weights, 0, 1)
                    predicted_tokens = torch.cat([predicted_tokens, torch.argmax(pred[:, :, -1], dim=1, keepdim=True)], dim=1)
                    reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.dataset.tokens["end"]))
                    predicted_tokens_len += 1

                    prediction_len[reached_end == False] = i + 1
                    if torch.all(reached_end):
                        break

                confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
                predicted_tokens = predicted_tokens[:, 1:]
                prediction_len[torch.eq(reached_end, False)] = max_chars - 1
                predicted_tokens = [predicted_tokens[i, :prediction_len[i]] for i in range(b)]
                confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]

                str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]
        
        process_time = time.time() - start_time

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"][1:],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
        }
        return values
    

    def ssim(self, x, y, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        C1 = (k1 * max_val) ** 2
        C2 = (k2 * max_val) ** 2
        kernel = self.create_gaussian_kernel(filter_size, filter_sigma, x.size(1))
        kernel = kernel.to(device=x.device, dtype=x.dtype)
        mu_x = F.conv2d(x, kernel, padding=filter_size//2, groups=x.size(1))
        mu_y = F.conv2d(y, kernel, padding=filter_size//2, groups=y.size(1))
            
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_x_mu_y = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, kernel, padding=filter_size//2, groups=x.size(1)) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, kernel, padding=filter_size//2, groups=y.size(1)) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel, padding=filter_size//2, groups=y.size(1)) - mu_x_mu_y

        ssim_numerator = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        ssim_denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = ssim_numerator / ssim_denominator
        return ssim_map.mean()

    def create_gaussian_kernel(self, size, sigma, channels):
        """Utility function for creating a Gaussian kernel for SSIM"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()  # Normalize to ensure sum is 1
        kernel = g[:, None] * g[None, :]
        kernel = kernel.to(torch.float32).repeat(channels, 1, 1, 1)
        return kernel
    
    def random_masking(self, x, mask_ratio=0.75):
        N, C, H, W = x.shape
        mask_count = int(mask_ratio * H * W)
        mask = torch.rand(N, C, H, W, device=x.device) < mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0  # Apply masking
        return x_masked, mask
    
    def freeze_params(self, model):
        self.hooks = []
        self.frozen_modules = []

        # Loop through the named modules in the model
        for name, module in model.named_modules():
            if 'prompt_embeddings_tb' not in name and 'prompt_embeddings_lr' not in name and 'final_layer' not in name:
                self.frozen_modules.append(module)
                hook = module.register_forward_hook(self._disable_gradients)
                self.hooks.append(hook)

    def _disable_gradients(self, module, input, output):
        
        with torch.no_grad():
            return output  # Returning the output without gradient tracking

    def manually_adapt_learner(self, learner, inner_loss, lr=1e-3):
        with torch.cuda.amp.autocast(enabled=True):
            inner_loss.backward(retain_graph=True)

            learner_grads = {}
            for name, param in learner.named_parameters():
                if param.requires_grad and param.grad is not None:
                    learner_grads[name] = param.grad.clone()  # Store a copy of the gradient
                else:
                    learner_grads[name] = None

            prev_state_dict = learner.state_dict()  # Save the current state
            leaf_params = [p for p in learner.parameters() if p.is_leaf and p.requires_grad]
            inner_optimizer = optim.Adam(leaf_params, lr=lr)

            self.meta_update_learner(learner, inner_optimizer, inner_loss, learner_grads)

    def meta_update_learner(self, learner, optimizer, loss, gradients):
        # Register hooks for each parameter in the learner
        hooks = []
        for name, param in learner.named_parameters():
            if gradients[name] is not None:
                hooks.append(param.register_hook(lambda grad, g=gradients[name]: g))

        # Zero out existing gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for hook in hooks:
            hook.remove()
            
    def make_padding_trainable(self,model, crop_size=224):
        c_count = 0
        m_count = 0
        for i,m in enumerate(model.children()):
            
            if i < 1: 
                for idx, q in enumerate(m.children()):
                    if idx == 0:
                        # First innermost loop
                        for ids, w in enumerate(q.children()):
                            for idc, e in enumerate(w.children()):
                                if isinstance(e, Conv2d):
                                    c_count += 1
                                    if c_count > 18:
                                        break
                                    P = e.padding[0]
                                    S = e.stride[0]
                                    K = e.kernel_size[0]
                                    e.data_crop_size = crop_size
                                    e.make_padding_trainable()
                                    for k, v in e.state_dict().items():
                                        if k in ['prompt_embeddings_tb', 'prompt_embeddings_lr']:
                                            for param in v:
                                                param.requires_grad = True





