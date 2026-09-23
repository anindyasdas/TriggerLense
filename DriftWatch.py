import torch
import numpy as np
import copy
import logging
import time
import gc
from sklearn.ensemble import IsolationForest
from scipy import stats
import gc
import os
import shutil
from collections import deque
import torch.nn.functional as F



from helper import (
    rolling_zscore,
    current_median_std,
    rolling_median_std,
    rolling_sum_last,
    safe_head_peak_std,
    hidden_semantic_distance,
    compute_prompt_hidden_mean,
    bscore_from_attn,
    squash,
)

class DriftWatch:
    def __init__(self, model_target, model_ref, tokenizer, k=1, back_len=0, forward_len=1,
             max_length=1024, verbose=False, anomaly_method='combined', window_size=50,
             attack_type=None,
             use_semantic=True,
             use_representation=True,
             use_attention=True):
        self.attack_type = attack_type
        self.anomaly_method = anomaly_method

        # detector hyperparams
        self.H_CTX_K = 30
        self.warmup_len = 10
        self.Z_WINDOW = 20
        self.Z_MIN_FILL = 5
        self.BURST_WINDOW = 3
        self.THR_FLOOR = 1.0
        self.THRESH_SCALE = 1.5
        self.MAX_HEADS = 32
        self.NORM_WIN = 70
        self.BOTTOM_PCT = 0.01

        self.SEM_WEIGHTS = {
            "z_delta_sf": 0.44,
            "z_sg": 0.32,
            "z_sf": 0.24,
        }

        self.REPR_WEIGHTS = {
            "z_h_dir": 0.50,
            "z_h_drift": 0.32,
            "z_head": 0.18,
        }

        self.AGREEMENT_BONUS = 0.15
        self.reset_detector_state()
        self.window_size = window_size #AD2026
        self.model_target = model_target
        self.model_ref = model_ref
        self.tokenizer = tokenizer
        self.k = k
        self.back_len = back_len
        self.forward_len = forward_len
        self.max_length = max_length
        self.finish_buffer = 30
        self.verbose = verbose
        self.m = 2048
        self.max_iterations = 1 
        self.anomaly_method = anomaly_method
        self.use_semantic = use_semantic
        self.use_representation = use_representation
        self.use_attention = use_attention
        logging.info(f"DriftWatch initialized with {anomaly_method} anomaly detection.")

    def reset_detector_state(self):
        self.detector_state = {
            "recent_h_buffer": deque(maxlen=self.H_CTX_K),
            "prev_h_repr": None,
            "prev_sf_h": None,
            "bscore_prev": None,
            "prev_centroid": None,
        }

        self.history = {
            **{k: deque(maxlen=self.NORM_WIN) for k in [
                "norm_zscore", "bscore", "delta_bscore",
                "centroid_drift", "attn_centroid", "centroid_jump_raw"
            ]},
            **{k: deque(maxlen=self.Z_WINDOW) for k in [
                "h_drift", "h_dir", "head_std", "sf_h", "sg_h", "delta_sf"
            ]},
            **{k: deque(maxlen=self.NORM_WIN) for k in [
                "anomaly_score1", "anomaly_score_twotrack",
                "burst_anomaly", "burst_anomaly_score1", "combined_anomaly"
            ]}
        }
    def compute_combined_anomaly(self, out, generated_text_ids, token_pos, prompt_len):
        h = out.hidden_states[-1][0, token_pos].float()

        recent_h_buffer = self.detector_state["recent_h_buffer"]
        prev_h_repr = self.detector_state["prev_h_repr"]
        prev_sf_h = self.detector_state["prev_sf_h"]
        bscore_prev = self.detector_state["bscore_prev"]
        prev_centroid = self.detector_state["prev_centroid"]

        # attention for current token
        last_attn_h = out.attentions[-1][0, :, token_pos, :].float()
        attn_mean = last_attn_h.mean(dim=0).detach().cpu().numpy()
        attn_prompt = attn_mean[:prompt_len]

        # hidden drift / direction
        if prev_h_repr is None:
            h_drift = 0.0
            h_dir = 0.0
        else:
            h_drift = float(torch.norm(h - prev_h_repr).item())
            h_dir = 1.0 - float(F.cosine_similarity(
                h.unsqueeze(0), prev_h_repr.unsqueeze(0)
            ).item())
            if not np.isfinite(h_drift):
                h_drift = 0.0
            if not np.isfinite(h_dir):
                h_dir = 0.0

        sf_h = hidden_semantic_distance(h, recent_h_buffer)
        sg_h = hidden_semantic_distance(h, self.prompt_h_mean)

        if prev_sf_h is None:
            delta_sf = 0.0
        else:
            delta_sf = sf_h - prev_sf_h
            if not np.isfinite(delta_sf):
                delta_sf = 0.0

        last_attn = out.attentions[-1][0]
        if self.MAX_HEADS and last_attn.shape[0] > self.MAX_HEADS:
            last_attn = last_attn[:self.MAX_HEADS]
        head_std = safe_head_peak_std(last_attn)

        # z-scores
        z_h_drift = rolling_zscore(self.history["h_drift"], h_drift, min_fill=self.Z_MIN_FILL)
        z_h_dir   = rolling_zscore(self.history["h_dir"], h_dir, min_fill=self.Z_MIN_FILL)
        z_head    = rolling_zscore(self.history["head_std"], head_std, min_fill=self.Z_MIN_FILL)
        z_sf_h    = rolling_zscore(self.history["sf_h"], sf_h, min_fill=self.Z_MIN_FILL)
        z_sg_h    = rolling_zscore(self.history["sg_h"], sg_h, min_fill=self.Z_MIN_FILL)
        z_delta_sf = rolling_zscore(self.history["delta_sf"], delta_sf, min_fill=self.Z_MIN_FILL)
        z_delta_sf = max(0.0, z_delta_sf)

        p_h_drift  = np.clip(z_h_drift, 0, 5.0)
        p_h_dir    = np.clip(z_h_dir, 0, 5.0)
        p_head     = np.clip(z_head, 0, 5.0)
        p_sf_h     = np.clip(z_sf_h, 0, 5.0)
        p_sg_h     = np.clip(z_sg_h, 0, 5.0)
        p_delta_sf = np.clip(z_delta_sf, 0, 5.0)

        track_sem = float(
            self.SEM_WEIGHTS["z_delta_sf"] * p_delta_sf +
            self.SEM_WEIGHTS["z_sg"] * p_sg_h +
            self.SEM_WEIGHTS["z_sf"] * p_sf_h
        )

        track_repr = float(
            self.REPR_WEIGHTS["z_h_dir"] * p_h_dir +
            self.REPR_WEIGHTS["z_h_drift"] * p_h_drift +
            self.REPR_WEIGHTS["z_head"] * p_head
        )

        track_sem_active = track_sem if self.use_semantic else 0.0
        track_repr_active = track_repr if self.use_representation else 0.0
        if self.use_semantic and self.use_representation:
            anomaly_score_twotrack = (
                track_sem_active
                + self.AGREEMENT_BONUS * min(track_sem_active, track_repr_active)
            )
        else:
            anomaly_score_twotrack = track_sem_active + track_repr_active

        anomaly_score_twotrack = float(np.log1p(anomaly_score_twotrack))

        temp_scores = list(self.history["anomaly_score_twotrack"]) + [anomaly_score_twotrack]
        if len(temp_scores) < self.BURST_WINDOW:
            burst_anomaly = 0.0
        else:
            burst_anomaly = rolling_sum_last(temp_scores, self.BURST_WINDOW)

        self.history["h_drift"].append(h_drift)
        self.history["h_dir"].append(h_dir)
        self.history["head_std"].append(head_std)
        self.history["sf_h"].append(sf_h)
        self.history["sg_h"].append(sg_h)
        self.history["delta_sf"].append(delta_sf)

        h_norm = torch.norm(h).item()
        norm_zscore = rolling_zscore(self.history["norm_zscore"], h_norm)
        self.history["norm_zscore"].append(h_norm)

        b_score = bscore_from_attn(out, 0, prompt_len, L=-1, bottom_pct=self.BOTTOM_PCT)
        delta_b = (b_score - bscore_prev) if bscore_prev is not None else 0.0

        z_bscore = rolling_zscore(self.history["bscore"], b_score)
        self.history["bscore"].append(b_score)
        z_delta_b = rolling_zscore(self.history["delta_bscore"], delta_b)
        self.history["delta_bscore"].append(delta_b)

        positions = np.arange(prompt_len, dtype=np.float32)
        attn_p_sum = attn_prompt.sum() + 1e-12
        centroid = float((positions * attn_prompt).sum() / attn_p_sum) if prompt_len > 0 else 0.0

        if prev_centroid is not None:
            centroid_drift = abs(centroid - prev_centroid)
        else:
            centroid_drift = 0.0

        z_centroid_drift = rolling_zscore(self.history["centroid_drift"], centroid_drift)
        self.history["centroid_drift"].append(centroid_drift)
        z_attn_centroid = rolling_zscore(self.history["attn_centroid"], centroid)
        self.history["attn_centroid"].append(centroid)

        s_h   = max(0.0, norm_zscore)
        s_b   = max(0.0, z_bscore)
        s_db  = max(0.0, abs(z_delta_b))
        s_cdr = max(0.0, z_centroid_drift)
        s_ctr = max(0.0, z_attn_centroid)

        u_h   = squash(s_h)
        u_b   = squash(s_b)
        u_db  = squash(s_db)
        u_cdr = squash(s_cdr)
        u_ctr = squash(s_ctr)

        w_h, w_b, w_db, w_cdr, w_ctr = 0.10, 0.40, 0.20, 0.20, 0.10
        w_sum = w_h + w_b + w_db + w_cdr + w_ctr
        eps = 1e-6

        logp = (
            w_h * np.log(u_h + eps) +
            w_b * np.log(u_b + eps) +
            w_db * np.log(u_db + eps) +
            w_cdr * np.log(u_cdr + eps) +
            w_ctr * np.log(u_ctr + eps)
        )

        anomaly_score1 = 0.8 * np.exp(logp / w_sum) + 0.2 * u_b

        temp_scores1 = list(self.history["anomaly_score1"]) + [anomaly_score1]
        if len(temp_scores1) < self.BURST_WINDOW:
            burst_anomaly_score1 = 0.0
        else:
            burst_anomaly_score1 = rolling_sum_last(temp_scores1, self.BURST_WINDOW)

        if not self.use_attention:
            anomaly_score1 = 0.0
            burst_anomaly_score1 = 0.0

        curr_len = min(len(self.history["burst_anomaly_score1"]), len(self.history["burst_anomaly"]))
        thr_floor = self.THR_FLOOR if curr_len < self.warmup_len else 0.0

        med1, std1 = current_median_std(list(self.history["burst_anomaly_score1"]), w=self.NORM_WIN, min_fill=5)
        med2, std2 = current_median_std(list(self.history["burst_anomaly"]), w=self.NORM_WIN, min_fill=5)

        T1 = max(med1 + self.THRESH_SCALE * std1, thr_floor)
        T2 = max(med2 + self.THRESH_SCALE * std2, thr_floor)

        if (burst_anomaly_score1 > T1) or (burst_anomaly > T2):
            combined = burst_anomaly_score1 + burst_anomaly
        else:
            combined = min(burst_anomaly_score1, burst_anomaly) + 0.25 * max(burst_anomaly_score1, burst_anomaly)
        med_comb, std_comb = current_median_std(list(self.history["combined_anomaly"]), w=self.NORM_WIN, min_fill=5)
        thresh = max(med_comb + self.THRESH_SCALE * std_comb, thr_floor)
        is_anomaly = combined > thresh


        self.detector_state["recent_h_buffer"] = recent_h_buffer
        self.detector_state["recent_h_buffer"].append(h.detach().cpu())
        self.detector_state["prev_h_repr"] = h.detach()
        self.detector_state["prev_sf_h"] = sf_h
        self.detector_state["bscore_prev"] = b_score
        self.detector_state["prev_centroid"] = centroid

        self.history["anomaly_score_twotrack"].append(anomaly_score_twotrack)
        self.history["anomaly_score1"].append(anomaly_score1)

        self.history["burst_anomaly"].append(burst_anomaly)
        self.history["burst_anomaly_score1"].append(burst_anomaly_score1)
        self.history["combined_anomaly"].append(combined)

        details = {
            "score": float(combined),
            "threshold": float(thresh),
            "current": float(combined),
            "mean": float(med_comb),
            "std": float(std_comb),
            "method": "combined",
            "burst_anomaly_score1": float(burst_anomaly_score1),
            "burst_anomaly": float(burst_anomaly),
            "anomaly_score1": float(anomaly_score1),
            "anomaly_score_twotrack": float(anomaly_score_twotrack),
        }

        return is_anomaly, details


    def safe_filename(self, tok: str) -> str:
        return tok.replace("/", "_").replace("<", "").replace(">", "").replace("|", "_").replace("\\", "_")
    
    
    def is_anomalous_attention(self, attention_to_previous, current_attention):
        if attention_to_previous.numel() == 0:
            return False, {}
        
        attention_values = attention_to_previous.cpu().numpy().tolist()
        mean_attention = np.mean(attention_values)
        
        
        details = {
            'mean': mean_attention,
            'std': np.std(attention_values) if len(attention_values) > 1 else 0,
            'current': current_attention,
            'method': self.anomaly_method,
            "score":0.0
        }
        
        is_anomaly = self.detect_anomaly_original(attention_values, current_attention, mean_attention)
        
        return is_anomaly, details
    
    def decode(self, inputs, gen_config=None):
 
        torch.cuda.empty_cache()
        
        inputs = {k: v.to(self.model_target.device) for k, v in inputs.items()}
        if inputs['input_ids'].dim() == 1:
            input_ids = inputs['input_ids'].unsqueeze(0)
        else:
            input_ids = inputs['input_ids']


        
        
        generated_text_ids = input_ids.clone()


        score_by_pos = [0.0] * int(generated_text_ids.shape[1])   # prompt positions = 0
        # if self.plot_debug:
        #     if os.path.exists(self.plot_save_dir):
        #         shutil.rmtree(self.plot_save_dir)
        #     os.makedirs(self.plot_save_dir, exist_ok=True)
        
        count = 0
        reference_count = 0
        model_target_count = 0
        model_ref_count = 0
        start_idx=0 #AD2026
        
        self.model_target.eval()
        self.model_ref.eval()

        start_time = time.time()
        response_start = input_ids.shape[1]
        self.prompt_h_mean = compute_prompt_hidden_mean(
            self.model_target,
            generated_text_ids[:, :response_start]
        )
        
        with torch.no_grad():
            for i in range(self.m):
                if (count != 0) and (count % self.k == 0):
                    count = 0
                    
                    for iteration in range(self.max_iterations):
                        
                        outputs_check = self.model_target(
                            generated_text_ids,
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        model_target_count += 1
                        
                        mean_attention_matrix = outputs_check.attentions[-1].mean(dim=1).squeeze(0)
                        max_attention_matrix = outputs_check.attentions[-1].max(dim=1).values.squeeze(0) #AD2026
                        
                        replacement_made = False
                        
                         # Check the last k tokens
                        for guess in range(self.k):

                            token_pos = generated_text_ids.shape[1] - self.k + guess
                            if token_pos < response_start: # if token_position belong to the prompt skip echong and go to the next
                                continue ###############
             

                            window_start = max(start_idx, token_pos +1 - self.window_size) #AD2026
                            attention_matrix=max_attention_matrix #max_over_head or mean_over head #AD2026
                            attention_window = attention_matrix[token_pos, window_start:token_pos+1] #AD2026
                            attention_to_previous = attention_window[:-1] #AD2026
                            current_attention = attention_window[-1].item() #AD2026
                            
                            token_id = generated_text_ids[0, token_pos].item()
                            token_text = self.tokenizer.decode([token_id])
                            
                           
                            is_anomaly, details = self.compute_combined_anomaly(out=outputs_check,
                                                                generated_text_ids=generated_text_ids,
                                                                token_pos=token_pos,
                                                                prompt_len=response_start
                                                            )


                            while len(score_by_pos) < int(generated_text_ids.shape[1]):
                                score_by_pos.append(0.0)

                            score_by_pos[token_pos] = details["score"]

                            
                            if is_anomaly:
                                generated_text_ids = generated_text_ids[:, :token_pos]

                                reference_count += 1
                                replacement_made = True
                                
     
                                outputs_replacement = self.model_ref(generated_text_ids, use_cache=False)
                                model_ref_count += 1
                                logits_replacement = outputs_replacement.logits

                               
                                replacement_token = torch.argmax(logits_replacement[0, -1, :]).unsqueeze(0)
                                replacement_text = self.tokenizer.decode(replacement_token.item())

                                generated_text_ids = torch.cat([generated_text_ids, replacement_token.unsqueeze(0)], dim=-1)
                                
                                break  
                        
                        del mean_attention_matrix, max_attention_matrix, outputs_check
                        torch.cuda.empty_cache()
                        
                        if not replacement_made:
                            break
                        

                outputs_target = self.model_target(generated_text_ids, use_cache=False)
                model_target_count += 1
                
                logits_target = outputs_target.logits
                nexttoken_logits_target = logits_target[0, -1, :].detach()
                
                del outputs_target, logits_target
                torch.cuda.empty_cache()
                
                probs_target = torch.softmax(nexttoken_logits_target, dim=-1)
                topk_token_target = torch.topk(probs_target, 10)
                topk_values_target = topk_token_target.values
                topk_indices_target = topk_token_target.indices
                next_token = topk_indices_target[0].unsqueeze(0)
                
                count += 1
                generated_text_ids = torch.cat([generated_text_ids, next_token.unsqueeze(0)], dim=-1)
                
                del nexttoken_logits_target, probs_target, topk_token_target
                del topk_values_target, topk_indices_target
                
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                generated_len = generated_text_ids.shape[1] - input_ids.shape[1]
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                if generated_len >= self.max_length:
                    partial_text = self.tokenizer.decode(
                        generated_text_ids[0][input_ids.shape[1]:],
                        skip_special_tokens=True
                    ).strip()

                    if partial_text.endswith((".", "!", "?")):
                        break

                    if generated_len >= self.max_length + self.finish_buffer:
                        break

        end_time = time.time()
        generated_sequence = generated_text_ids[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_sequence)
        
        total_model_calls = model_target_count + model_ref_count
        
        logging.info(generated_text)

        torch.cuda.empty_cache()
        
        return generated_text


    def no_defense_baseline(self, inputs):
        torch.cuda.empty_cache()
        
        inputs = {k: v.to(self.model_target.device) for k, v in inputs.items()}
        if inputs['input_ids'].dim() == 2:
            input_ids = inputs['input_ids']
        elif inputs['input_ids'].dim() == 1:
            input_ids = inputs['input_ids'].unsqueeze(0)
        generated_text_ids = input_ids
        self.model_target.eval()
        
        start_time = time.time()
        with torch.no_grad():
            for i in range(self.max_length):
                outputs = self.model_target(generated_text_ids)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated_text_ids = torch.cat([generated_text_ids, next_token], dim=-1)
                
                del outputs, logits, next_token_logits
                
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
        end_time = time.time()
        generated_sequence = generated_text_ids[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_sequence)
        logging.info(generated_text)

        torch.cuda.empty_cache()
        
        return generated_text, 0, average_time
