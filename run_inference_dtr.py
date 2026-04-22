import math
import argparse
import json
import os
import sys
import types
import gc
from typing import Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, get_model_name_from_path
from videollava.model.builder import load_pretrained_model

DATASETS = {
    'obj_rel': {
        'json_path': 'object_relation/object_relation.json',
        'video_dir': 'object_relation/videos'
    },
    'temporal': {
        'json_path': 'temporal/temporal.json',
        'video_dir': 'temporal/videos'
    },
    'semantic': {
        'json_path': 'semantic_detail/semantic_detail.json',
        'video_dir': 'semantic_detail/videos'
    },
    'interaction': {
        'json_path': 'interaction/interaction.json',
        'video_dir': 'interaction/videos'
    },
    'fact': {
        'json_path': 'external_factual/external_factual.json',
        'video_dir': 'external_factual/videos'
    },
    'nonfact': {
        'json_path': 'external_nonfactual/external_nonfactual.json',
        'video_dir': 'external_nonfactual/videos'
    },
    'factdet': {
        'json_path': 'fact_detect/fact_detect.json',
        'video_dir': 'fact_detect/videos'
    }
}


class FrameAttentionModifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.frame_token_ranges = {}
            cls._instance.num_frames = 0
            cls._instance.enabled = False
            cls._instance.start_layer = 0
            cls._instance.end_layer = 999
            cls._instance.video_token_start_pos = 0
            cls._instance.video_token_end_pos = 0
            cls._instance._token_to_frame_map = None
            cls._instance._valid_frame_ranges_cache = None
        return cls._instance

    @classmethod
    def set_num_frames(cls, num_frames: int):
        if cls._instance is None:
            cls()
        cls._instance.num_frames = num_frames
        cls._instance._token_to_frame_map = None
        cls._instance._valid_frame_ranges_cache = None

    @classmethod
    def set_frame_token_range(cls, frame_idx: int, start_idx: int, end_idx: int):
        if cls._instance is None:
            cls()
        cls._instance.frame_token_ranges[frame_idx] = (start_idx, end_idx)
        cls._instance._token_to_frame_map = None
        cls._instance._valid_frame_ranges_cache = None

    @classmethod
    def _build_token_to_frame_map(cls):
        if cls._instance is None:
            return {}
        if cls._instance._token_to_frame_map is None:
            token_map = {}
            for frame_idx, (start_idx, end_idx) in cls._instance.frame_token_ranges.items():
                for token_idx in range(start_idx, end_idx):
                    token_map[token_idx] = frame_idx
            cls._instance._token_to_frame_map = token_map
        return cls._instance._token_to_frame_map

    @classmethod
    def get_valid_frame_ranges(cls, visual_len: int):
        if cls._instance is None:
            return []
        if cls._instance._valid_frame_ranges_cache is None:
            valid_ranges = []
            for frame_idx in range(cls._instance.num_frames):
                start_idx, end_idx = cls._instance.frame_token_ranges.get(frame_idx, (0, 0))
                start_idx = max(0, min(start_idx, visual_len))
                end_idx = max(0, min(end_idx, visual_len))
                if end_idx > start_idx:
                    valid_ranges.append((frame_idx, start_idx, end_idx))
            cls._instance._valid_frame_ranges_cache = valid_ranges
        return cls._instance._valid_frame_ranges_cache

    @classmethod
    def get_num_frames(cls) -> int:
        if cls._instance is None:
            return 0
        return cls._instance.num_frames

    @classmethod
    def enable(cls):
        if cls._instance is None:
            cls()
        cls._instance.enabled = True

    @classmethod
    def disable(cls):
        if cls._instance is None:
            cls()
        cls._instance.enabled = False
        cls._instance.frame_token_ranges = {}
        cls._instance.num_frames = 0
        cls._instance._token_to_frame_map = None
        cls._instance._valid_frame_ranges_cache = None

    @classmethod
    def is_enabled(cls) -> bool:
        if cls._instance is None:
            return False
        return cls._instance.enabled

    @classmethod
    def set_layer_range(cls, start_layer: int, end_layer: int):
        if cls._instance is None:
            cls()
        cls._instance.start_layer = start_layer
        cls._instance.end_layer = end_layer

    @classmethod
    def should_modify_layer(cls, layer_idx: int) -> bool:
        if cls._instance is None:
            return False
        return cls._instance.start_layer <= layer_idx <= cls._instance.end_layer

    @classmethod
    def set_video_token_start_pos(cls, start_pos: int):
        if cls._instance is None:
            cls()
        cls._instance.video_token_start_pos = start_pos

    @classmethod
    def get_video_token_start_pos(cls) -> int:
        if cls._instance is None:
            return 0
        return cls._instance.video_token_start_pos

    @classmethod
    def set_video_token_end_pos(cls, end_pos: int):
        if cls._instance is None:
            cls()
        cls._instance.video_token_end_pos = end_pos

    @classmethod
    def get_video_token_end_pos(cls) -> int:
        if cls._instance is None:
            return 0
        return cls._instance.video_token_end_pos


class GlobalRebalanceParams:
    _instance = None
    alpha = 0.5
    beta = 0.4
    eps = 1e-6

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_alpha(cls, alpha: float):
        if cls._instance is None:
            cls()
        cls._instance.alpha = alpha

    @classmethod
    def get_alpha(cls) -> float:
        if cls._instance is None:
            return 0.5
        return cls._instance.alpha

    @classmethod
    def set_beta(cls, beta: float):
        if cls._instance is None:
            cls()
        cls._instance.beta = beta

    @classmethod
    def get_beta(cls) -> float:
        if cls._instance is None:
            return 0.4
        return cls._instance.beta

    @classmethod
    def set_eps(cls, eps: float):
        if cls._instance is None:
            cls()
        cls._instance.eps = eps

    @classmethod
    def get_eps(cls) -> float:
        if cls._instance is None:
            return 1e-6
        return cls._instance.eps


from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def llama_new_forward_with_global_rebalance(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [nn.functional.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [nn.functional.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [nn.functional.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    use_frame_attn = getattr(self, "use_frame_attn", False)
    if use_frame_attn and FrameAttentionModifier.is_enabled():
        layer_idx = getattr(self, 'layer_idx', -1)
        visual_len = key_states.shape[-2]
        num_frames = FrameAttentionModifier.get_num_frames()

        if num_frames > 0 and FrameAttentionModifier.should_modify_layer(layer_idx):
            is_prefill = q_len > 1
            video_token_start_pos = FrameAttentionModifier.get_video_token_start_pos()
            video_token_end_pos = FrameAttentionModifier.get_video_token_end_pos()

            with torch.no_grad():
                if is_prefill:
                    start_token_idx = max(0, video_token_end_pos)
                    end_token_idx = q_len
                    if start_token_idx < end_token_idx:
                        avg_weights = attn_weights[:, :, start_token_idx:end_token_idx, video_token_start_pos:video_token_end_pos].mean(dim=(0, 1, 2))
                    else:
                        avg_weights = attn_weights[:, :, -1, video_token_start_pos:video_token_end_pos].mean(dim=(0, 1))
                else:
                    avg_weights = attn_weights[:, :, -1, video_token_start_pos:video_token_end_pos].mean(dim=(0, 1))

                valid_frame_ranges = FrameAttentionModifier.get_valid_frame_ranges(visual_len)
                frame_score_dict = {}
                for frame_idx, start_idx, end_idx in valid_frame_ranges:
                    frame_score_dict[frame_idx] = avg_weights[start_idx:end_idx].mean()

                if len(frame_score_dict) > 0:
                    frame_scores = torch.zeros(num_frames, dtype=attn_weights.dtype, device=attn_weights.device)
                    for frame_idx in range(num_frames):
                        frame_scores[frame_idx] = frame_score_dict.get(
                            frame_idx, torch.tensor(0.0, dtype=attn_weights.dtype, device=attn_weights.device)
                        )

                    alpha = GlobalRebalanceParams.get_alpha()
                    beta = GlobalRebalanceParams.get_beta()
                    eps = GlobalRebalanceParams.get_eps()

                    max_score = torch.max(frame_scores)
                    score_gap = max_score - frame_scores
                    norm_gap = score_gap / (torch.max(score_gap) + eps)
                    frame_bias = alpha + beta * norm_gap

                    for frame_idx, start_idx, end_idx in valid_frame_ranges:
                        bias_value = frame_bias[frame_idx].to(attn_weights.dtype)
                        attn_slice = attn_weights[:, :, -1, start_idx:end_idx]
                        attn_weights[:, :, -1, start_idx:end_idx] += bias_value * attn_slice.abs()

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([nn.functional.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def apply_global_rebalance_modification(model, start_layer, end_layer, alpha, beta):
    GlobalRebalanceParams.set_alpha(alpha)
    GlobalRebalanceParams.set_beta(beta)
    FrameAttentionModifier.set_layer_range(start_layer, end_layer)
    FrameAttentionModifier.enable()

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i in range(start_layer, min(end_layer + 1, len(model.model.layers))):
            layer = model.model.layers[i]
            if hasattr(layer, 'self_attn'):
                layer.self_attn.use_frame_attn = True
                layer.self_attn.layer_idx = i
                layer.self_attn.forward = types.MethodType(llama_new_forward_with_global_rebalance, layer.self_attn)

    return model


def load_model_with_global_rebalance(model_path, device, start_layer, end_layer, alpha, beta):
    cache_dir = 'cache_dir'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name,
        load_8bit=False,
        load_4bit=False,
        device=device,
        cache_dir=cache_dir
    )
    model = model.cuda()

    video_processor = processor['video']
    video_tower = model.get_model().get_video_tower()

    model = apply_global_rebalance_modification(
        model,
        start_layer=start_layer,
        end_layer=end_layer,
        alpha=alpha,
        beta=beta
    )

    original_prepare = model.prepare_inputs_labels_for_multimodal

    def hook_prepare_inputs(*args_prepare, **kwargs_prepare):
        result = original_prepare(*args_prepare, **kwargs_prepare)

        new_input_embeds = result[4] if len(result) > 4 else None

        if new_input_embeds is not None:
            input_ids = kwargs_prepare.get('input_ids', None)
            if input_ids is None and len(args_prepare) > 0:
                input_ids = args_prepare[0]

            if input_ids is not None:
                video_token_indices = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].tolist()

                if len(video_token_indices) > 0:
                    num_patches = video_tower.num_patches + 1
                    num_frames_detected = len(video_token_indices)
                    FrameAttentionModifier.set_num_frames(num_frames_detected)

                    current_embed_pos = 0
                    frame_counter = 0
                    video_token_embed_start = None
                    video_token_embed_end = None

                    for token_id in input_ids[0]:
                        if token_id.item() == IMAGE_TOKEN_INDEX:
                            if video_token_embed_start is None:
                                video_token_embed_start = current_embed_pos

                            start_pos = current_embed_pos
                            end_pos = start_pos + num_patches
                            FrameAttentionModifier.set_frame_token_range(frame_counter, start_pos, end_pos)

                            current_embed_pos = end_pos
                            video_token_embed_end = current_embed_pos
                            frame_counter += 1
                        else:
                            current_embed_pos += 1

                    if len(video_token_indices) > 0:
                        video_token_start_pos = video_token_embed_start if video_token_embed_start is not None else 0
                        video_token_end_pos = video_token_embed_end if video_token_embed_end is not None else current_embed_pos
                        FrameAttentionModifier.set_video_token_start_pos(video_token_start_pos)
                        FrameAttentionModifier.set_video_token_end_pos(video_token_end_pos)

        return result

    model.prepare_inputs_labels_for_multimodal = hook_prepare_inputs

    return model, tokenizer, video_processor


def get_model_output(model, tokenizer, video_processor, video_path, question, conv_mode="llava_v1"):
    conv = conv_templates[conv_mode].copy()

    video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    num_frames = model.get_video_tower().config.num_frames
    instruction = ''.join([DEFAULT_IMAGE_TOKEN] * num_frames) + '\n' + question

    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=10,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs


def run_inference(model, tokenizer, video_processor, qa_path, qa_type, video_dir,
                  output_dir_path, skip_existing=True, conv_mode="llava_v1"):
    output_path = os.path.join(output_dir_path, f"{qa_type}_predictions.json")

    if skip_existing and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            return json.load(f)

    paired_qas = json.load(open(qa_path))

    for qa_dct in tqdm(paired_qas):
        FrameAttentionModifier.disable()
        FrameAttentionModifier.enable()

        basic = qa_dct["basic"]
        basic_question = basic["question"]
        basic_question = f"{basic_question}\nAnswer the question using 'yes' or 'no'."
        basic_video_path = os.path.join(video_dir, basic["video"])

        if not os.path.exists(basic_video_path):
            basic_predict = "Video not found"
        else:
            basic_predict = get_model_output(
                model, tokenizer, video_processor,
                basic_video_path, basic_question, conv_mode
            )

        qa_dct["basic"]["predict"] = basic_predict

        halluc = qa_dct["hallucination"]
        halluc_question = halluc["question"]
        halluc_question = f"{halluc_question}\nAnswer the question using 'yes' or 'no'."
        halluc_video_path = os.path.join(video_dir, halluc["video"])

        if not os.path.exists(halluc_video_path):
            halluc_predict = "Video not found"
        else:
            halluc_predict = get_model_output(
                model, tokenizer, video_processor,
                halluc_video_path, halluc_question, conv_mode
            )

        qa_dct["hallucination"]["predict"] = halluc_predict

        gc.collect()
        torch.cuda.empty_cache()

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(output_path, "w") as jp:
        json.dump(paired_qas, jp, indent=4)

    return paired_qas


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run inference on VideoHallucer using VideoLLaVA with DTR'
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the VideoLLaVA model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of VideoHallucer dataset")
    parser.add_argument("--output_dir_path", type=str, required=True,
                        help="Directory to save inference results")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID")

    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Shared Adjustment Strength')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Deficit Compensation Strength')
    parser.add_argument('--start_layer', type=int, default=18,
                        help='Starting layer index for DTR')
    parser.add_argument('--end_layer', type=int, default=31,
                        help='Ending layer index for DTR')

    parser.add_argument("--eval_obj_rel", default=True)
    parser.add_argument("--eval_temporal", default=True)
    parser.add_argument("--eval_semantic", default=True)
    parser.add_argument("--eval_interaction", default=False)
    parser.add_argument("--eval_fact", default=True)
    parser.add_argument("--eval_nonfact", default=True)
    parser.add_argument("--detect_fact", default=False)

    parser.add_argument('--skip_existing', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer, video_processor = load_model_with_global_rebalance(
        model_path=args.model_path,
        device=args.device,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        alpha=args.alpha,
        beta=args.beta
    )

    if args.eval_obj_rel:
        run_inference(
            model=model, tokenizer=tokenizer, video_processor=video_processor,
            qa_path=os.path.join(args.data_dir, DATASETS['obj_rel']['json_path']),
            qa_type='obj_rel',
            video_dir=os.path.join(args.data_dir, DATASETS['obj_rel']['video_dir']),
            output_dir_path=args.output_dir_path,
            skip_existing=args.skip_existing,
        )

    if args.eval_temporal:
        run_inference(
            model=model, tokenizer=tokenizer, video_processor=video_processor,
            qa_path=os.path.join(args.data_dir, DATASETS['temporal']['json_path']),
            qa_type='temporal',
            video_dir=os.path.join(args.data_dir, DATASETS['temporal']['video_dir']),
            output_dir_path=args.output_dir_path,
            skip_existing=args.skip_existing,
        )

    if args.eval_semantic:
        run_inference(
            model=model, tokenizer=tokenizer, video_processor=video_processor,
            qa_path=os.path.join(args.data_dir, DATASETS['semantic']['json_path']),
            qa_type='semantic',
            video_dir=os.path.join(args.data_dir, DATASETS['semantic']['video_dir']),
            output_dir_path=args.output_dir_path,
            skip_existing=args.skip_existing,
        )

    if args.eval_interaction:
        run_inference(
            model=model, tokenizer=tokenizer, video_processor=video_processor,
            qa_path=os.path.join(args.data_dir, DATASETS['interaction']['json_path']),
            qa_type='interaction',
            video_dir=os.path.join(args.data_dir, DATASETS['interaction']['video_dir']),
            output_dir_path=args.output_dir_path,
            skip_existing=args.skip_existing,
        )

    if args.eval_fact:
        run_inference(
            model=model, tokenizer=tokenizer, video_processor=video_processor,
            qa_path=os.path.join(args.data_dir, DATASETS['fact']['json_path']),
            qa_type='fact',
            video_dir=os.path.join(args.data_dir, DATASETS['fact']['video_dir']),
            output_dir_path=args.output_dir_path,
            skip_existing=args.skip_existing,
        )

    if args.eval_nonfact:
        run_inference(
            model=model, tokenizer=tokenizer, video_processor=video_processor,
            qa_path=os.path.join(args.data_dir, DATASETS['nonfact']['json_path']),
            qa_type='nonfact',
            video_dir=os.path.join(args.data_dir, DATASETS['nonfact']['video_dir']),
            output_dir_path=args.output_dir_path,
            skip_existing=args.skip_existing,
        )

    if args.detect_fact:
        run_inference(
            model=model, tokenizer=tokenizer, video_processor=video_processor,
            qa_path=os.path.join(args.data_dir, DATASETS['factdet']['json_path']),
            qa_type='factdet',
            video_dir=os.path.join(args.data_dir, DATASETS['factdet']['video_dir']),
            output_dir_path=args.output_dir_path,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
