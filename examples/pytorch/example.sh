set -ex

# 原始ckpt
MODEL="/path/to/Qwen2.5-VL-7B-Instruct/"
# 量化后ckpt
FP8_MODEL="/path/to/TensorRT-Model-Optimizer/examples/vlm_ptq/saved_models_Qwen2_5-VL-7B-Instruct_dense_fp8_tp1_pp1_hf"
# 修改KV cache 整体精度：在 ${FP8_MODEL}/hf_quant_config.json
#     "kv_cache_quant_algo": “FP8”  # FP8
#     "kv_cache_quant_algo": null   # BF16/FP16
# KV cache 混合精度：修改 tensorrt_llm/_torch/models/modeling_utils.py:494 的判断条件


# 运行并打印输出
# 可在脚本中修改 example_image 和 example_image_prompts，从而指定输入
MAX_OUTPUT_TOKENS=120
python ./quickstart_multimodal.py --max_tokens ${MAX_OUTPUT_TOKENS} --model_dir $MODEL --use_cuda_graph
python ./quickstart_multimodal.py --max_tokens ${MAX_OUTPUT_TOKENS} --model_dir $FP8_MODEL --use_cuda_graph
