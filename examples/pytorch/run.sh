set -ex

MODEL="/llm-models/Qwen2.5-VL-3B-Instruct/"
export TLLM_MULTIMODAL_DISAGGREGATED=0
export TLLM_WORKER_USE_SINGLE_PROCESS=1

MT=300
ISL=600
MSL=$((ISL + MT))
BS=128
MNT=$((BS * MSL))

python ./quickstart_multimodal.py --max_tokens ${MT} --model_dir $MODEL --tp_size 1 --max_batch_size $BS --use_cuda_graph
