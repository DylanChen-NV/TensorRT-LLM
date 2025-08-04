set -ex
#export PYTHONPATH="/path/to/TensorRT-LLM/:$PYTHONPATH"
MODEL="/home/scratch.trt_llm_data/llm-models/DeepSeek-V3-Lite/nvfp4_moe_only"

# origin
#[0] Prompt: 'Hello, my name is', Generated text: ' {name} and I am {age} years old.")\r\n\r\n# 3. Write a Python program to check if a number is even'
#[1] Prompt: 'The president of the United States is', Generated text: ' the head of state and head of government of the United States, indirectly elected to a four-year term by the American people through the Electoral College. The officeholder'
#[2] Prompt: 'The capital of France is', Generated text: ' Paris. It is located in the north-central part of the country and is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum,'
#[3] Prompt: 'The future of AI is', Generated text: ' bright, and it is only a matter of time before it becomes an integral part of our lives. As AI continues to evolve, it will become more sophisticated and'

export FLASH_DECODING=1
#export CUDA_LAUNCH_BLOCKING=1
#export TLLM_LOG_LEVEL="DEBUG"
#export TLLM_WORKER_USE_SINGLE_PROCESS=1
MAX_TOKENS=16
WORLD_SIZE=2
python quickstart_advanced.py --max_batch_size 1 --max_tokens ${MAX_TOKENS} --model_dir ${MODEL} --moe_ep_size $WORLD_SIZE --tp_size $WORLD_SIZE
