# # GPT-2
# for model_type in "openai-community/gpt2" "dram-1_gpt2_public" "dram-2_gpt2-from-scratch" "dram-3_gpt2-LinearDRAMAttention" "dram-4_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention_no_fine-tuning" "dram-5_gpt2-DRAMAttention_adapted_from_LinearDRAMAttention+fine-tuning"
# GPT-2 XL
for model_type in "dram-gpt2-xl-DRAMAttention" # "openai-community/gpt2-xl" # "dram-gpt2-xl-LinearDRAMAttention"
do
    echo
    echo "Running model: $model_type"
    model_type_clean=$(echo $model_type | sed 's/\//__/g')
    echo "Running model: $model_type_clean"
    # if model type starts with dram add kwargs
    BATCH_SIZE=64
    MODEL_KWARGS="pretrained=$model_type,dtype=float,tokenizer=gpt2"
    if [[ $model_type == "dram"* ]]; then
        MODEL_KWARGS=$MODEL_KWARGS",use_tokenizer_fast=True,_commit_hash=\"\",batch_size_model=$BATCH_SIZE,additional_suffix=_1000_iters"
        filename=./$model_type_clean
        if [ ! -f $filename ]; then
            echo -e "{ \n \"model_path\": null \n}" > $filename
            echo written to filename: $filename
        fi    
    fi

    echo $MODEL_KWARGS 
    TASKS="wikitext,lambada_openai,hellaswag,arc_easy,arc_challenge,winogrande,xwinograd,piqa"
    
    python3 eval_gpt_ncs.py --model hf \
        --model_args $MODEL_KWARGS \
        --tasks $TASKS \
        --device "cuda" \
        --batch_size $BATCH_SIZE \
        --output_path ./results/results_eval \
        --trust_remote_code \
        --verbosity DEBUG \
	--gen_kwargs "max_gen_toks=1," \
        | tee logs/log_eval_$model_type_clean.log

done
