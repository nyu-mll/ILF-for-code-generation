#!/bin/bash

# Assumes that the Codegen checkpoints are stored in a directory
# named "checkpoints" that is a subdirectory of the current directory.

FEEDBACK_COLUMN="Feedback"
REFINEMENTS_COLUMN="Refinement"
INPUT_FILE="surge_annotations.jsonl"
LEARNING_RATE=5e-6
GRADIENT_ACCUMULATION_STEPS=32
NUM_OUTPUT_SAMPLES=30
NUM_EPOCHS=2
while getopts "i:f:r:n:l:g:o:e:d:" option; do
    case $option in
        i)  # File containing all Surge annotations. Feedback is in column named via option -f, and refinements are under "unedited_annotator_completion".
            INPUT_FILE=$OPTARG;;
        f)  # Name of feedback column
            FEEDBACK_COLUMN=$OPTARG;;
        r)  # Name of refinements column that will be created in all intermediate outputs.
            REFINEMENTS_COLUMN=$OPTARG;;
        n)  # Experiment name
            EXP_NAME=$OPTARG;;
        l)  # Learning rate
            LEARNING_RATE=$OPTARG;;
        g)  # Gradient accumulation steps. Determines effective batch size because the per-device train batch size is 1
            GRADIENT_ACCUMULATION_STEPS=$OPTARG;;
        o)  # Number of final MBPP samples to output from the final fine-tuned CodeGen-6B model.
            NUM_OUTPUT_SAMPLES=$OPTARG;;
        e)  # Number of epochs to train for.
            NUM_EPOCHS=$OPTARG;;
        d)  # Parent directory to save results in. Experiment results will be saved in a subdirectory of this directory named ${EXP_NAME}.
            PARENT_DIR=$OPTARG;;
        \?) # Invalid option
            echo "Error: Invalid option ${option}"
            exit;;
    esac
done

TRAIN_START_TASK_ID=111
TRAIN_END_TASK_ID=310 # inclusive
TRAIN_N=$(( $TRAIN_END_TASK_ID - $TRAIN_START_TASK_ID + 1 ))
VAL_START_TASK_ID=311
VAL_END_TASK_ID=974  # inclusive
VAL_N=$(( $VAL_END_TASK_ID - $VAL_START_TASK_ID + 1 ))
TEST_START_TASK_ID=11
TEST_END_TASK_ID=111 # (should be exclusive)

CONDA_ENV="ilf"
EXPERIMENT_DIR="${PARENT_DIR}/${EXP_NAME}"

echo "Running with arguments -i=${INPUT_FILE}, -f=${FEEDBACK_COLUMN}, -r=${REFINEMENTS_COLUMN}," \
    "-n=${EXP_NAME}, -l=${LEARNING_RATE}, -g=${GRADIENT_ACCUMULATION_STEPS}, -o=${NUM_OUTPUT_SAMPLES}," \
    "-e=${NUM_EPOCHS}, -d=${PARENT_DIR}."
echo "Outputting experiment results in ${EXPERIMENT_DIR}."

conda deactivate
conda activate ${CONDA_ENV}

mkdir -p ${EXPERIMENT_DIR}
python preprocess_feedback_spreadsheet.py --input_file=${INPUT_FILE} \
    --model_completion_column=original_model_completion \
    --old_refinement_column=unedited_annotator_completion \
    --training_n=$TRAIN_N --val_n=$VAL_N \
    --feedback_column=${FEEDBACK_COLUMN} --refinement_column=${REFINEMENTS_COLUMN} \
    --one_per_task --filter_for_correct --output_dir=${EXPERIMENT_DIR} \
    --training_start_id=${TRAIN_START_TASK_ID} --training_end_id=${TRAIN_END_TASK_ID} \
    --val_start_id=${VAL_START_TASK_ID} --val_end_id=${VAL_END_TASK_ID}  || exit
OUTPUT_FILE_PREFIX=$(python -c "print(''.join('${INPUT_FILE}'.split('.')[:-1]).split('/')[-1])")
OUTPUT_FILE_PREFIX=${EXPERIMENT_DIR}/${OUTPUT_FILE_PREFIX}
REF_TRAINING_FILE="${OUTPUT_FILE_PREFIX}-train.jsonl"
REF_VAL_FILE="${OUTPUT_FILE_PREFIX}-val.jsonl"

echo "Training data for Pi_Ref: ${REF_TRAINING_FILE}"
echo "Val data for Pi_Ref: ${REF_VAL_FILE}"

# Fine-tune a model to generate refinements.
# We trained with per-device batch size of 1 due to computational constraints 
# (but used gradient accumulation to reach the desired effective batch size).
PI_REF_DIR="${EXPERIMENT_DIR}/mref_lr${LEARNING_RATE}_ga${GRADIENT_ACCUMULATION_STEPS}_${NUM_EPOCHS}epochs"
CHECKPOINTS_DIR="$(pwd)/checkpoints"
python finetune_refinement_model.py  \
    --do_train \
    --codegen_model_dir=${CHECKPOINTS_DIR} \
    --model_name_or_path=codegen-6B \
    --num_train_epochs=${NUM_EPOCHS} \
    --save_strategy=no \
    --learning_rate=${LEARNING_RATE} \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --logging_steps=1 \
    --output_dir ${PI_REF_DIR} \
    --pad_to_max_length \
    --generation_max_length=512 \
    --max_seq_length=1024 \
    --max_answer_length=512 \
    --parallelize \
    --overwrite_output_dir \
    --save_total_limit=2 \
    --feedback_column=${FEEDBACK_COLUMN} \
    --refinement_column=${REFINEMENTS_COLUMN} \
    --train_file ${REF_TRAINING_FILE} || exit

# Generate refinements using Pi_Ref
python generate_refinements_codegen_finetuned.py \
    --arch=codegen-6B \
    --codegen-model-dir=${CHECKPOINTS_DIR} \
    --num-samples=${NUM_OUTPUT_SAMPLES} --output-dir=${PI_REF_DIR} \
    --temperature=0.8 --feedback-file=${REF_VAL_FILE} \
    --output-file-suffix=${EXP_NAME} \
    --model-path=${PI_REF_DIR} || exit

# Evaluate refinements generated for tasks in MBPP_Train, and 
# keep only the correct ones for training Pi_Theta
python eval_mbpp.py \
    --input-file=${PI_REF_DIR}/refinements_codegen-6B_temp0.8_${EXP_NAME}.jsonl \
    --k=1,10 || exit
python create_finetuning_data_from_refinements.py \
    --one-per-task \
    --refinement-file=${PI_REF_DIR}/refinements_codegen-6B_temp0.8_${EXP_NAME}.jsonl_results.jsonl \
    --output-dir=${PI_REF_DIR} \
    --output-file-suffix=surge_final || exit

# Fine-tune two separate models: 
#   1) fine-tuned on MBPP gold data, 
#   2) fine-tuned on Pi_Refine-generated refinements
TRAINING_FILE="${PI_REF_DIR}/finetuning_prompts_mbpp_refinements_surge_final.jsonl"
GOLD_TRAINING_FILE="${PI_REF_DIR}/finetuning_prompts_mbpp_gold_surge_final.jsonl"
# Fine-tune (1)
FINAL_GOLD_FINETUNE_DIR=${EXPERIMENT_DIR}/final_gold_finetune_lr${LEARNING_RATE}_ga${GRADIENT_ACCUMULATION_STEPS}_${NUM_EPOCHS}epochs
python finetune.py  \
    --codegen_repo=${CHECKPOINTS_DIR} \
    --do_train \
    --model_name_or_path=codegen-6B \
    --save_strategy=no \
    --num_train_epochs=${NUM_EPOCHS} \
    --learning_rate=${LEARNING_RATE} \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --logging_steps=1 \
    --output_dir ${FINAL_GOLD_FINETUNE_DIR} \
    --parallelize \
    --pad_to_max_length \
    --generation_max_length=512 \
    --max_seq_length=1024 \
    --max_answer_length=512 \
    --save_total_limit=2 \
    --parallelize \
    --prompt_column=finetuning_prompt \
    --completion_column=finetuning_completion \
    --overwrite_output_dir \
    --train_file ${GOLD_TRAINING_FILE} || exit
# Fine-tune (2)
FINAL_FINETUNE_DIR=${EXPERIMENT_DIR}/final_finetune_lr${LEARNING_RATE}_ga${GRADIENT_ACCUMULATION_STEPS}_${NUM_EPOCHS}epochs
python finetune.py  \
    --codegen_repo=${CHECKPOINTS_DIR} \
    --do_train \
    --model_name_or_path=codegen-6B \
    --save_strategy=no \
    --num_train_epochs=${NUM_EPOCHS} \
    --learning_rate=${LEARNING_RATE} \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --logging_steps=1 \
    --output_dir ${FINAL_FINETUNE_DIR} \
    --parallelize \
    --pad_to_max_length \
    --generation_max_length=512 \
    --max_seq_length=1024 \
    --max_answer_length=512 \
    --save_total_limit=2 \
    --parallelize \
    --prompt_column=finetuning_prompt \
    --completion_column=finetuning_completion \
    --overwrite_output_dir \
    --train_file ${TRAINING_FILE} || exit

# Evaluate models (1) and (2) on MBPP_Test
## First generate programs for MBPP_Test
python generate_code_for_mbpp.py \
    --codegen-model-dir=${CHECKPOINTS_DIR} \
    --num-samples=${NUM_OUTPUT_SAMPLES} \
    --output-dir=${FINAL_GOLD_FINETUNE_DIR} \
    --arch=codegen-6B \
    -n=1 \
    --temperature=0.8 \
    --debug -s ${TEST_START_TASK_ID} -e ${TEST_END_TASK_ID} \
    --model-path=${FINAL_GOLD_FINETUNE_DIR} || exit
python generate_code_for_mbpp.py \
    --codegen-model-dir=${CHECKPOINTS_DIR} \
    --num-samples=${NUM_OUTPUT_SAMPLES} \
    --output-dir=${FINAL_FINETUNE_DIR} \
    --arch=codegen-6B \
    -n=1 \
    --temperature=0.8 \
    --debug -s ${TEST_START_TASK_ID} -e ${TEST_END_TASK_ID} \
    --model-path=${FINAL_FINETUNE_DIR} || exit
## Now evaluate final generations
python eval_mbpp.py \
    --input-file=${FINAL_GOLD_FINETUNE_DIR}/samples_test_codegen-6B_1shot_temp0.8_${TEST_START_TASK_ID}-${TEST_END_TASK_ID}.jsonl \
    --k=1,10 || exit
python eval_mbpp.py \
    --input-file=${FINAL_FINETUNE_DIR}/samples_test_codegen-6B_1shot_temp0.8_${TEST_START_TASK_ID}-${TEST_END_TASK_ID}.jsonl \
    --k=1,10 || exit