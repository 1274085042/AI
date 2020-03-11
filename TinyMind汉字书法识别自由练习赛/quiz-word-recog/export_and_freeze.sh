TEMP_GRAPH_FILE=/tmp/inception_v3_inf_graph.pb
DATASET_DIR='../train/'
DATASET_NAME='quiz'
OUTPUT_GRAPH='./freezed.pb'
python3 export_inference_graph.py \
  --model_name=inception_v3 \
  --output_file=$TEMP_GRAPH_FILE \
  --dataset_dir=$DATASET_DIR \
  --dataset_name=$DATASET_NAME

python3 freeze_graph.py \
  --input_graph=$TEMP_GRAPH_FILE \
  --input_checkpoint=../train_ckpt/model.ckpt-37830 \
  --output_node_names=output,final_probs \
  --input_binary=True \
  --output_graph=$OUTPUT_GRAPH
  
cp ${DATASET_DIR%*/}/labels.txt ${OUTPUT_GRAPH%*.pb}.label
