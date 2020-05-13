#!/bin/bash

#Example script for running run_classifier.py


for EPOCHS in 10 ; do # 3 4 5; do 
	for LR in 5e-5; do  #2e-5 3e-5 5e-5; do
		for BATCH_SZ in 32; do #16 32 ; do
			MAX_SEQ_LEN=150

			DATA_DIR=/usr/local/share/cambia_nlp/ #Modify this to be the path to the MedNLI data
			OUTPUT_DIR=/home/mrbarnes/output/goc_distilbluebert/ #Modify this to be the path to your output directory
			CLINICAL_BERT_LOC=/home/mrbarnes/models/distilbluebert_train_output # /home/mrbarnes/bluebert #Modify this to be the path to the clinical BERT model

			echo $OUTPUT_DIR

			BERT_MODEL=clinical_bert # You can change this to biobert or bert-base-cased

			mkdir -p $OUTPUT_DIR

		  	python run_updated_classifier.py \
			  --data_dir=$DATA_DIR \
			  --bert_model=$BERT_MODEL \
			  --model_loc $CLINICAL_BERT_LOC \
			  --task_name mednli \
			  --do_train \
			  --do_eval \
			  --do_test \
			  --do_lower_case \
			  --output_dir=$OUTPUT_DIR  \
			  --num_train_epochs $EPOCHS \
			  --learning_rate $LR
			  --train_batch_size $BATCH_SZ \
			  --max_seq_length $MAX_SEQ_LEN \
			  --gradient_accumulation_steps 2 
		done
	done
done 

