args=(
	--batch-size 11
	-j 6 
	--epochs 1200 
	--lr 0.0007
	--seed 777 
	--opt-level O0 
	-p 10
	--alp 0.01
	#--pretrained 
	#--distributed 
	--save /DIRECTION/TO/CHECKPOINT/FOLDER
	-train /DIRECTION/TO/TRAIN_DATA_FOLDER/refined_mpi_train.json 
	-val /DIRECTION/TO/VAL_DATA_FOLDER/refined_mpi_val.json 
	-img /DIRECTION/TO/IMAGE_DATA_FOLDER
	)
mkdir /DIRECTION/TO/CHECKPOINT/FOLDER 
CUDA_VISIBLE_DEVICES=0 python main.py "${args[@]}" 2>&1 |tee -a log.txt
