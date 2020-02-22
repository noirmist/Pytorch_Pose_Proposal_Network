args=(
    --batch-size 1
    -j 6 
    --epochs 1200 
    --lr 0.0007
    --seed 777 
    --opt-level O0 
    -p 1
    --alp 0.01
    #--pretrained 
    #--parallel_ckpt
    --resume /DIRECTION/TO/CHECKPOINT/FOLDER/PPN_model_best.pth.tar
    --evaluate
	--savefig /DIRECTION/TO/TEST_RESULT_FOLDER 
    --save /DIRECTION/TO/CHECKPOINT/FOLDER
    -train /DIRECTION/TO/TRAIN_DATA_FOLDER/refined_mpi_train.json
    -val /DIRECTION/TO/VAL_DATA_FOLDER/mpi_val.json
    -img /DIRECTION/TO/IMAGE_DATA_FOLDER
    )

mkdir -p /DIRECTION/TO/TEST_RESULT_FOLDER 
CUDA_VISIBLE_DEVICES=0 python main.py "${args[@]}" 2>&1 |tee -a test_log.txt
