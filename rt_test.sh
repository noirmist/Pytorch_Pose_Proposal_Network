args=(
	--resume ./best_weight/PPN_model_best.pth.tar
	)

CUDA_VISIBLE_DEVICES=0 python rt_test.py "${args[@]}" 2>&1 |tee -a 1213_webcam_test2.log
