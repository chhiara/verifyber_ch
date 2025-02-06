

#!/usr/bin/bash 
#docker run -v /raid/home/nilab/chiara/datasets/input_model/:/app/data/input_model/ -v /raid/home/nilab/chiara/local/:/app/src/local/  --rm -it --gpus device=3 verifyber --shm-size=256m

path_code_parent="/app/src/local/"

#use this if running with container
#path_code_parent="/home/chiara/local/"

repo_verifyber_path="${path_code_parent}/verifyber_ch/"
experiment_path="${repo_verifyber_path}/checkpoints/sdec_tr_anomaly_ilf/"

log_file_train="${experiment_path}/log_file_train.txt"
log_file_test="${experiment_path}/log_file_test.txt"


cd $repo_verifyber_path


#Train
#python main.py TR-ANOMALY -opt train --exp ${experiment_path}\
#              --with_gt &> $log_file_train 


#Test
best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-ilf-l_1/models/best_model_ep-300_score-0.778140.pth"
python main.py TR-ANOMALY -opt test --exp ${experiment_path}\
              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test 
              

