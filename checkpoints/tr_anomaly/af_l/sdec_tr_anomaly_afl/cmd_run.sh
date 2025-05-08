

#!/usr/bin/bash 
#docker run -v /raid/home/nilab/chiara/datasets/input_model/:/app/data/input_model/ -v /raid/home/nilab/chiara/local/:/app/src/local/  --rm -it --gpus device=3 verifyber --shm-size=256m
#conda run --no-capture-output -n verifyber bash cmd_run.sh

path_code_parent="/app/src/local/"

#use this if running with container
#path_code_parent="/home/chiara/local/"

repo_verifyber_path="${path_code_parent}/verifyber_ch/"
experiment_path="${repo_verifyber_path}/checkpoints/tr_anomaly/af_l/sdec_tr_anomaly_afl/"

log_file_train="${experiment_path}/log_file_train.txt"
log_file_test="${experiment_path}/log_file_test.txt"
log_file_test_allTum="${experiment_path}/log_file_test_AllTumArea.txt"



cd $repo_verifyber_path

#-----train and test with balanced samples: 40% positive, 30% nn negative, 30% random negative
#Train
python main.py TR-ANOMALY -opt train --exp ${experiment_path}\
              --with_gt &> $log_file_train 


#Test
#best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-pyt-l_1/models/best_model_ep-220_score-0.849185.pth"
#python main.py TR-ANOMALY -opt test --exp ${experiment_path}\
#              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test 
              

#Test on all the streamline in the perilesional area
#best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-ilf-l_1/models/best_model_ep-300_score-0.778140.pth"
#config_allTum="${experiment_path}/config_testAllTumArea.txt"

#python main.py TR-ANOMALY -opt test --exp ${experiment_path} --config ${config_allTum}\
#              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test_allTum 
              
