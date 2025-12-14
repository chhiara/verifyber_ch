

#!/usr/bin/bash 
#docker run -v /raid/home/nilab/chiara/datasets/input_model/:/app/data/input_model/ -v /raid/home/nilab/chiara/local/:/app/src/local/  --rm -it --gpus device=0  --shm-size=800m verifyber:latest
#conda run --no-capture-output -n verifyber bash cmd_run.sh

path_code_parent="/app/src/local/"

#use this if running with container
#path_code_parent="/home/chiara/local/"

repo_verifyber_path="${path_code_parent}/verifyber_ch/"
experiment_path="${repo_verifyber_path}/checkpoints/tr_anomaly/af_l/sdec_tr_anomaly_afl/"

log_file_train="${experiment_path}/log_file_train.txt"
log_file_test="${experiment_path}/log_file_test.txt"
log_file_test_allTum="${experiment_path}/log_file_test_AllTumArea.txt"
log_file_test_apss="${experiment_path}/log_file_test_Apss21.txt"
log_file_test_apssFiltTumConv="${experiment_path}/log_file_test_Apss21_FiltTum_TumCCConv.txt"
log_file_test_apssFiltTumConv_csd="${experiment_path}/log_file_test_ApssExt_FiltTum_TumCCConv_csd_afl.txt"
log_file_test_apssFiltTumConv_dti="${experiment_path}/log_file_test_ApssExt_FiltTum_TumCCConv_dti_afl.txt"




cd $repo_verifyber_path

#-----train and test with balanced samples: 40% positive, 30% nn negative, 30% random negative
#Train
#python -u main.py TR-ANOMALY -opt train --exp ${experiment_path}\
#              --with_gt &> $log_file_train 


#Test
#best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-af-l_0/models/best_model_ep-420_score-0.816775.pth"
#python main.py TR-ANOMALY -opt test --exp ${experiment_path}\
#              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test 
              

#Test on all the streamline in the perilesional area -NOT DONE
#best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-ilf-l_1/models/best_model_ep-300_score-0.778140.pth"
#config_allTum="${experiment_path}/config_testAllTumArea.txt"

#python main.py TR-ANOMALY -opt test --exp ${experiment_path} --config ${config_allTum}\
#              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test_allTum 
              


#Test on apss subjects in all streamlines of left hemisphere
#best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-af-l_0/models/best_model_ep-420_score-0.816775.pth"
#config_apss="${experiment_path}/config_testApss21.txt"

#python main.py TR-ANOMALY-SINGLE-SUBID -opt test --exp ${experiment_path} --config ${config_apss}\
#              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test_apss





#Test on apss subjects in the perilesional area of tum I estimated is moved for at least 5 mm; the tum used are not the original ones but are corrected to
#to be convex and connected  
#best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-af-l_0/models/best_model_ep-420_score-0.816775.pth"
#config_apss="${experiment_path}/config_testApss21_filtTum_CCConvTum.txt"

#python main.py TR-ANOMALY-SINGLE-SUBID -opt test --exp ${experiment_path} --config ${config_apss}\
#              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test_apssFiltTumConv
       





#Test on apss Ext subjects in the perilesional area of tum I estimated is moved for at least 5 mm; the tum used are not the original ones but are corrected to
#to be convex and connected  
#---test on AF_L  CSD TRACTO
#best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-af-l_0/models/best_model_ep-420_score-0.816775.pth"

#config_apss="${experiment_path}/config_testApssExt_csd_AF_L.txt"
#python main.py TR-ANOMALY-SINGLE-SUBID -opt test --exp ${experiment_path} --config ${config_apss}\
#              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test_apssFiltTumConv
       

#---test on AF_L  DTI TRACTO
#best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-af-l_0/models/best_model_ep-420_score-0.816775.pth"
#config_apss="${experiment_path}/config_testApssExt_dti_AF_L.txt"
#python main.py TR-ANOMALY-SINGLE-SUBID -opt test --exp ${experiment_path} --config ${config_apss}\
#              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test_apssFiltTumConv_dti
       


#---test on AF_R  DTI TRACTO
bundle="AF_R"
tracto="dti"
best_model_path="${repo_verifyber_path}/runs/sdec_nodropout_loss_nll-tracto-anomaly-af-l_0/models/best_model_ep-420_score-0.816775.pth"
config_apss="${experiment_path}/config_testApssExt_${tracto}_${bundle}.txt"
log_file_test_apssFiltTumConv_dti="${experiment_path}/log_file_test_ApssExt_FiltTum_TumCCConv_${tracto}_afr.txt"
python main.py TR-ANOMALY-SINGLE-SUBID -opt test --exp ${experiment_path} --config ${config_apss}\
              --with_gt  --weights ${best_model_path} --save_pred &> $log_file_test_apssFiltTumConv_dti
       
       