path=$1
epoch=$2
a=$3

#gen_wrong=$3
#imp_corr=$4
#imp_wrong=$5

##### compute EER ################


#cd /home/liu/Desktop/kaldi/egs/rsr_system_transfree/male
#cd /home/liu/Desktop/kaldi/egs/rsr_system_transfree/male/scores/dev_p1/combined_scores
cd $path
#cd scores/dev_p1/combined_scores
rm imp_c_trials_$epoch\_$a imp_w_trials_$epoch\_$a gen_w_trials_$epoch\_$a
  
cat epoch${epoch}gen_c_perf_trials$a epoch${epoch}imp_c_perf_trials$a > imp_c_trials_$epoch\_$a
cat epoch${epoch}gen_c_perf_trials$a epoch${epoch}imp_w_perf_trials$a > imp_w_trials_$epoch\_$a
cat epoch${epoch}gen_c_perf_trials$a epoch${epoch}gen_w_perf_trials$a > gen_w_trials_$epoch\_$a
compute-eer gen_w_trials_$epoch\_$a
echo ' '

compute-eer imp_c_trials_$epoch\_$a
echo ' '
compute-eer imp_w_trials_$epoch\_$a
echo ' '





# cat gen_corr_kaldi imp_corr_kaldi >imp_corr_perf
# compute-eer imp_corr_perf 


# cat gen_corr_kaldi gen_wrong_kaldi >gen_wrong_perf
# compute-eer gen_wrong_perf


# cat gen_corr_kaldi imp_wrong_kaldi >imp_wrong_perf
# compute-eer imp_wrong_perf
