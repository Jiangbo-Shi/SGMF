export CUDA_VISIBLE_DEVICES=5
python main.py \
--seed 0 \
--drop_out \
--early_stopping \
--lr 1e-5 \
--k_start 0 \
--k_end 1 \
--k 5 \
--label_frac 1 \
--bag_loss ce \
--task task_3_pt_staging_cls3 \
--results_dir results/Yifuyuan \
--exp_code task_3_pt_staging_20x_10x_SAMF_5x_cls3 \
--model_type patch_gcn --mode graph \
--log_data --data_root_dir \
../our_work_yfy/DATA_ROOT_DIR/Yifuyuan \
--data_folder_s gastric_pt_staging_20x \
--data_folder_l gastric_pt_staging_10x \
--tg_file tissue_graph_files_5x \
--split_dir Yifuyuan/task_3_pt_staging_cls3_100 \


