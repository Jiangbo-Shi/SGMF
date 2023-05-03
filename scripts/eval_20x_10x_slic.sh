export CUDA_VISIBLE_DEVICES=0
python eval.py \
--drop_out \
 --k 5 \
 --k_start 0 \
 --k_end 1 \
 --models_exp_code task_3_pt_staging_20x_10x_SAMF_new_cls3_s0 \
 --save_exp_code task_3_pt_staging_20x_10x_SAMF_new_cls3_s0 \
 --task task_3_pt_staging_cls3 \
 --model_type patch_gcn \
 --mode graph \
 --results_dir results/Yifuyuan \
 --splits_dir /home1/sjb/gastric_cancer/pt_staging/our_work_SAMF/splits/Yifuyuan/task_3_pt_staging_cls3_100 \
 --data_root_dir ../our_work_yfy/DATA_ROOT_DIR/Yifuyuan \
 --data_folder_s gastric_pt_staging_20x \
 --data_folder_l gastric_pt_staging_10x \
 --tg_file tissue_graph_files_slic \
