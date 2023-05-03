export CUDA_VISIBLE_DEVICES=4
python main.py \
--seed 1 \
--drop_out \
--early_stopping \
--lr 1e-5 \
--k_start 0 \
--k 5 \
--label_frac 1 \
--bag_loss ce \
--task task_2_tumor_subtyping \
--results_dir results/TCGA_RCC \
--exp_code task_2_tumor_subtyping_SCMF_cls3 \
--model_type patch_gcn \
--mode graph \
--log_data \
--data_root_dir ../our_work_yfy/DATA_ROOT_DIR/TCGA_RCC \
--data_folder_s kidney_subtyping_20x \
--data_folder_l kidney_subtyping_10x \
--tg_file tissue_graph_files_slic \
--split_dir RCC/task_2_tumor_subtyping_100 \

python main.py \
--seed 2 \
--drop_out \
--early_stopping \
--lr 1e-5 \
--k_start 0 \
--k 5 \
--label_frac 1 \
--bag_loss ce \
--task task_2_tumor_subtyping \
--results_dir results/TCGA_RCC \
--exp_code task_2_tumor_subtyping_SCMF_cls3 \
--model_type patch_gcn \
--mode graph \
--log_data \
--data_root_dir ../our_work_yfy/DATA_ROOT_DIR/TCGA_RCC \
--data_folder_s kidney_subtyping_20x \
--data_folder_l kidney_subtyping_10x \
--tg_file tissue_graph_files_slic \
--split_dir RCC/task_2_tumor_subtyping_100
