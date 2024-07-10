
# AVSBENCH - OBJECT 
python main_avss_resize.py --experiment_name "CAVP" --setup avss_binary --resize_flag --avsbench_split "all" --gpus 1 --batch_size 16 --lr 1e-3 --weight_decay 1e-4 --epochs 60 --wandb_mode disabled --num_workers 16

python main_avss_resize.py --experiment_name "CAVP" --setup avss_binary --resize_flag --avsbench_split "v1s" --gpus 1 --batch_size 16 --lr 1e-3 --weight_decay 1e-4 --epochs 60 --wandb_mode disabled --num_workers 16

python main_avss_resize.py --experiment_name "CAVP" --setup avss_binary --resize_flag --avsbench_split "v1m" --gpus 1 --batch_size 16 --lr 1e-3 --weight_decay 1e-4 --epochs 60 --wandb_mode disabled --num_workers 16

# AVSBENCH - SEMANTIC 
python main_avss.py --experiment_name "CAVP" --setup avss --gpus 1 --batch_size 16 --lr 1e-3 --weight_decay 1e-4 --epochs 80 --wandb_mode disabled --num_workers 16

# VPO - MONO
python main_vpo_mono.py --experiment_name "CAVP" --setup "vpo_ss" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

python main_vpo_mono.py --experiment_name "CAVP" --setup "vpo_ms" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

python main_vpo_mono.py --experiment_name "CAVP" --setup "vpo_msmi" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

# VPO - STEREO
python main_vpo_stereo.py --experiment_name "CAVP" --setup "vpo_ss" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

python main_vpo_stereo.py --experiment_name "CAVP" --setup "vpo_ms" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online

python main_vpo_stereo.py --experiment_name "CAVP" --setup "vpo_msmi" --gpus 2 --batch_size 8 --lr 5e-4 --weight_decay 5e-4 --epochs 80 --num_workers 16 --wandb_mode online