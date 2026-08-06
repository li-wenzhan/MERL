# conda activate  /mnt/shared-storage-user/zhouheng/miniconda3/envs/vla

python /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/scripts/plot_log_metrics.py \
 --log_file /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/checkpoints/MeRL/train_Openvla-oft-SFT-libero_10-debug-2gpu-512x-loss_w-pro_0130/run_20260130_212414.log \
 --keys train_reward/verifier train_reward/reward_all actor/pg_loss actor_after/entropy_loss_eval actor/ppo_kl val/test_score/all \
 --save_path /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/checkpoints/MeRL/train_Openvla-oft-SFT-libero_10-debug-2gpu-512x-loss_w-pro_0130/run_20260130_212414_actor.png

python /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/scripts/plot_log_metrics.py \
 --log_file /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/checkpoints/MeRL/train_Openvla-oft-SFT-libero_10-debug-2gpu-512x-loss_w-pro_0130/run_20260130_212414.log \
 --keys wm/ratio_real wm/ratio_wm wm/loss wm/loss_ema \
 --save_path /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/checkpoints/MeRL/train_Openvla-oft-SFT-libero_10-debug-2gpu-512x-loss_w-pro_0130/run_20260130_212414_wm.png

python /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/scripts/plot_log_metrics.py \
 --log_file /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/checkpoints/MeRL/train_Openvla-oft-SFT-libero_10-debug-2gpu-512x-loss_w-pro_0130/run_20260130_212414.log \
 --keys wm/loss wm/loss_ema \
 --save_path /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/checkpoints/MeRL/train_Openvla-oft-SFT-libero_10-debug-2gpu-512x-loss_w-pro_0130/run_20260130_212414_wm_loss.png

# python /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/MeRL_new/scripts/plot_log_metrics.py \
#  --log_file /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/SimpleVLA-RL/model_ckpts/SimpleVLA-RL/eval_Openvla-oft-SFT-libero10-traj-all-1205/run_20251205_011843.log \
#  --keys train_reward/verifier train_reward/reward_all actor/pg_loss actor_after/entropy_loss_eval actor/ppo_kl \
#  --save_path /mnt/shared-storage-user/zhouheng/wenzhan/projects/VLA-RL/SimpleVLA-RL/model_ckpts/SimpleVLA-RL/eval_Openvla-oft-SFT-libero10-traj-all-1205/run_20251205_011843.png
