  source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
  conda activate /scratch/lg154/conda-envs/gr00t


# eval of baseline_s30000 DONE

  CKPT_DIR=$PWD/checkpoints/gr00t_libero_baseline_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=baseline_30k \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh

  # DONE 

  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_s2_tune_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=geo_s2_tune_30k \
  USE_CAMERA_PARAMS=true \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh

  CKPT_DIR=$PWD/checkpoints/gr00t_libero_baseline_2gpu_b16/checkpoint-26000 \
  MODEL_NAME=baseline_26k \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh


  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_s2_tune_2gpu_b16/checkpoint-26000 \
  MODEL_NAME=geo_s2_tune_26k \
  USE_CAMERA_PARAMS=true \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh


  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_s2_tune_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=geo_s2_tune_30k \
  USE_CAMERA_PARAMS=true \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh

  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_s2_aux05_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=geo_s2_aux05_30k \
  USE_CAMERA_PARAMS=true NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh

  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_gtonly_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=geo_distill_gt_30k \
  USE_CAMERA_PARAMS=true \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh


  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=geo_distill_30k \
  USE_CAMERA_PARAMS=true \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh


  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_s2_gtonly_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=geo_s2_gt_30k \
  USE_CAMERA_PARAMS=true \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh


  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_s2_gtonly_lr1e4_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=geo_s2_gt_lr_30k \
  USE_CAMERA_PARAMS=true \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh

  CKPT_DIR=$PWD/checkpoints/gr00t_libero_geo_distill_s2_gtonly_re_2gpu_b16/checkpoint-30000 \
  MODEL_NAME=geo_s2_gt_re_30k \
  USE_CAMERA_PARAMS=true \
  NUM_TRIALS_PER_TASK=50 \
  bash scripts/sbatch_lg/submit_eval_libero_4suites.sh

  