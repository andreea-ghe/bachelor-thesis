## https://wandb.ai/andreea-ghe-babes-bolyai-university/jigsaw/runs/hgkr6wya?nw=nwuserandreeaghe

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
initial: self.w_cls_loss 1.0, self.w_mat_loss 0.0, self.w_rig_loss 0.0
self.pc_cls_method: binary
Finish Setting -----
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/ndreeaheorghe/miniconda3/envs/assembly/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:224: PossibleUserWarning: The dataloader, test_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 16 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Testing DataLoader 0:  66%|######6   | 1578/2387 [54:27<27:55,  2.07s/it]WARNING: No critical points found in batch. Skipping matching.
Testing DataLoader 0:  76%|#######5  | 1804/2387 [1:01:40<19:55,  2.05s/it]WARNING: No critical points found in batch. Skipping matching.
Testing DataLoader 0: 100%|##########| 2387/2387 [1:20:32<00:00,  2.02s/it]test/cls_loss: 0.056748; test/cls_acc: 1.000000; test/cls_precision: 0.999162; test/cls_recall: 0.999162; test/cls_f1: 0.999162; test/mat_loss: 6.637864; test/N_: 581.749817; test/loss: 6.694612; test/mat_f1: 0.039723; test/mat_precision: 0.038978; test/mat_recall: 0.041430; test/part_acc: 0.462078; test/chamfer_distance: 0.379877; test/trans_mse: 0.040259; test/rot_mse: 4934.861328; test/trans_rmse: 0.135360; test/rot_rmse: 51.711319; test/trans_mae: 0.108607; test/rot_mae: 44.591778
Testing DataLoader 0: 100%|##########| 2387/2387 [1:20:32<00:00,  2.02s/it]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
         test/N_             581.7498168945312
  test/chamfer_distance     0.3798772990703583
      test/cls_acc                  1.0
       test/cls_f1           0.999161958694458
      test/cls_loss        0.056748416274785995
   test/cls_precision        0.999161958694458
     test/cls_recall         0.999161958694458
        test/loss             6.6946120262146
       test/mat_f1          0.03972260653972626
      test/mat_loss          6.637863636016846
   test/mat_precision       0.03897792473435402
     test/mat_recall        0.04143049195408821
      test/part_acc         0.46207764744758606
      test/rot_mae           44.59177780151367
      test/rot_mse            4934.861328125
      test/rot_rmse          51.71131896972656
     test/trans_mae         0.10860680788755417
     test/trans_mse         0.04025927931070328
     test/trans_rmse        0.13536040484905243
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Done evaluation
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:                 epoch ▁
wandb:               test/N_ ▁
wandb: test/chamfer_distance ▁
wandb:          test/cls_acc ▁
wandb:           test/cls_f1 ▁
wandb:         test/cls_loss ▁
wandb:    test/cls_precision ▁
wandb:       test/cls_recall ▁
wandb:             test/loss ▁
wandb:           test/mat_f1 ▁
wandb:         test/mat_loss ▁
wandb:    test/mat_precision ▁
wandb:       test/mat_recall ▁
wandb:         test/part_acc ▁
wandb:          test/rot_mae ▁
wandb:          test/rot_mse ▁
wandb:         test/rot_rmse ▁
wandb:        test/trans_mae ▁
wandb:        test/trans_mse ▁
wandb:       test/trans_rmse ▁
wandb:   trainer/global_step ▁
wandb: 
wandb: Run summary:
wandb:                 epoch 0
wandb:               test/N_ 581.74982
wandb: test/chamfer_distance 0.37988
wandb:          test/cls_acc 1.0
wandb:           test/cls_f1 0.99916
wandb:         test/cls_loss 0.05675
wandb:    test/cls_precision 0.99916
wandb:       test/cls_recall 0.99916
wandb:             test/loss 6.69461
wandb:           test/mat_f1 0.03972
wandb:         test/mat_loss 6.63786
wandb:    test/mat_precision 0.03898
wandb:       test/mat_recall 0.04143
wandb:         test/part_acc 0.46208
wandb:          test/rot_mae 44.59178
wandb:          test/rot_mse 4934.86133
wandb:         test/rot_rmse 51.71132
wandb:        test/trans_mae 0.10861
wandb:        test/trans_mse 0.04026
wandb:       test/trans_rmse 0.13536
wandb:   trainer/global_step 0