test/cls_loss: 0.053309; test/cls_acc: 1.000000; test/cls_precision: 1.000000; test/cls_recall: 1.000000; test/cls_f1: 1.000000; test/mat_loss: 6.510522; test/n_critical_max: 556.640259; test/loss: 6.563831; test/mat_precision: 0.045090; test/mat_recall: 0.047037; test/mat_f1: 0.045717; test/part_acc: 0.467582; test/chamfer_distance: 0.377532; test/trans_mse: 0.040114; test/rot_mse: 4999.684082; test/trans_rmse: 0.134277; test/rot_rmse: 52.167927; test/trans_mae: 0.108104; test/rot_mae: 44.936474
Testing DataLoader 0: 100%|##########| 2387/2387 [1:16:46<00:00,  1.93s/it]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  test/chamfer_distance     0.3775315284729004
      test/cls_acc                  1.0
       test/cls_f1                  1.0
      test/cls_loss         0.05330881103873253
   test/cls_precision               1.0
     test/cls_recall                1.0
        test/loss            6.563830852508545
       test/mat_f1          0.04571746662259102
      test/mat_loss          6.510522365570068
   test/mat_precision      0.045090336352586746
     test/mat_recall       0.047036655247211456
   test/n_critical_max       556.6402587890625
      test/part_acc         0.46758171916007996
      test/rot_mae           44.93647384643555
      test/rot_mse           4999.68408203125
      test/rot_rmse          52.16792678833008
     test/trans_mae         0.10810409486293793
     test/trans_mse         0.04011351987719536
     test/trans_rmse        0.13427676260471344
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Done evaluation
wandb: Waiting for W&B process to finish... (success).
wandb: - 0.231 MB of 0.231 MB uploaded (0.000 MB deduped)
wandb: Run history:
wandb:                 epoch ▁
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
wandb:   test/n_critical_max ▁
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
wandb: test/chamfer_distance 0.37753
wandb:          test/cls_acc 1.0
wandb:           test/cls_f1 1.0
wandb:         test/cls_loss 0.05331
wandb:    test/cls_precision 1.0
wandb:       test/cls_recall 1.0
wandb:             test/loss 6.56383
wandb:           test/mat_f1 0.04572
wandb:         test/mat_loss 6.51052
wandb:    test/mat_precision 0.04509
wandb:       test/mat_recall 0.04704
wandb:   test/n_critical_max 556.64026
wandb:         test/part_acc 0.46758
wandb:          test/rot_mae 44.93647
wandb:          test/rot_mse 4999.68408
wandb:         test/rot_rmse 52.16793
wandb:        test/trans_mae 0.1081
wandb:        test/trans_mse 0.04011
wandb:       test/trans_rmse 0.13428
wandb:   trainer/global_step 0
wandb: 
wandb: 🚀 View run jigsaw_multi_4x4_128_512_250e_cosine_everyday_2026-01-10-08-37-55 at: https://wandb.ai/andreea-ghe-babes-bolyai-university/jigsaw/runs/3qzz3ite
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: results/jigsaw_multi_4x4_128_512_250e_cosine_everyday/model_save/wandb/run-20260110_083759-3qzz3ite/logs