[Checkpoint] Saved final 332 results to results/jigsaw_4x4_128_512_250e_cosine_artifact/test_checkpoint_final.pt
test/cls_loss: 0.075046; test/cls_acc: 1.000000; test/cls_precision: 1.000000; test/cls_recall: 1.000000; test/cls_f1: 1.000000; test/mat_loss: 6.583780; test/n_critical_max: 1095.918579; test/loss: 6.658826; test/mat_precision: 0.038380; test/mat_recall: 0.038494; test/mat_f1: 0.038427; test/part_acc: 0.566791; test/chamfer_distance: 0.377062; test/trans_mse: 0.026716; test/rot_mse: 4224.596680; test/trans_rmse: 0.099842; test/rot_rmse: 44.685463; test/trans_mae: 0.083491; test/rot_mae: 38.478504
Testing DataLoader 0: 100%|##########| 332/332 [13:41<00:00,  2.48s/it]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  test/chamfer_distance     0.37706220149993896
      test/cls_acc                  1.0
       test/cls_f1                  1.0
      test/cls_loss         0.07504638284444809
   test/cls_precision               1.0
     test/cls_recall                1.0
        test/loss           6.6588263511657715
       test/mat_f1         0.038426704704761505
      test/mat_loss          6.583779811859131
   test/mat_precision       0.03837992995977402
     test/mat_recall        0.03849441930651665
   test/n_critical_max      1095.9185791015625
      test/part_acc         0.5667913556098938
      test/rot_mae           38.4785041809082
      test/rot_mse            4224.5966796875
      test/rot_rmse         44.685462951660156
     test/trans_mae         0.08349128812551498
     test/trans_mse        0.026715688407421112
     test/trans_rmse        0.09984241425991058
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Done evaluation
wandb: Waiting for W&B process to finish... (success).
wandb: / 0.037 MB of 0.037 MB uploaded (0.000 MB deduped)
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
wandb: test/chamfer_distance 0.37706
wandb:          test/cls_acc 1.0
wandb:           test/cls_f1 1.0
wandb:         test/cls_loss 0.07505
wandb:    test/cls_precision 1.0
wandb:       test/cls_recall 1.0
wandb:             test/loss 6.65883
wandb:           test/mat_f1 0.03843
wandb:         test/mat_loss 6.58378
wandb:    test/mat_precision 0.03838
wandb:       test/mat_recall 0.03849
wandb:   test/n_critical_max 1095.91858
wandb:         test/part_acc 0.56679
wandb:          test/rot_mae 38.4785
wandb:          test/rot_mse 4224.59668
wandb:         test/rot_rmse 44.68546
wandb:        test/trans_mae 0.08349
wandb:        test/trans_mse 0.02672
wandb:       test/trans_rmse 0.09984
wandb:   trainer/global_step 0
wandb: 
wandb: 🚀 View run jigsaw_4x4_128_512_250e_cosine_artifact_2026-01-10-21-32-06 at: https://wandb.ai/andreea-ghe-babes-bolyai-university/jigsaw/runs/ep8nsi2j
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)