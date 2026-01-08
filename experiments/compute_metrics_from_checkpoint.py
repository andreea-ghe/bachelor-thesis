"""
Compute evaluation metrics from a saved test checkpoint.
Usage: python -m experiments.compute_metrics_from_checkpoint --checkpoint path/to/checkpoint.pt
"""
import argparse
import torch


def compute_metrics(checkpoint_path):
    """Load checkpoint and compute aggregated metrics."""
    print(f"Loading checkpoint: {checkpoint_path}")
    outputs = torch.load(checkpoint_path)
    print(f"Loaded {len(outputs)} batch results")
    
    # Find all keys that appear in any output
    all_keys = set()
    for out in outputs:
        all_keys.update(out.keys())
    
    print(f"\nAll keys found: {sorted(all_keys)}")
    
    # Get batch sizes
    batch_sizes = []
    for out in outputs:
        bs = out.get('batch_size', 1)
        if torch.is_tensor(bs):
            batch_sizes.append(bs.item())
        else:
            batch_sizes.append(bs)
    batch_sizes = torch.tensor(batch_sizes, dtype=torch.float32)
    
    # Compute metrics for each key
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    metrics = {}
    for key in sorted(all_keys):
        if key == 'batch_size':
            continue
            
        # Collect values, using 0 for missing entries
        values = []
        valid_mask = []
        for out in outputs:
            if key in out:
                val = out[key]
                if torch.is_tensor(val):
                    values.append(val.float())
                else:
                    values.append(torch.tensor(float(val)))
                valid_mask.append(1.0)
            else:
                values.append(torch.tensor(0.0))
                valid_mask.append(0.0)
        
        values = torch.stack(values)
        valid_mask = torch.tensor(valid_mask)
        
        # Weighted average (only over valid entries)
        if valid_mask.sum() > 0:
            weighted_sum = (values * batch_sizes * valid_mask).sum()
            total_weight = (batch_sizes * valid_mask).sum()
            avg_value = weighted_sum / total_weight
            metrics[key] = avg_value.item()
            print(f"test/{key}: {avg_value.item():.6f}")
    
    # Save metrics to file
    output_path = checkpoint_path.replace('.pt', '_metrics.txt')
    with open(output_path, 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("="*60 + "\n")
        for key, value in metrics.items():
            f.write(f"test/{key}: {value:.6f}\n")
    
    print(f"\nMetrics saved to: {output_path}")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to test checkpoint file')
    args = parser.parse_args()
    
    compute_metrics(args.checkpoint)

