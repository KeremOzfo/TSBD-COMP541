"""
Quick Reference: Final Test Epoch Logging

This module provides comprehensive experiment logging with all visualizations.

USAGE:
------
from utils.exp_logging import log_final_test_epoch

# After training (in main.py or test.py)
exp_dir = log_final_test_epoch(
    model=trained_model,
    trigger_model=trigger_model,
    train_loader=train_loader,          # Clean training data
    test_loader=test_loader,            # Test data
    poisoned_loader=poisoned_train_loader,  # Poisoned training data
    args=args,
    trigger_results=trigger_results,    # Optional: trigger training history
    model_poison_dic=model_poison_dic,  # Optional: CA/ASR history
    save_dir="Results",
)

WHAT GETS LOGGED:
-----------------

1. METRICS (via bd_test_with_samples):
   ✓ Clean Accuracy (CA)
   ✓ Attack Success Rate (ASR)
   ✓ Sample cases (success/failure)

2. VISUALIZATIONS (via log_all):
   ✓ PCA plot (train set: clean vs poisoned)
   ✓ t-SNE plot (train set: clean vs poisoned)
   ✓ Sample backdoor cases (success/failure examples)
   ✓ GradCAM heatmaps (for successful attacks)
   ✓ Training curves (loss, accuracy, ASR over epochs)

3. FILES CREATED:
   - args_and_results.txt       : All parameters + final metrics
   - trigger_loss.png           : Trigger training loss curves
   - trigger_accuracy.png       : Trigger training accuracy
   - poison_metrics.png         : CA/ASR over epochs
   - examples/
     - example_plots.txt        : Manifest
     - sample_*_success.png     : Successful attacks
     - sample_*_failure.png     : Failed attacks
     - gradcam/
       - sample_*_gradcam_map.png : GradCAM visualizations
   - latent/
     - pca_*.png               : PCA on train set
     - tsne_*.png              : t-SNE on train set

KEY FEATURES:
-------------
- PCA/t-SNE done on TRAIN SET (as specified)
- Sample collection via bd_test_with_samples
- GradCAM for successful backdoor examples
- All in one function call
- Consistent output structure

EXAMPLE OUTPUT MESSAGE:
-----------------------
======================================================================
FINAL TEST EPOCH - COMPREHENSIVE EVALUATION
======================================================================

[1/2] Running backdoor test with sample collection...
  ✓ Clean Accuracy: 92.45%
  ✓ Attack Success Rate (ASR): 98.76%
  ✓ Collected 16 sample cases

[2/2] Generating comprehensive visualizations and logs...
  - PCA and t-SNE plots (on training set)
  - Sample backdoor cases (success/failure)
  - GradCAM heatmaps
  - Training curves and metrics

======================================================================
FINAL TEST EPOCH COMPLETED
Results saved to: Results/BasicMotions_G-TriggerNet_C-PatchTST_a1b2c3d4
======================================================================
"""
