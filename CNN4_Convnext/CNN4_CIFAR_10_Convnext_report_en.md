# ConvNeXt on CIFAR-10: Report

## 1. Project Overview
This notebook builds an image classification pipeline for the CIFAR-10 dataset using a pretrained ConvNeXt V2 tiny backbone. The goal is to classify 10 object categories from RGB images by fine-tuning an ImageNet-pretrained vision model with a carefully designed training loop and evaluation workflow.

The notebook has not been executed yet, so this report is written from the implementation itself rather than from recorded metrics. As a result, it describes the experimental design, training strategy, and analysis plan in full detail, while leaving numerical results to be filled in after execution.

## 2. Dataset and Preprocessing
### 2.1 CIFAR-10 Dataset
CIFAR-10 is a standard multi-class image classification benchmark with 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, and truck. The original images are 32 × 32 RGB images. The training split contains 50,000 images and the official test set contains 10,000 images.

### 2.2 Data Split
The notebook loads the official CIFAR-10 test set and then randomly splits it into two equal subsets:

1. Validation set: 5,000 images
2. Test set: 5,000 images

This gives the following experimental layout:

1. Training set: 50,000 images
2. Validation set: 5,000 images
3. Test set: 5,000 images

This setup allows the training process to monitor generalization on a validation subset and then evaluate the final model on a separate test subset. However, from a stricter experimental design perspective, a more standard approach would be to split validation data from the training set and keep the full official test set untouched for final reporting.

### 2.3 Preprocessing
The notebook uses a simple preprocessing pipeline rather than aggressive augmentation:

1. Resize each image to 224 × 224
2. Convert the image to a tensor
3. Normalize using ImageNet mean and standard deviation

The 224 × 224 input size matches the expected scale of ConvNeXt-style pretrained models. This is important because ConvNeXt V2 was designed around ImageNet-scale inputs, so upscaling CIFAR-10 images makes the input distribution closer to the original pretraining regime.

There is no explicit data augmentation such as random crop, horizontal flip, color jitter, RandAugment, MixUp, or CutMix. This keeps the pipeline simple, but it also limits the amount of regularization available for a small dataset like CIFAR-10.

### 2.4 DataLoader Configuration
The DataLoader settings are:

1. Training loader: batch size 64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
2. Validation loader: batch size 64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2
3. Test loader: batch size 64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2

These settings improve throughput and reduce data-loading overhead during GPU training.

## 3. Model Architecture
### 3.1 Backbone Choice
The notebook does not implement a hand-written CNN. Instead, it uses timm.create_model("convnextv2_tiny", pretrained=True, num_classes=10), which creates a pretrained ConvNeXt V2 tiny model and adapts the final classification layer to the 10 CIFAR-10 classes.

This is a transfer learning setup, not a scratch-built CNN. The pretrained backbone can reuse rich ImageNet features and usually performs better than a small custom CNN when data is limited.

### 3.2 Architectural Characteristics
The model has the following characteristics:

1. ImageNet pretrained weights are loaded
2. The input resolution is 224 × 224
3. The classifier head is configured for 10 classes
4. No custom convolutional blocks are defined in the notebook
5. No freezing or staged unfreezing strategy is used; all parameters are trainable

The notebook also prints a torchinfo summary to inspect tensor shapes and parameter counts.

### 3.3 Mismatch Between Markdown Text and Code
There is an important inconsistency in the notebook: some markdown sections still describe a traditional CNN with Conv2d, ReLU, MaxPool, BatchNorm, Flatten, and Linear layers, but the actual code uses ConvNeXt V2 tiny. Another markdown note mentions training for 50 epochs, while the code sets EPOCHS = 5.

For any final report, the code should be treated as the source of truth.

## 4. Training Setup
### 4.1 Loss Function and Optimizer
The training configuration is:

1. Epochs: 5
2. Loss: CrossEntropyLoss(label_smoothing=0.02)
3. Backup loss for soft labels: CrossEntropyLoss()
4. Optimizer: AdamW
5. Learning rate: 1e-4
6. Weight decay: 5e-5
7. Betas: (0.9, 0.999)

Label smoothing reduces overconfidence in the target distribution, while AdamW provides a decoupled weight decay formulation that is widely used in modern vision training.

### 4.2 Learning Rate Scheduler
The notebook uses CosineAnnealingLR with per-optimizer-step updates:

1. T_max = EPOCHS × len(train_loader)
2. eta_min = 0.8e-4

This creates a smooth cosine decay over the full training run and is often more stable than a fixed learning rate or a stepwise decay schedule.

### 4.3 Stability and Performance Features
The training loop includes several modern training enhancements:

1. Mixed precision training with AMP
2. channels_last memory format on CUDA
3. torch.compile(mode="reduce-overhead") when available
4. Gradient clipping with max_norm=1.0
5. Pinned memory and non-blocking transfers
6. Exponential Moving Average (EMA) model weights

The EMA model is especially important. The notebook updates a separate EMA copy of the model after each optimizer step, and validation/testing use the EMA weights for more stable evaluation.

### 4.4 Training Loop
Each training iteration performs the following steps:

1. Load a mini-batch
2. Move data and labels to the device
3. Run the forward pass under autocast when CUDA is available
4. Compute loss
5. Backpropagate gradients
6. Clip gradients
7. Apply optimizer step
8. Advance the scheduler
9. Update EMA parameters
10. Accumulate training loss and accuracy
11. Run validation at the end of the epoch

The notebook also saves one checkpoint per epoch and tracks the best validation accuracy to save both raw and EMA best checkpoints.

### 4.5 Checkpoint Strategy
The notebook produces the following checkpoint files:

1. Epoch checkpoints: Models/convnext_model_1.pth, Models/convnext_model_2.pth, and so on
2. Best raw checkpoint: Models/convnext_model_best.pth
3. Best EMA checkpoint: Models/convnext_model_best_ema.pth

The best model is selected based on validation accuracy, not training loss.

## 5. Evaluation Strategy
### 5.1 Validation
At the end of each epoch, the notebook computes validation loss and validation accuracy. Validation is performed with the EMA model, which usually gives a smoother estimate of generalization performance.

### 5.2 Test Evaluation
After training, the notebook reloads the best checkpoint and evaluates the model on the test subset. The test phase records:

1. Average test loss
2. Overall test accuracy

The code also stores predicted labels and true labels for further analysis.

### 5.3 Confusion Matrix
The notebook computes a confusion matrix using sklearn.metrics.confusion_matrix and stores the result in a pandas DataFrame. This is useful for identifying class-level confusions, such as visually similar classes that the model may mix up.

### 5.4 Classification Report
The notebook also generates precision, recall, F1-score, and support for each class.

One important caveat: the notebook calls classification_report(pred_vec, label_vec, ...). The conventional argument order should be y_true first and y_pred second. If this is not corrected, the per-class metrics may be misinterpreted. This should be fixed before using the numbers in a formal report.

## 6. Visualization and Analysis
The notebook contains a fairly complete visualization workflow:

1. Learning-rate schedule plot
2. Training and validation loss curves
3. Training and validation accuracy curves
4. A 4 × 4 grid of test predictions
5. A 50-image prediction gallery
6. Misclassified examples for each class

These plots support both training diagnosis and qualitative model inspection:

1. The learning-rate curve verifies the cosine schedule
2. The loss and accuracy curves reveal convergence and possible overfitting
3. The prediction grids make it easy to inspect how the model behaves on individual examples
4. The misclassification gallery highlights the most failure-prone classes

## 7. Results and Discussion
Because the notebook has not been run, there are no recorded numerical results to report yet. The following outputs are planned but not yet produced:

1. Best validation accuracy and best epoch
2. Final test loss and test accuracy
3. The confusion matrix values
4. The classification report table
5. The learning-curve plots
6. The qualitative prediction figures

From the implementation alone, however, the experimental design is clear: the notebook aims to fine-tune a pretrained ConvNeXt V2 tiny model on CIFAR-10 using a stable optimization recipe and a thorough evaluation pipeline.

## 8. Strengths
1. Uses a pretrained ConvNeXt V2 tiny backbone
2. Employs AdamW, label smoothing, weight decay, gradient clipping, and EMA
3. Uses AMP and torch.compile for training efficiency
4. Includes a complete evaluation workflow beyond top-1 accuracy
5. Saves both raw and EMA best checkpoints for later comparison

## 9. Limitations
1. No explicit data augmentation is used
2. The validation/test split is created from the official test set, which is not the most standard protocol
3. The notebook text and the actual code are inconsistent in several places
4. The classification_report call likely uses the wrong argument order
5. No execution has been run, so final metrics are still unavailable

## 10. Possible Improvements
1. Add stronger data augmentation, such as random crop, horizontal flip, color jitter, RandAugment, MixUp, or CutMix
2. Split validation data from the training set and preserve the full official test set for final reporting
3. Correct the classification_report argument order
4. Add early stopping or a more formal model-selection policy
5. Run ablation studies on backbone choice, optimizer, scheduler, and augmentation strategy
6. Add figure numbers and result tables for a more formal report layout

## 11. Conclusion
This project presents a CIFAR-10 classification pipeline based on a pretrained ConvNeXt V2 tiny model. Rather than building a plain CNN from scratch, the notebook focuses on transfer learning, training stability, and systematic evaluation. The use of AdamW, cosine learning-rate scheduling, mixed precision, EMA, and detailed post-training analysis shows a modern and well-structured training workflow.

Although the notebook has not yet been executed, the implementation already forms a complete experimental framework. Once the actual metrics are generated, this report can be finalized with the numerical results and visualizations inserted in the appropriate sections.
