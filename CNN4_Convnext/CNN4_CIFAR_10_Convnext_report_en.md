# Assignment 2: Summary Report

This report summarizes the improvements made to the baseline CIFAR-10 model in the ConvNeXt pipeline, including model architecture design, training process, experimental results, evaluation, and conclusions. The goal is to improve test performance and generalization while maintaining stable and efficient training.

### **1. Model Architecture Design**

I upgraded the original CNN baseline into a pretrained ConvNeXt V2 tiny model and fine-tuned it for CIFAR-10.

* **Model Backbone (Transfer Learning)**
  * **Baseline:** from-scratch 2-layer CNN on 32×32 CIFAR-10.
  * **Final:** `ConvNeXt V2 tiny` pretrained on ImageNet, input resized to 224×224, entire backbone fine-tuned.
  * *Improvement:* pretrained backbone provides robust low- and mid-level features and dramatically increases representation power.

* **Input Processing**
  * **Baseline:** 32×32 normalized CIFAR-10 images.
  * **Final:** 224×224 input with same normalization strategy as ConvNeXt pretraining.
  * *Improvement:* larger input resolution gives the model more spatial information and better feature extraction.

* **Head / Classifier**
  * **Baseline:** small dense FC head for 10 classes.
  * **Final:** `NormMlpClassifierHead` (from timm): AdaptiveAvgPool2d → LayerNorm2d → Flatten → Dropout(p=0.0) → Linear(768 → 10).
  * *Improvement:* LayerNorm stabilizes the pooled features before the linear projection; the lightweight head directly maps the 768-dim backbone output to 10 class logits while leveraging the full pretrained representation.

### **2. Training Process**

I improved the optimizer and training recipe for smoother convergence and stronger generalization.

* **Optimizer and Loss**
  * **Baseline:** Adam + CrossEntropyLoss.
  * **Final:** AdamW + CrossEntropyLoss(label_smoothing=0.03).
  * *Improvement:* AdamW with weight decay is more compatible with modern vision backbones; label smoothing reduces overconfidence.

* **Learning Rate Schedule**
  * **Baseline:** fixed learning rate.
  * **Final:** CosineAnnealingLR schedule.
  * *Improvement:* smooth decay helps avoid late-epoch instability and overfitting.

* **Mixed Precision / Performance**
  * **Baseline:** full precision.
  * **Final:** AMP training + `torch.compile` + channels_last + gradient clipping + EMA (Exponential Moving Average).
  * *Improvement:* better hardware utilization, stable gradients, and smoother model checkpoint weights.

### **3. Experimental Results**

The improved model achieved significant gains across all metrics, demonstrating better generalization and robustness.

* **Test Accuracy**
  * **Baseline:** 71.88%
  * **Final:** 98.96%
  * *Improvement:* Absolute improvement of +27.08%, demonstrating the dramatic effectiveness of transfer learning with ConvNeXt V2 over a from-scratch CNN on CIFAR-10.

* **Validation Accuracy**
  * **Baseline:** 72.66% (best at epoch 14)
  * **Final:** 98.56% (best at epoch 4)
  * *Improvement:* Best validation accuracy reached far earlier and at a far higher level, showing both faster convergence and superior generalization.

* **Test Loss**
  * **Baseline:** 0.0099 (computed as sum-of-batch-averages / num_samples, which underestimates true per-sample cross-entropy)
  * **Final:** 0.21833 (per-sample cross-entropy with label_smoothing=0.03)
  * *Improvement:* Loss values are not directly comparable due to different aggregation methods and label smoothing; the significant accuracy gain (+27.08%) is the more meaningful indicator of improvement.

### **4. Evaluation**

I evaluated both models using comprehensive metrics and error analysis to understand generalization and remaining limitations.

* **Evaluation Metrics**
  * **Baseline:** Accuracy, test loss, and per-class precision/recall/F1 (with swapped arguments in `classification_report`)
  * **Final:** Accuracy, test loss, per-class precision/recall/F1, macro and weighted F1 averages (same reversed argument order as baseline)
  * *Improvement:* Comprehensive metrics provide a complete picture of model performance across all classes. Note: F1 scores are valid in both reports; precision and recall values are technically swapped in both due to the shared argument ordering.

* **F1 Score (Macro Average)**
  * **Baseline:** 0.7175 (precision/recall swapped due to reversed `classification_report` args; F1 value remains valid)
  * **Final:** 0.9897
  * *Improvement:* Improved by +37.9%, showing dramatically more balanced performance across all classes.

* **F1 Score (Weighted Average)**
  * **Baseline:** 0.7204 (same caveat as above)
  * **Final:** 0.9896
  * *Improvement:* Improved by +37.4%, demonstrating robust performance across all class distributions.

* **Hard Class Performance**
  * **Baseline:** "cat" class F1 = 0.5189 (worst); "car" class F1 = 0.8247 (best)
  * **Final:** "cat" class F1 = 0.9766 (weakest); "frog" class F1 = 0.9959 (strongest); "dog" F1 = 0.9843; all other classes exceed 0.988
  * *Improvement:* The hardest class (cat) improved by approximately +88.2% (from 0.5189 to 0.9766), showing the model's dramatically enhanced ability to discriminate visually similar categories.

* **Metrics Correctness (Shared Limitation)**
  * **Baseline:** `classification_report` was called with reversed arguments `(pred_vec, label_vec)` instead of `(label_vec, pred_vec)`, causing precision and recall to be swapped (F1 remains valid)
  * **Final:** Same reversed argument order `(pred_vec, label_vec)` — the bug is not fixed in this notebook
  * *Note:* F1 scores are unaffected and remain valid for both models; precision and recall values are technically swapped in both reports. This is a shared limitation.

### **5. Conclusions**

I summarize the key findings, effective techniques, and remaining limitations from this work.

* **Key Techniques That Worked Well**
  * **Architecture:** The **pretrained ConvNeXt V2 tiny backbone** provided powerful low- and mid-level features, enabling rapid high-performance fine-tuning without training from scratch.
  * **Optimization:** **AdamW + CosineAnnealingLR** delivered stable convergence and better generalization compared to Adam with a fixed learning rate.
  * **Generalization:** **AMP + gradient clipping + EMA** improved hardware efficiency, stabilized gradients, and produced smoother model weights for evaluation.
  * *Impact:* These improvements jointly increased test accuracy from 71.88% to 98.96% and macro F1 from 0.7175 to 0.9897, achieved within just 5 training epochs.

* **Balanced Class Improvements**
  * **Baseline:** Macro F1 = 0.7175, weighted F1 = 0.7204; hardest class "cat" F1 = 0.5189
  * **Final:** Macro F1 = 0.9897, weighted F1 = 0.9896; "cat" F1 = 0.9766 (weakest), "frog" F1 = 0.9959 (strongest)
  * *Improvement:* The model significantly raised macro/weighted F1 and notably strengthened the hardest class (approximately +88.2% improvement), showing improved robustness across all categories.

* **Remaining Limitations**
  * **Main Issue:** Validation set is split from the original test set (5,000 validation + 5,000 test) rather than from the training set, which may lead to overly optimistic performance estimates.
  * **Root Cause:** No advanced augmentation pipeline (e.g., RandomHorizontalFlip, RandomCrop, ColorJitter, RandAugment, MixUp) was added beyond the resize required for ConvNeXt input.
  * *Future Direction:* Adopt a proper train/val/test split protocol and add strong augmentation strategies to further improve robustness and close the gap between validation and test distributions.
