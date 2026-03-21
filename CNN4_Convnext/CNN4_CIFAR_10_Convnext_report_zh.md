# Assignment 1: Summary Report

本報告總結了 ConvNeXt 管線相對於 baseline 的改進，包括模型架構設計、訓練流程、實驗結果、評估和結論。目標是在保持訓練穩定和效率的同時提升測試性能和泛化能力。

### **1. 模型架構設計**

我把原本的 CNN baseline 升級為預訓練的 ConvNeXt V2 tiny，並在 CIFAR-10 上微調。

* **骨幹網路 (轉移學習)**
  * **Baseline:** 從頭訓練的 2 層 CNN，輸入 32×32。
  * **Final:** 以 ImageNet 預訓練的 ConvNeXt V2 tiny，輸入調整到 224×224，整個 backbone 微調。
  * *Improvement:* 預訓練特徵有強大的低階與中階表徵能力，性能大幅提升。

* **輸入處理**
  * **Baseline:** 原生 32×32 CIFAR-10 正規化輸入。
  * **Final:** 224×224，採用 ConvNeXt 預訓練一致的正規化。
  * *Improvement:* 更大尺寸可獲得更多空間資訊和更強特徵。

* **分類頭**
  * **Baseline:** 簡單全連接輸出 10 類。
  * **Final:** `NormMlpClassifierHead`（來自 timm）：AdaptiveAvgPool2d → LayerNorm2d → Flatten → Dropout(p=0.0) → Linear(768 → 10)。
  * *Improvement:* LayerNorm 穩定池化後的特徵再進行線性映射；輕量分類頭直接將 768 維 backbone 輸出映射至 10 個類別 logit，同時保留完整的預訓練表徵。

### **2. 訓練流程**

我改進了優化器和訓練流程，使收斂更平滑、泛化更好。

* **優化器與損失**
  * **Baseline:** Adam + CrossEntropyLoss。
  * **Final:** AdamW + CrossEntropyLoss(label_smoothing=0.03)。
  * *Improvement:* AdamW 適合大模型，label smoothing 降低過度自信。

* **學習率調度**
  * **Baseline:** 固定學習率。
  * **Final:** CosineAnnealingLR。
  * *Improvement:* 平滑衰減可避免後期不穩定與過擬合。

* **性能優化**
  * **Baseline:** 全精度訓練。
  * **Final:** AMP、torch.compile、channels_last、梯度裁剪、EMA。
  * *Improvement:* GPU 利用率高、梯度穩定、評估權重平滑。

### **3. 實驗結果**

改進後的模型在各項指標上均取得顯著提升，展現出更好的泛化能力與魯棒性。

* **測試準確率**
  * **Baseline:** 71.88%
  * **Final:** 98.96%
  * *Improvement:* 絕對提升 +27.08%，充分展示遷移學習相較於從零訓練 CNN 在 CIFAR-10 上的巨大優勢。

* **驗證準確率**
  * **Baseline:** 72.66%（第 14 epoch 最佳）
  * **Final:** 98.56%（第 4 epoch 最佳）
  * *Improvement:* 最佳驗證準確率更早達到且遠高於 baseline，展現出更快的收斂速度與更強的泛化能力。

* **測試損失**
  * **Baseline:** 0.0099（以各 batch 平均值之和除以樣本總數計算，低估了真實的每樣本交叉熵）
  * **Final:** 0.21833（每樣本交叉熵，含 label_smoothing=0.03）
  * *Improvement:* 兩者計算方式不同且最終模型含 label smoothing，無法直接比較；準確率提升（+27.08%）才是更有意義的改進依據。

### **4. 評估**

我使用分類指標與誤差分析評估兩個模型，以了解泛化能力與剩餘限制。

* **評估指標**
  * **Baseline:** 準確率、測試損失與各類別 precision/recall/F1（`classification_report` 參數傳入順序錯誤）
  * **Final:** 準確率、測試損失、各類別 precision/recall/F1、macro 與 weighted F1 平均（與 baseline 相同的參數倒置順序）
  * *Improvement:* 完整指標提供各類別性能的全貌。注意：兩個報告的 F1 值均正確有效；precision 與 recall 在兩個報告中均因參數順序問題而互換，為共同限制。

* **F1 分數（Macro 平均）**
  * **Baseline:** 0.7175（因 `classification_report` 參數倒置導致 precision/recall 互換；F1 值仍正確）
  * **Final:** 0.9897
  * *Improvement:* 提升 +37.9%，顯示所有類別的表現均大幅提升且更加均衡。

* **F1 分數（Weighted 平均）**
  * **Baseline:** 0.7204（同上注意）
  * **Final:** 0.9896
  * *Improvement:* 提升 +37.4%，展示在所有類別分布上的穩健性。

* **困難類別表現**
  * **Baseline:** "cat" 類 F1 = 0.5189（最差）；"car" 類 F1 = 0.8247（最佳）
  * **Final:** "cat" 類 F1 = 0.9766（最差）；"frog" 類 F1 = 0.9959（最佳）；"dog" F1 = 0.9843；其他所有類別均超過 0.988
  * *Improvement:* 最難類別（cat）提升約 +88.2%（從 0.5189 提升至 0.9766），展示模型辨別視覺相似類別的能力大幅增強。

* **指標正確性（共同限制）**
  * **Baseline:** `classification_report` 以 `(pred_vec, label_vec)` 呼叫（參數倒置），導致 precision 與 recall 互換（F1 不受影響）
  * **Final:** 相同的參數倒置順序 `(pred_vec, label_vec)`——此問題在本 notebook 中未修正
  * *Note:* 兩個模型的 F1 分數均不受影響且有效；precision 與 recall 在兩個報告中均技術性互換，為共同限制。

### **5. 結論**

以下總結本次實驗的主要發現、有效技術與剩餘限制。

* **有效技術**
  * **架構:** **預訓練 ConvNeXt V2 tiny backbone** 提供強大的低階與中階特徵，無需從零訓練即可快速達到高性能。
  * **優化:** **AdamW + CosineAnnealingLR** 相較於 Adam 固定學習率，帶來更穩定的收斂與更好的泛化能力。
  * **泛化:** **AMP + 梯度裁剪 + EMA** 提升硬體效率、穩定梯度，並產出更平滑的評估權重。
  * *Impact:* 上述改進共同將測試準確率從 71.88% 提升至 98.96%，macro F1 從 0.7175 提升至 0.9897，且僅需 5 個訓練 epoch。

* **類別均衡改進**
  * **Baseline:** Macro F1 = 0.7175，Weighted F1 = 0.7204；最難類別 "cat" F1 = 0.5189
  * **Final:** Macro F1 = 0.9897，Weighted F1 = 0.9896；"cat" F1 = 0.9766（最差），"frog" F1 = 0.9959（最佳）
  * *Improvement:* 模型大幅提升 macro/weighted F1，最難類別（cat）提升約 +88.2%，展示在所有類別上的全面改進與強健性。

* **剩餘限制**
  * **主要問題:** 驗證集從原始測試集劃分（5,000 驗證 + 5,000 測試），而非從訓練集中劃分，可能導致性能估計偏於樂觀。
  * **根本原因:** 除 ConvNeXt 輸入所需的 resize 外，未加入進階增強策略（如 RandomHorizontalFlip、RandomCrop、ColorJitter、RandAugment、MixUp）。
  * *Future Direction:* 採用正確的 train/val/test 劃分協議，並加入強增強策略，進一步提升魯棒性並縮小驗證集與測試集分布的差距。
