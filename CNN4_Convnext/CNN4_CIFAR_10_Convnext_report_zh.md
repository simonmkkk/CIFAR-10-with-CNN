# ConvNeXt on CIFAR-10: 報告

## 1. 專案摘要
本筆記本使用 CIFAR-10 影像分類資料集，建立一個以預訓練 ConvNeXt V2 tiny 為核心的分類模型，目標是將 10 類自然影像正確分類。與傳統從零開始訓練的 CNN 不同，本專案採用來自 ImageNet 的預訓練權重，並搭配 AdamW、CosineAnnealingLR、label smoothing、混合精度訓練、梯度裁剪與 EMA 等技巧，以提升訓練穩定度與泛化能力。

需要特別說明的是，這份 notebook 目前尚未執行，因此本文是根據程式設計與實驗流程撰寫的完整報告稿，不包含實際跑出的數值結果。若要填入最終版報告，只需在執行 notebook 後補上 validation accuracy、test accuracy、confusion matrix 與 classification report 的實際數值即可。

## 2. 資料集與前處理
### 2.1 CIFAR-10 資料集
CIFAR-10 是常見的多分類影像資料集，包含 10 個類別：plane、car、bird、cat、deer、dog、frog、horse、ship、truck。原始影像大小為 32 × 32 的彩色 RGB 圖片，訓練集共有 50,000 張，官方測試集共有 10,000 張。

### 2.2 資料切分策略
本 notebook 先載入官方 CIFAR-10 測試集，再將其隨機切分成 validation set 與 test set，各 5,000 張。也就是說，實驗中最終的評估流程為：

1. Training set：50,000 張
2. Validation set：5,000 張
3. Test set：5,000 張

這樣的做法可以讓模型在訓練過程中使用 validation set 監控泛化表現，並在訓練結束後以獨立的 test set 評估最終結果。不過，從嚴謹的實驗設計角度來看，更標準的方式通常是從 training set 再切出 validation set，保留完整的官方 test set 做最終測試。這一點可列入限制與未來改進。

### 2.3 前處理與資料格式
本專案沒有使用複雜的資料增強，而是採用以下基本前處理：

1. Resize：將影像從 32 × 32 放大到 224 × 224
2. ToTensor：轉為 PyTorch tensor
3. Normalize：使用 ImageNet 的 mean 與 std

採用 224 × 224 的輸入尺寸，是為了與 ConvNeXt V2 的預訓練設定一致。ConvNeXt 系列通常以較大的輸入解析度進行訓練，因此將 CIFAR-10 影像放大後再輸入模型，可以更接近其原始設計條件。

需要注意的是，程式中 train_transform 與 test_transform 完全相同，代表此版本沒有額外的資料增強，例如 random crop、horizontal flip、color jitter、RandAugment、MixUp 或 CutMix。這使得訓練流程較簡潔，但也可能限制模型在小型資料集上的泛化能力。

### 2.4 DataLoader 設定
資料載入器設定如下：

1. train_loader：batch size 64，shuffle=True，num_workers=4，pin_memory=True，persistent_workers=True，prefetch_factor=2
2. val_loader：batch size 64，shuffle=False，num_workers=4，pin_memory=True，prefetch_factor=2
3. test_loader：batch size 64，shuffle=False，num_workers=4，pin_memory=True，prefetch_factor=2

這些設定的重點是讓訓練資料在每個 epoch 中隨機打亂，同時利用多工載入與 pinned memory 提升 GPU 資料吞吐效率。

## 3. 模型架構
### 3.1 模型選擇
這份 notebook 最重要的設計是使用 timm 套件中的 convnextv2_tiny，而且載入 pretrained=True 的 ImageNet 權重。分類頭透過 num_classes=10 對應 CIFAR-10 的 10 個類別。

換句話說，這不是一個從零開始手寫的傳統 CNN，而是以 ConvNeXt V2 tiny 作為 backbone 的遷移學習模型。這類模型通常具備更好的特徵抽取能力，也更適合在資料量有限的情況下快速取得不錯的表現。

### 3.2 架構特性
此模型具備以下特性：

1. 使用 ImageNet 預訓練權重
2. 以 224 × 224 輸入進行特徵萃取
3. 最後分類層對應 10 類輸出
4. 沒有額外自訂卷積 block 或手工設計的 CNN 層
5. 沒有 freeze backbone 的流程，表示整個模型都會一起微調

模型摘要透過 torchinfo.summary 印出，用於確認輸入輸出維度與參數規模。

### 3.3 與筆記本文字說明的不一致
此 notebook 中的 markdown 說明仍保留部分傳統 CNN 的敘述，例如 Conv2d、ReLU、MaxPool、BatchNorm、Flatten、Linear 等層的描述，但實際 code 已經改成 ConvNeXt V2 tiny。另有一處文字說明提到訓練 50 epochs，但實際程式將 EPOCHS 設為 5。

因此，若要將這份 notebook 作為正式報告來源，建議以 code 為準，而不是以舊的 markdown 文字說明為準。這點非常重要，否則報告內容會與實作不一致。

## 4. 訓練流程
### 4.1 損失函數與最佳化器
訓練設定如下：

1. Epochs：5
2. Loss function：CrossEntropyLoss(label_smoothing=0.02)
3. 備用 loss：CrossEntropyLoss()，供 soft label 情況使用
4. Optimizer：AdamW
5. Learning rate：1e-4
6. Weight decay：5e-5
7. Betas： (0.9, 0.999)

label smoothing 的作用是避免模型對單一類別過度自信，通常能帶來更穩定的分類效果。AdamW 則透過 decoupled weight decay 改善一般 Adam 在正則化上的行為，因此在 vision 類任務中很常見。

### 4.2 Learning Rate Scheduler
本專案使用 CosineAnnealingLR，並且在每個 optimizer step 更新一次學習率，而不是每個 epoch 更新一次。設定如下：

1. T_max = EPOCHS × len(train_loader)
2. eta_min = 0.8e-4

這種餘弦退火策略能讓 learning rate 逐步平滑下降，有助於在訓練後期細緻收斂，通常比固定 learning rate 更穩定。

### 4.3 混合精度與效能最佳化
訓練流程包含多項效能與穩定性設計：

1. AMP 混合精度訓練
2. CUDA 上使用 channels_last
3. torch.compile(mode="reduce-overhead")
4. Gradient clipping，max_norm=1.0
5. pin_memory 與 non_blocking data transfer
6. EMA（Exponential Moving Average）權重更新

其中 EMA 是很重要的設計。每次 optimizer step 後，都會用 decay=0.999 更新一份 ema_model。實際驗證與測試時，程式也是使用 EMA 權重，這通常能比單純使用最後一個 epoch 的權重更穩定。

### 4.4 訓練迴圈
訓練迴圈的流程如下：

1. 讀取 batch 與 labels
2. 將資料送入 GPU
3. 執行前向傳播
4. 計算 loss
5. backward 反向傳播
6. 梯度裁剪
7. optimizer step
8. scheduler step
9. EMA 更新
10. 累計 training loss 與 accuracy
11. 切換至 eval 模式計算 validation loss 與 accuracy

此外，程式也會在每個 epoch 存一份 checkpoint，並根據 validation accuracy 保存 best checkpoint 與 best EMA checkpoint。

### 4.5 Checkpoint 策略
訓練期間會產生以下模型檔案：

1. 每個 epoch 一份：Models/convnext_model_1.pth、Models/convnext_model_2.pth 等
2. 最佳 raw 權重：Models/convnext_model_best.pth
3. 最佳 EMA 權重：Models/convnext_model_best_ema.pth

best 模型的判定標準為 validation accuracy。這代表模型選擇是以泛化表現作為依據，而不是只看 training loss。

## 5. 評估方法
### 5.1 驗證階段
每個 epoch 結束時，都會在 validation set 上計算：

1. Validation loss
2. Validation accuracy

程式在驗證時使用 EMA 模型，這能降低參數震盪對驗證結果的影響。因此，validation accuracy 不只是訓練過程中的監控指標，也直接作為最佳 checkpoint 的依據。

### 5.2 測試階段
訓練完成後，程式會載入最佳 checkpoint，再在 test set 上做最終評估。測試階段會輸出：

1. Test loss
2. Test accuracy

同時也會收集所有預測結果與 ground truth，供後續 confusion matrix 與 classification report 使用。

### 5.3 混淆矩陣
程式使用 confusion_matrix(label_vec, pred_vec) 產生混淆矩陣，再轉為 pandas DataFrame 顯示。這能幫助我們觀察哪些類別最容易互相混淆，例如 cat 與 dog、deer 與 horse、ship 與 plane 等 CIFAR-10 常見混淆對。

### 5.4 Classification report
程式也會輸出 precision、recall、F1-score 與 support。這些指標能比單一 accuracy 更細緻地描述模型行為，特別是在不同類別表現不均時更有參考價值。

不過，這份 notebook 中 classification_report 的參數順序寫成 classification_report(pred_vec, label_vec, ...)，標準寫法應該是 y_true 在前、y_pred 在後。若不修正，precision、recall 與 F1 的解讀可能會出現偏差。這一點應列入報告的限制與修正建議。

## 6. 視覺化分析
本 notebook 的視覺化設計相當完整，包含以下部分：

1. Learning rate 曲線
2. Training/validation loss 曲線
3. Training/validation accuracy 曲線
4. 4 × 4 測試樣本預測展示
5. 50 張測試圖片的預測結果
6. 各類別錯誤分類樣本視覺化

這些圖表的用途如下：

1. Learning rate 曲線可檢查 scheduler 是否按預期平滑下降
2. Loss/accuracy 曲線可觀察是否收斂，以及是否有過擬合
3. 測試樣本圖可快速人工檢視單筆預測是否合理
4. 50 張預測圖可直觀看出模型在不同類別上的成功與失敗情形
5. 各類別錯誤分類圖可協助分析模型對哪些類別最不穩定

## 7. 結果與討論
由於 notebook 目前尚未執行，因此這份報告不能填入實際的 accuracy、loss、confusion matrix 數值或 classification report 指標。換句話說，目前能確認的是實驗流程與模型設計，而不是實驗結果本身。

若 notebook 執行完成，報告中應該補上以下內容：

1. 最佳 validation accuracy 與對應 epoch
2. 最終 test accuracy 與 test loss
3. 混淆矩陣中最常見的誤判對
4. 各類別 precision、recall、F1-score
5. loss 與 accuracy 曲線是否呈現穩定收斂

從設計角度來看，這份實驗具備相當完整的訓練與分析管線，因此即使目前沒有數值結果，仍然可以看出作者的目標是利用預訓練 vision backbone、穩定的優化策略與系統化的評估流程，來完成 CIFAR-10 分類任務。

## 8. 優點
1. 使用 pretrained ConvNeXt V2 tiny，能有效利用先驗特徵
2. 搭配 AdamW、label smoothing、weight decay 與 EMA，訓練策略完整
3. 使用混合精度與 torch.compile，可提升訓練效率
4. 評估面向完整，包含 accuracy、confusion matrix、classification report 與圖像檢視
5. 保存 best checkpoint 與 EMA checkpoint，方便後續比較與部署

## 9. 限制
1. 沒有額外資料增強，對 CIFAR-10 的泛化可能有限
2. validation/test 是從官方 test set 再切分，不是最標準的實驗切法
3. notebook 內 markdown 說明與實際 code 有不一致之處
4. classification_report 的參數順序可能有誤，需修正後才適合正式報告
5. notebook 尚未執行，因此沒有可直接引用的實驗結果數值

## 10. 未來改進方向
1. 加入更完整的資料增強，例如 RandomHorizontalFlip、RandomCrop、ColorJitter、RandAugment、MixUp 或 CutMix
2. 將 validation set 從 training set 中切出，保留完整官方 test set 進行最終測試
3. 修正 classification_report 的 y_true / y_pred 順序
4. 加入 early stopping 或更完整的模型選擇條件
5. 嘗試不同 backbone、不同 optimizer、不同 scheduler 的 ablation study
6. 在報告中補上正式的表格與圖表編號，讓文件更接近論文格式

## 11. 結論
本專案展示了一個以 ConvNeXt V2 tiny 為基礎的 CIFAR-10 影像分類流程。整體設計重點不是從零建立傳統 CNN，而是利用預訓練模型、穩定化訓練技巧與完整的評估分析流程，建立一個較成熟的影像分類實驗框架。

雖然目前 notebook 尚未執行，無法提供最終性能數值，但從程式結構來看，這份實作已經包含資料處理、模型建構、訓練、驗證、測試與錯誤分析等完整步驟。若補上實際跑出的結果，這份報告即可直接作為一份完整且一致的實驗說明文件。
