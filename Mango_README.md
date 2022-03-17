# AI CUP 2020 愛文芒果三類等級辨識競賽
#### 隊伍: ML@NTUT AIOTLab208
#### 成員: 鍾岳璁,鄒昌諭,王鼎鈞,林芷羽

## 大綱
<ol>
  <li><a href="#環境">環境</a></li>
  <li><a href="#資料處理">資料處理</a></li>
  <li><a href="#模型架構">模型架構</a></li>
  <li><a href="#訓練方式">訓練方式</a></li>
  <li><a href="#分析結論">分析結論</a></li>
  <li><a href="#程式碼">程式碼</a></li>
  <li><a href="#使用的外部資源與參考文獻">使用的外部資源與參考文獻</a></li>
</ol>

## 環境
```python
OS:Ubuntu 18.04
Python:3.8.3

套件:
numpy==1.18.2
torch==1.4.0
scikit_image==0.16.2
matplotlib==2.2.3
tqdm==4.43.0
easydict==1.9
Pillow==7.1.2
tensorboardX

預訓練模型: Tianxiaomo / Yolov4 on pytorch framework
```

## 資料處理
僅使用官方提供之train & dev資料集，train當作訓練訓練集，dev當作驗證資料集。
將官方給的標註格式轉換成本次訓練模型可接受之格式，圖片預處理的部分使用mirror padding, 並resize成608x608的大小。

## 模型架構
使用yolov4 pytorch版本來當作本次的辨識模型。

## 訓練方式
圖片的輸入大小使用608*608,Training epoch最終為105個

## 分析結論
因官方提供的dataset有提供bounding boxes，為了善用這些資訊，就使用目前在物件偵測上公認性能較好的yolov4模型來進行本次競賽。
其實訓練過程中有發現5大類中，除D3與D4的資料較多外，其他三類的資料及都較少，導致這三類的準確率也較低，若可以針對這三類做更多的資料擴充，或許對整體的準確率會有幫助。
另外模型部分可以考慮選擇用scaled yolov4,對於辨識的準確率也應該會有幫助，也可考慮同時使用多種模型，再根據每個模型對那些類型的照片準確率較高，來選擇最後輸出要參考哪個模型。

## 程式碼
標註轉換程式: [連結](https://drive.google.com/file/d/1h-NbkqMOYuAds1_RB_ShBh2HUN17ZxpD/view?usp=sharing)
已將本次專案所有資料上傳至google雲端硬碟，如下:
主體程式： [Folder](https://drive.google.com/drive/folders/1hLIgkbzQ1kvGgrUHH1hy8xIM-oLwOAVQ?usp=sharing) | [Zip file](https://drive.google.com/file/d/10Gy2vA5WDsmINkQp1rH-ENbJnZq47CxY/view?usp=sharing)
基於[Tianxiaomo](https://github.com/Tianxiaomo)/  [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) 此框架來進行本次模型的訓練與預測。
原作者的sigmoid function計算時會有overflow的問題，自行修改過相關程式碼如下，檔案位置為Yolov4_pytorch/pytorch-YOLOv4/tool/utils.py。
```python
def sigmoid(x):
    xc=[]
    for v in x:
        vc = np.exp(v) / (1.+np.exp(v)) if v<0 else 1.0 / (np.exp(-v) + 1.)
        xc.append(vc)
    return np.array(xc)
```

train的方式為執行Yolov4_pytorch/pytorch-YOLOv4/train.py此檔案,command如下
```python
CUDA_VISIBLE_DEVICES=2,3 python train.py -b 8 -s 1 -g 0 -l 0.001 -pretrained ./yolov4.conv.137.pth -classes 5 -dir ./train -epochs [epochs]
```

每個epoch訓練完的權重會存在checkpoints/裡面
測試集的照片可用Yolov4_pytorch/ resize_testimg.py 此程式來進行預處理
測試的程式有自己改寫成輸出競賽規定之格式，程式路徑為Yolov4_pytorch/pytorch-YOLOv4/pred_yu.py, 執行的 command如下
```python
CUDA_VISABLE_DEVICES=0,1 python models.py 5 checkpoints/[weight.pth] [img_path] valid/_classes.txt
```

## 使用的外部資源與參考文獻
1. Bochkovskiy, Wang and Liao (2020) YOLOv4: Optimal Speed and Accuracy of Object Detection ([arXiv:2004.10934](https://arxiv.org/abs/2004.10934))
1. [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)