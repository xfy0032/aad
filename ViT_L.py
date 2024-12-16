import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer, ViTConfig
import numpy as np
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def numpy_to_pil(image_np):
    """
    将NumPy数组转换为PIL Image对象。
    参数:
    image_np (numpy.ndarray): NumPy数组，通常为(H, W, C)格式。
    返回:
    PIL.Image.Image: 转换后的PIL图像。
    """
    # 确保数组格式为(H, W, C)，如果通道数为1，则可能需要调整
    if len(image_np.shape) == 3 and image_np.shape[2] == 1:  # 如果是灰度图，扩展通道维度
        image_np = np.repeat(image_np, 3, axis=2)
    elif len(image_np.shape) == 2:  # 如果是二维数组，假定为灰度图，并转换为RGB
        image_np = np.stack((image_np,)*3, axis=-1)
    # 将数组数据类型转换为uint8，范围从0-255
    image_np = (image_np * 255).astype(np.uint8)
    # 根据数组形状创建PIL图像
    image_pil = Image.fromarray(image_np)
    return image_pil

# 自定义数据集类

class CustomDataset(Dataset):
    def __init__(self, imgdata_path, labels_path=None, transform=None):
    
        self.data=[]
        for path in imgdata_path:
            temp = np.load(path)
            self.data.append(temp)
        self.imgs = np.concatenate(self.data, axis=0)
        
        if labels_path:
            self.data=[]
            for path in labels_path:
                temp = np.load(path)
                self.data.append(temp)
            self.labels = np.concatenate(self.data, axis=0)
        else:
            self.labels=None
        
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    
    def __getitem__(self, idx):
        
        #label = self.labels[idx]
        #print(self.imgs[idx].shape)
        temp=self.imgs[idx]*10000000
        max_t=np.max(temp)
        min_t=np.min(temp)
        temp = (temp-min_t)/(max_t-min_t)
        temp=temp.reshape(128, 32, 1)
        img = numpy_to_pil(temp)

        if self.transform:
            img = self.transform(img)
            
        if self.labels is not None:
            #print(self.labels[idx])
            return img, self.labels[idx]
            
        else:
            return img, [0]
        #return img, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[125, 125, 125], std=[0.5, 0.5, 0.5]),
])


def main(path,id):
  train_folders  =  ['SA','SB','SC','SD','SE','SF','SG','SH']
  train_data_path=[]
  train_labels_path=[]
  
  
  #for i in range(len(train_folders)):
  #    if i==id:
  #        continue
  #    else:
  #        train_data_path.append('/dataset/xfy/ADD/CS/'+path+train_folders[i]+'.npy')        # 替换为实际训练数据路径
  #        train_labels_path.append('/dataset/xfy/ADD/CS/label/'+train_folders[i]+'.npy')  
  
  #val_data_path = ['/dataset/xfy/ADD/CS/'+path+train_folders[id]+'.npy'] # 验证集图像路径列表
  #val_labels_path =  ['/dataset/xfy/ADD/CS/label/'+train_folders[id]+'.npy']
  
  
  train_data_path.append('/dataset/xfy/ADD/SS/'+path+train_folders[id]+'/train_eeg.npy')  
  train_labels_path.append('/dataset/xfy/ADD/SS/'+path+train_folders[id]+'/train_label.npy')  
  val_data_path   =  ['/dataset/xfy/ADD/SS/'+path+train_folders[id]+'/val_eeg.npy'] 
  val_labels_path =  ['/dataset/xfy/ADD/SS/'+path+train_folders[id]+'/val_label.npy']
  
  
  test_data_path   =  ['/dataset/xfy/ADD/SS/test/preprocessed_eeg/'+path+train_folders[id]+'/test_eeg.npy'] 
  
  
  # 加载预训练模型
  #model_name = "google/vit-base-patch16-224"
  #feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
  #model = ViTForImageClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
  
  
  
  # 初始化ViT模型配置
  config = ViTConfig(
    hidden_size=1024,           # 模型的隐藏层大小
    num_hidden_layers=24,       # 隐藏层数量
    num_attention_heads=16,     # 注意力头的数量
    intermediate_size=4096,     # 中间层大小
    hidden_act="gelu",          # 激活函数
    layer_norm_eps=1e-12,       # 层归一化的epsilon
    dropout_rate=0.1,           # dropout比例
    attention_probs_dropout_prob=0.1, # 注意力概率的dropout比例
    initializer_range=0.02,     # 初始化范围
    image_size=128,             # 输入图像大小
    patch_size=32,              # 图像patch的大小
    num_channels=3,             # 输入图像的通道数
    num_labels=2                # 分类标签数
  )
  
  # 初始化ViT模型
  #model = ViTForImageClassification(config)
  
  path_='/data0/home/xfy/Projects/ADD/vit7/my_vit_result/'+path+train_folders[id]+'/'
  save_dir=None
  for root, dirs, files in os.walk(path_, topdown=True):
    for name in dirs:
        save_dir = os.path.join(root, name)
        print("##########",test_data_path[0])
        print("##########",os.path.join(root, name))
        break 
  
  
  model = ViTForImageClassification.from_pretrained(save_dir)



  
  # 创建数据集实例
  train_dataset = CustomDataset(train_data_path, train_labels_path, transform=transform)
  val_dataset = CustomDataset(val_data_path, val_labels_path, transform=transform)
  
  test_dataset = CustomDataset(test_data_path, None, transform=transform)
  
  
  # 数据加载器
  #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  #val_loader = DataLoader(val_dataset, batch_size=32)
  
  
  result_save='./track1_Z-LAB-AAD_submission_testset/'+path
  if not os.path.exists(result_save):
    # 如果文件夹不存在，创建它
    os.makedirs(result_save)

  
  
  
  # 微调参数设置
  
  training_args = TrainingArguments(
  
      output_dir='./my_vit_result/'+path+train_folders[id],          # 输出目录
  
      num_train_epochs=150,              # 总轮数
  
      per_device_train_batch_size=256,  # 每个GPU的训练批次大小
  
      per_device_eval_batch_size=128,   # 每个GPU的评估批次大小
  
      warmup_steps=100,                # 预热步数
  
      weight_decay=0.1,               # 权重衰减
  
      logging_dir='./logs',            # 日志目录
  
      evaluation_strategy="epoch",     # 每个epoch评估一次
      
      save_strategy="epoch",
      
      load_best_model_at_end=True,              # 在训练结束时加载最佳模型
      
      metric_for_best_model="accuracy",         # 使用的评估指标
      
      greater_is_better=True,                   # 指标是否越大越好
      
      save_total_limit=1                        # 最多保存的检查点数
  
  )
  
  
  def compute_metrics(p):
      preds = np.argmax(p.predictions, axis=1)
      print('***************************************   ',preds.shape,sum(preds))
      labels = p.label_ids
      accuracy = accuracy_score(labels, preds)
      #with open("e.txt", "a") as f:
      #    f.write(f"Accuracy: {accuracy}\n")
      print("accuracy: ", accuracy)
      return {"accuracy": accuracy}
  
  
  # 训练器
  
  def mydata_collator(features):
        pixel_values = torch.stack([f[0] for f in features])
        print(features[0])
        print(len(features[0]),print(len(features)))
        if len(features[1]) > 1:
            labels = torch.tensor([f[1] for f in features])
            return {'pixel_values': pixel_values, 'labels': labels}
        return {'pixel_values': pixel_values}
  
  
  trainer = Trainer(
  
      model=model,
  
      args=training_args,
  
      #train_dataset=train_dataset,
  
      #eval_dataset=val_dataset,
  
      data_collator = lambda data: {'pixel_values': torch.stack([d[0] for d in data]), 'labels': torch.tensor([d[1] for d in data])},
  
      compute_metrics=compute_metrics,
  
  )
  
  
  # 开始训练
  
  #trainer.train() 
  # Perform evaluation
  print("Evaluating...")
  #evaluation_results = trainer.evaluate()
  #print(f"Evaluation results: {evaluation_results}")

  # Perform predictions on test set
  print("Predicting on test set...")
  predictions = trainer.predict(test_dataset)
  predicted_labels = np.argmax(predictions.predictions, axis=1)
  
  print("============================================================================")
  print(len(predicted_labels),len(test_dataset.imgs))
  
  #allsum=0
  #for k in range(len(predicted_labels)):
  #    if(predicted_labels[k]==val_dataset.labels[k]):
  #        allsum=allsum+1
  #print("result:  ",  float(allsum)/float(len(predicted_labels)))
  
  
  with open(result_save+"/output_"+train_folders[id]+".txt", "w") as f:
          for i, label in enumerate(predicted_labels):
              f.write(f'Segment_ID: {i} Label: {label}\n')
  
  
  # Save predictions
  #np.save(f'./my_vit_result/{path}/test_predictions.npy', predicted_labels)
  #print(f"Predictions saved to ./my_vit_result/{path}/test_predictions.npy")
  
  
  
if __name__ == '__main__':

    path = ['audio-only/','audio-video/']

    train_folders  =  ['SA','SB','SC','SD','SE','SF','SG','SH']
    
    for p in range(len(path)):
        temp = path[p]

        for i in range(len(train_folders)):
            train_folder=temp+train_folders[i]
            print(train_folder)
            main(temp,i)
