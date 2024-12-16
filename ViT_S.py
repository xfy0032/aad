import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer, ViTConfig
import numpy as np
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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


class CustomDataset2(Dataset):
    def __init__(self, imgs_data, label_data, transform=None):
        self.imgs = imgs_data
        self.labels = label_data
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        #print(self.imgs[idx].shape)
        temp=self.imgs[idx]*10000000
        max_t=np.max(temp)
        min_t=np.min(temp)
        temp = (temp-min_t)/(max_t-min_t)
        temp=temp.reshape(128, 32, 1)
        img = numpy_to_pil(temp)

        if self.transform:
            img = self.transform(img)
        return img, label
        
        

class CustomDataset(Dataset):
    def __init__(self, imgdata_path, labels_path, transform=None):
    
        self.data=[]
        for path in imgdata_path:
            temp = np.load(path)
            self.data.append(temp)
        self.imgs = np.concatenate(self.data, axis=0)
        
        self.data=[]
        for path in labels_path:
            temp = np.load(path)
            self.data.append(temp)
        self.labels = np.concatenate(self.data, axis=0)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        #print(self.imgs[idx].shape)
        temp=self.imgs[idx]*10000000
        max_t=np.max(temp)
        min_t=np.min(temp)
        temp = (temp-min_t)/(max_t-min_t)
        temp=temp.reshape(128, 32, 1)
        img = numpy_to_pil(temp)

        if self.transform:
            img = self.transform(img)
        return img, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[125, 125, 125], std=[0.5, 0.5, 0.5]),
])



def main(path):
  train_folders  =  ['SA','SB','SC','SD','SE','SF','SG','SH']
  train_data_path=[]
  train_labels_path=[]
  
  
  for i in range(len(train_folders)):
          train_data_path.append('/dataset/xfy/AAD/CS/'+path+train_folders[i]+'.npy')        # 替换为实际训练数据路径
          train_labels_path.append('/dataset/xfy/AAD/CS/label/'+train_folders[i]+'.npy')  
  
  
  
 
  all_data=CustomDataset(train_data_path, train_labels_path, transform=transform)
  print(all_data.imgs.shape, all_data.labels.shape)
  
  train_data, valid_data, train_label, valid_label= train_test_split(all_data.imgs, all_data.labels, test_size=0.2, random_state=42)
  
  
  
  
  #train_data_path.append('/dataset/xfy/AAD/SS/audio-only/SA/'+'train_eeg.npy')  
  #train_labels_path.append('/dataset/xfy/AAD/SS/audio-only/SA/'+'train_label.npy')  
  #val_data_path   =  ['/dataset/xfy/AAD/SS/audio-only/SA/val_eeg.npy'] 
  #val_labels_path =  ['/dataset/xfy/AAD/SS/audio-only/SA/val_label.npy']
  
  
  # 加载预训练模型
  
  #model_name = "google/vit-base-patch16-224"
  #feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
  #model = ViTForImageClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
  
  
  
  # 初始化ViT模型配置
  config = ViTConfig(
      hidden_size=768,
      num_hidden_layers=12,
      num_attention_heads=24,
      intermediate_size=3072,
      image_size=128,
      patch_size=32,
      num_channels=3,
      num_labels=2,
  )
  
  # 初始化ViT模型
  model = ViTForImageClassification(config)
  
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total number of parameters: {total_params}")
  
  
  #save_directory = "./my_saved_model"
  #model.save_pretrained(save_directory)
  #feature_extractor.save_pretrained(save_directory)
  
  
  # 创建数据集实例
  
  train_dataset = CustomDataset2(train_data, train_label,transform=transform)
  
  val_dataset = CustomDataset2(valid_data, valid_label, transform=transform)
  
  
  # 数据加载器
  
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  
  val_loader = DataLoader(val_dataset, batch_size=32)
  
  
  
  if not os.path.exists('./my_vit_result/'+path):

    # 如果文件夹不存在，创建它

    os.makedirs('./my_vit_result/'+path)

  
  
  
  # 微调参数设置
  
  training_args = TrainingArguments(
  
      output_dir='./my_vit_result/'+path,          # 输出目录
  
      num_train_epochs=200,              # 总轮数
  
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
      labels = p.label_ids
      accuracy = accuracy_score(labels, preds)
      with open("e.txt", "a") as f:
          f.write(f"Accuracy: {accuracy}\n")
      return {"accuracy": accuracy}
  
  
  # 训练器
  
  trainer = Trainer(
  
      model=model,
  
      args=training_args,
  
      train_dataset=train_dataset,
  
      eval_dataset=val_dataset,
  
      data_collator=lambda data: {'pixel_values': torch.stack([d[0] for d in data]), 'labels': torch.tensor([d[1] for d in data])},
  
      compute_metrics=compute_metrics,
  
  )
  
  
  # 开始训练
  
  trainer.train()
  
  
  
if __name__ == '__main__':
    path=['audio-only/','audio-video/']

    train_folders  =  ['SA','SB','SC','SD','SE','SF','SG','SH']
    #train_folders  =  ['SE','SF']
    for p in range(len(path)):
        temp = path[p]
        print(temp)
        main(temp)

