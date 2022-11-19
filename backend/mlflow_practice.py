import torch.nn as nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import mlflow 
import warnings
import random
import copy
import argparse
from importlib import import_module




def seed_everything(seed): # Seed 고정
    torch.manual_seed(seed) # torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True # 딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) # numpy를 사용할 경우 고정
    random.seed(seed) # 파이썬 자체 모듈 random 모듈의 시드 고정


class Net(nn.Module): # 모델 클래스 정의 부분
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,100) # MNIST 데이터셋이 28*28로 총 784개의 픽셀로 이루어져있기 때문에 784를 입력 크기로 넣음.
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100,100) # 은닉층
        self.fc3 = nn.Linear(100,10) # 출력층 0~9까지 총 10개의 클래스로 결과가 나옴.

    def forward(self, x): # 입력층 -> 활성화 함수(ReLU) -> 은닉층 -> 활성화 함수(ReLU) -> 출력층
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.fc3(x4)

        return x5

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

input_example = np.array([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118,
         0.0706, 0.0706, 0.0706, 0.4941, 0.5333, 0.6863, 0.1020, 0.6510, 1.0000,
         0.9686, 0.4980, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1176, 0.1412, 0.3686, 0.6039,
         0.6667, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.8824, 0.6745, 0.9922,
         0.9490, 0.7647, 0.2510, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1922, 0.9333, 0.9922, 0.9922,
         0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9843, 0.3647, 0.3216,
         0.3216, 0.2196, 0.1529, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.8588, 0.9922,
         0.9922, 0.9922, 0.9922, 0.9922, 0.7765, 0.7137, 0.9686, 0.9451, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3137,
         0.6118, 0.4196, 0.9922, 0.9922, 0.8039, 0.0431, 0.0000, 0.1686, 0.6039,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0549, 0.0039, 0.6039, 0.9922, 0.3529, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.5451, 0.9922, 0.7451, 0.0078, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0431, 0.7451, 0.9922, 0.2745,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1373, 0.9451,
         0.8824, 0.6275, 0.4235, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.3176, 0.9412, 0.9922, 0.9922, 0.4667, 0.0980, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.1765, 0.7294, 0.9922, 0.9922, 0.5882, 0.1059, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0627, 0.3647, 0.9882, 0.9922, 0.7333,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.9765, 0.9922,
         0.9765, 0.2510, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1804, 0.5098, 0.7176, 0.9922,
         0.9922, 0.8118, 0.0078, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.1529, 0.5804, 0.8980, 0.9922, 0.9922,
         0.9922, 0.9804, 0.7137, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0941, 0.4471, 0.8667, 0.9922, 0.9922, 0.9922,
         0.9922, 0.7882, 0.3059, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0902, 0.2588, 0.8353, 0.9922, 0.9922, 0.9922, 0.9922,
         0.7765, 0.3176, 0.0078, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0706, 0.6706, 0.8588, 0.9922, 0.9922, 0.9922, 0.9922, 0.7647,
         0.3137, 0.0353, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.2157, 0.6745, 0.8863, 0.9922, 0.9922, 0.9922, 0.9922, 0.9569, 0.5216,
         0.0431, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.5333, 0.9922, 0.9922, 0.9922, 0.8314, 0.5294, 0.5176, 0.0627,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000]],dtype=np.float32) # input 값의 예제

input_schema = Schema([
    TensorSpec(np.dtype(np.float32),(-1,784))
]) # input 값의 스키마 예제 [-1,784] 사이즈를 가진 numpy 배열 형태이다. 여기서 -1은 임의의 수를 의미한다. batch_size 까지 커질 수 있다.

output_schema = Schema([
    TensorSpec(np.dtype(np.float32),(-1,10))
]) # output 값의 스키마 예제 [-1,10] 총 10개의 클래스의 각각의 유사도를 출력값으로 내놓는다. softmax 등을 활용하여 0~9까지의 클래스의 정답일 확률로 표시할 수 있다.

signature = ModelSignature(inputs=input_schema,outputs=output_schema) # MLflow의 시그니쳐를 정의하는 부분이다. MLflow.log_model을 할 때 사용되며 Model의 입력값, 출력값 스키마를 넣어준다.


def train(args): # args를 통해 우리가 직접 넣어주는 hyperparameter 입력값들을 인자로 받는다.
  seed_everything(args.seed) # seed를 고정하는 함수 호출

  download_root = 'MNIST_data/' # 데이터 다운로드 경로

  train_dataset = datasets.MNIST(root=download_root,
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True) # 학습 dataset 정의
                          
  test_dataset = datasets.MNIST(root=download_root,
                          train=False,
                          transform=transforms.ToTensor(), 
                          download=True) # 평가 dataset 정의

  batch_size = args.batch_size # 배치 사이즈 정의. 데이터셋을 잘개 쪼개서 묶음으로 만드는 데 기여한다.
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 학습 데이터셋을 배치 사이즈 크기만큼씩 잘라서 묶음으로 만든다. 묶음의 개수는 train_dataset / batch_size
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # train_dataloader와 마찬가지

  model = Net() # 모델 정의
  loss_function = nn.CrossEntropyLoss() # 실제 정답과 예측값의 차이를 수치화해주는 함수.
  learning_rate = args.lr # AdamW:0.001, SGD:0.1, Adam:0.001

  if args.optimizer == 'SGD': # optimizer가 SGD일 경우
    opt_module = getattr(import_module("torch.optim"),args.optimizer) # torch.optim.SGD의 형태를 만들어주는 부분.
    optimizer = opt_module(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr = learning_rate,
      weight_decay = 1e-4,
      momentum=0.9
    ) # optimizer = torch.optim.SGD(lr,weight_decay,momentum) 의 형태
  
  else: # SGD가 아닌 다른 경우
    opt_module = getattr(import_module("torch.optim"),args.optimizer)
    optimizer = opt_module(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr = learning_rate,
      weight_decay=5e-4,
    ) # SGD와는 다르게 Adam, AdamW의 경우에는 momentum이란 인자가 사용되지않으므로 optimizer = torch.AdamW(lr,weight_decay) 의 형태.
  
  epochs = args.epochs # 얼마나 학습할 지 정하는 인자.

  model_name = args.model_name # 모델 버전 관리를 위해 모델의 이름을 정의해주는 부분.

  warnings.filterwarnings(action='ignore') # 경고 메시지 출력안하는 코드.
  experiment_name = args.experiment_name # 실험명, 실험관리를 용이하게 해줍니다. 

  if not mlflow.get_experiment_by_name(experiment_name): 
    mlflow.create_experiment(name=experiment_name)

  mlflow.set_tracking_uri('http://127.0.0.1:5001') # 로컬 서버에 실행을 기록하기 위해 함수 호출

  mlflow.set_experiment(experiment_name) # 위에서 정의한 실험명을 mlflow에 적용하는 코드.
  experiment = mlflow.get_experiment_by_name(experiment_name) # experiment_id 를 가져오기 위해 get_experiment_by_name 호출한다.


  best_accuracy = 0 # 평가 지표
  model.zero_grad() # 학습 전에 모델의 모든 weight, bias 값들을 초기화

  with mlflow.start_run(experiment_id = experiment.experiment_id,run_name=args.run_name) as run: # mlflow를 시작하는 코드. experiment_id 와 run_name을 적용했는데 이 부분은 자유롭게 적용할 수 있다.
    for epoch in range(epochs):
      
      model.train() # 학습
      train_accuracy = 0 # metric
      train_loss = 0 # metric

      for images, labels in train_loader:
        images = images.reshape(batch_size,784)
        image = model(images)
        loss = loss_function(image,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = torch.argmax(image,1)
        correct = (prediction == labels)
        train_accuracy+= correct.sum().item() / len(train_dataset)
        train_loss += loss.item() / len(train_loader)

      model.eval() # 평가
      val_accuracy = 0 # metric
      val_loss = 0 # metric

      for images,labels in test_loader:
        images = images.reshape(batch_size,784)
        image = model(images)
        loss = loss_function(image,labels)
        
        correct = (torch.argmax(image,1) == labels)
        val_accuracy += correct.sum().item() / len(test_dataset)
        val_loss += loss.item() / len(test_loader)
      
      print(f'epoch: {epoch}/{epochs} train_loss: {train_loss:.5} train_accuracy: {train_accuracy:.5} val_loss: {val_loss:.5} val_accuracy: {val_accuracy:.5}')
      # 학습과 평가 부분은 Mlflow에서 크게 생각하지 않아도 되니 주석을 따로 달지 않았습니다.

      if best_accuracy < val_accuracy: # 성능이 가장 좋은 모델로 갱신
        best_accuracy = val_accuracy
        torch.save(model.state_dict(),'best_model.pt')
        best_model = copy.deepcopy(model)
        print(f"===========> Save Model(Epoch: {epoch}, Accuracy: {best_accuracy:.5})")
        
      mlflow.log_param('learning-rate',learning_rate) # mlflow.log_param 을 사용하여 MLflow에 hypter parameter들을 기록할 수 있다.
      mlflow.log_param('epoch',epochs)
      mlflow.log_param('batch_size',batch_size)
      mlflow.log_param('optimizer',optimizer)
      mlflow.log_param('seed',args.seed)
      mlflow.log_param('loss_function',loss_function)

      mlflow.log_metric('train_accuracy',train_accuracy) # mlflow.log_metric을 사용하여 MLflow에 성능평가를 위한 metric을 기록할 수 있다.
      mlflow.log_metric('train_loss',train_loss)
      mlflow.log_metric('valid_accuracy',val_accuracy)
      mlflow.log_metric('valid_loss',val_loss)

      print("--------------------------------------------------------------------------------------------")
    mlflow.pytorch.log_model(best_model,'best_model',signature=signature,input_example=input_example,registered_model_name=model_name) # mlflow.log_model을 사용하여 모델을 mlflow에 저장할 수 있습니다.
    # 가장 중요한 부분은 registered_model_name 이 부분인데 이 코드를 통해서 ui에서 register model을 눌러서 모델을 저장하는 것이 아니라 바로 모델을 저장할 수 있다.
    # 모델이 없는 경우 새로 만들어서 버전 1이 되고, 이미 있는 모델의 경우에는 새로운 버전을 업데이트한다.

  print('best model saved to mlflow server')
  mlflow.end_run() # Mlflow 기록 종료.

if __name__ == '__main__':

  parser = argparse.ArgumentParser() # train.sh 파일의 script에 추가하여 사용하면 되는 부분.

  parser.add_argument('--epochs',type=int,default=10) 
  parser.add_argument('--batch-size',type=int,default=100)
  parser.add_argument('--optimizer',type=str,choices=['SGD','Adam','AdamW'],default='SGD')
  parser.add_argument('--seed',type=int,default=42)
  parser.add_argument('--lr',type=float,default=0.01)
  parser.add_argument('--model_name',type=str,default='basic_model')
  parser.add_argument('--experiment_name',type=str,default='sojt_project')
  parser.add_argument('--run_name',type=str,default='boom')
  args = parser.parse_args()

  train(args)