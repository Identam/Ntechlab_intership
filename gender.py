import sys
import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
#Если доступна видеокарта, обучение будет идти на первой, из доступных
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESCALE_SIZE = 224
#Загрузка модели
model = torch.load('entire_model.pt',DEVICE)
#Перемещение на видеокарту(по возможности)
model = model.to(DEVICE)

def predict_one_sample(model, inputs, device=DEVICE):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs.to(DEVICE)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=1).numpy()
    return probs

def prepare_im(im):
  """Преобразование картинки в numpy тензор"""
  ex_img = Image.open(im)
  ex_img.load()
  ex_img = np.array(ex_img.resize((RESCALE_SIZE, RESCALE_SIZE)))
  transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
          ])
  ex_img = np.array(ex_img / 255, dtype='float32')
  ex_img = transform(ex_img)
  return ex_img

def make_final_pred(model, dir):
  """Функция для предсказания по картинкам, находящимся в папке dir
     выводит результат в json файл в формате: {'img1.jpg': 'male', ...}
  """
  DIR = Path(dir)
  files = sorted(list(DIR.rglob('*.jpg')))
  fil = []
  preds = []
  for img in files:
    y_ = np.argmax(predict_one_sample(model, prepare_im(Path(img)).unsqueeze(0)), -1)
    fil.append(str(img))
    if y_[0] == 0:
      preds.append('female')
    else:
      preds.append('male')
    
    data = dict(zip(fil, preds))

    with open("process_results.json", 'w') as write_file:
      json.dump(data, write_file)


if __name__ == "__main__":
  DIRR = sys.argv[1]
  make_final_pred(model, DIRR)