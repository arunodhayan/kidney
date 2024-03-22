from flask import Flask, request, render_template, redirect, url_for
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import base64
import io
from PIL import Image
import torch.nn.functional as F
import numpy as np
import timm
from hireachial import *
app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available()  else 'cpu')



class CustomModel(nn.Module):
    def __init__(self,name, pretrained=False):
        super().__init__()

        model = timm.create_model(name, pretrained=pretrained, in_chans=3)
        clsf = model.default_cfg['classifier']
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.fc = nn.Linear(n_features, 4)
        self.model = model 

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
        

# Load your model
model     = CustomModel('efficientnet_b1',pretrained = False)
model = model.to(device)
model.load_state_dict(torch.load('best_epoch-00.bin', map_location=torch.device('cpu')))

model.eval()

# Albumentations transform
transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=(0), std=(1)),
    ToTensorV2(),
])

# Class labels
class_labels = {
    0: 'Tumor',
    1: 'Cyst',
    2: 'Stone',
    3: 'Normal',
}




def read_image(file):
    """Read and prepare the image for prediction."""
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image)
    # Convert the PIL Image to a numpy array and ensure RGB
    if image.shape[-1] == 4:  # Check if image is RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = read_image(file)
            
            transformed = transform(image=image)["image"].unsqueeze(0)
            transformed = transformed.to(device)
            # Make prediction
            with torch.no_grad():
                superclass_pred = model(transformed)
                probabilities = F.softmax(superclass_pred, dim=1)
                indices = torch.max(probabilities, dim=1)
                prob = indices[0].cpu().detach().numpy()
                prob = prob[0]*100
                prob = round(prob, 2)  # Round to two decimal points
                superclass_pred = superclass_pred.argmax().item()  # Get predicted index for superclass
                #subclass_pred = subclass_pred.argmax().item()  # Get predicted index for subclass

            # Translate indices to class labels
            superclass_label = class_labels[superclass_pred]
            #subclass_label = class_labels_sub[subclass_pred]  # Assuming the same label dictionary applies

            # Encode the original image for HTML display
            img_io = io.BytesIO()
            pil_img = Image.fromarray(image)
            pil_img.save(img_io, 'JPEG', quality=70)
            img_io.seek(0)
            base64_img = base64.b64encode(img_io.getvalue()).decode('ascii')

            return render_template('index.html', image=base64_img, superclass=superclass_label,prob=prob)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

