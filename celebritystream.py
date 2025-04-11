import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import json

# -------------------------------
# [1] 기본 설정
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# [2] 전처리 함수 (추론용)
# -------------------------------
testset_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.2)
])

# -------------------------------
# [3] 클래스 매핑 불러오기
# -------------------------------
@st.cache_resource
def load_label_map():
    with open("class_to_idx.json", "r", encoding='utf-8') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

# -------------------------------
# [4] 모델 불러오기
# -------------------------------
@st.cache_resource
def load_model(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# -------------------------------
# [5] Streamlit UI
# -------------------------------
st.title("✨ 닮은꼴 연예인 AI ✨")
uploaded_file = st.file_uploader("당신의 얼굴 사진을 업로드하세요!", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    # 예측
    idx_to_class = load_label_map()
    model = load_model(num_classes=len(idx_to_class))

    image_tensor = testset_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image_tensor)
        pred_idx = pred.argmax(dim=1).item()

    pred_name = idx_to_class[pred_idx]
    st.subheader(f"📸 당신은 **{pred_name}** 님과 닮았습니다!")
