import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms
import json

# -------------------------------
# [1] 기본 설정
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# [2] 전처리 함수 (추론용)
# -------------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])

# -------------------------------
# [3] 클래스 매핑 불러오기
# -------------------------------
@st.cache_resource
def load_label_map():
    with open("classes_to_idx.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

# -------------------------------
# [4] 모델 정의 및 불러오기
# -------------------------------
def load_model(num_classes):
    model = models.resnet34(weights=None)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, num_classes)
    )

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# -------------------------------
# [5] Streamlit UI
# -------------------------------
st.title("🌟 닮은꼴 연예인 분류기 🌟")
st.chat_input("성능은 망가져 내렸으니 재미로만 부탁드립니다.ㅠ^ㅠ")
# 파일 업로드 시
uploaded_file = st.file_uploader("사진을 업로드해주세요!", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image = ImageOps.exif_transpose(image)
    st.image(image, caption="업로드한 사진", use_container_width=True)

    # 클래스 이름 불러오기
    idx_to_class = load_label_map()
    num_classes = len(idx_to_class)

    # 모델 불러오기
    model = load_model(num_classes)

    # 이미지 전처리 및 예측
    image_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_name = idx_to_class[pred_idx]

    st.subheader(f"📸 당신은 **{pred_name}** 님과 닮았습니다!")
