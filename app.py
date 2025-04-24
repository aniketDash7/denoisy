import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import io


st.set_page_config(page_title="Denoisy", layout="centered")
st.markdown(
    """
    <style>
        .stButton>button {
            border-radius: 8px;
            padding: 0.5em 1em;
            background: linear-gradient(90deg, #4e54c8, #8f94fb);
            color: white;
            font-weight: 600;
        }
        .stFileUploader {
            padding-bottom: 1em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        return self.l1(self.vgg(sr), self.vgg(hr))

# Residual Block with Scaling
class ResidualBlock(nn.Module):
    def __init__(self, channels, scale=0.1):
        super().__init__()
        self.scale = scale
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.scale * self.block(x)


class SuperResolutionNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=128, num_res_blocks=16):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_features) for _ in range(num_res_blocks)
        ])

        self.mid_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        initial_feat = self.initial(x)
        x = self.res_blocks(initial_feat)
        x = self.mid_conv(x) + initial_feat  # Long skip connection
        x = self.upsample(x)
        x = self.output(x)
        return x


@st.cache_resource
def load_model():
    model = SuperResolutionNet()
    model.load_state_dict(torch.load("denoiser_second.pth", map_location="cpu"))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

# --- App UI ---
st.title("🌙 Denoisy")
st.subheader("Upload a low-light photo to denoise and upscale it ✨")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("🔍 Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_column_width=True)


    transform = T.Compose([
        T.Resize((160, 256)),
        T.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu().clamp(0, 1)

    output_image = T.ToPILImage()(output_tensor)

    with col2:
        st.image(output_image, caption="Enhanced", use_column_width=True)

    st.success("Enhancement complete!")
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    st.download_button("Download Result", data=buf.getvalue(), file_name="enhanced.png", mime="image/png")
