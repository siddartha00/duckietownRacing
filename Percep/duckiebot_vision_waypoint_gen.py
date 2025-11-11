"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


__all__ = ['FastSCNN', 'get_fast_scnn']


class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


# def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
#     acronyms = {
#         'pascal_voc': 'voc',
#         'pascal_aug': 'voc',
#         'ade20k': 'ade',
#         'coco': 'coco',
#         'citys': 'citys',
#     }
#     model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
#     if pretrained:
#         if(map_cpu):
#             model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
#         else:
#             model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
#     return model
def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    model = FastSCNN(6, **kwargs)
    if pretrained:
        if(map_cpu):
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
    return model

# --- Paths ---
model_path = "Percep/fast_scnn_citys.pth"
test_image_path = "Percep/b_CR_doort_frame00582.jpg"   # change this with live camera image

# --- Load device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
print("Loading model...")
model = get_fast_scnn(dataset='citys', aux=False)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("Model loaded.")

# --- Transform (same as training normalization) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

# --- Read image ---
orig = cv2.imread(test_image_path)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
img = cv2.resize(orig, (480, 480))  # resize to training crop size
tensor_img = transform(img).unsqueeze(0).to(device)

# --- Inference ---
with torch.no_grad():
    output = model(tensor_img)[0]
pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

print(pred)
# --- Define color map ---
colors = np.array([
    [0, 0, 0],          # undetected
    [0, 255, 0],        # road
    [255, 255, 0],      # lane
    [0, 0, 255],        # ducks
    [255, 0, 0],        # background
    [255, 0, 255],      # duckiebots
], dtype=np.uint8)

# --- Convert prediction to color mask ---
color_mask = colors[pred]
color_mask = cv2.resize(color_mask, (orig.shape[1], orig.shape[0]))

seg_rgb = color_mask.copy()

h, w = pred.shape
slice_y = int(h * 0.6)  # 40% from bottom = 60% from top
# band = pred[slice_y, :]  # single row at that height

# # # Get x indices of drivable pixels (road or lane)
# drivable_xs = np.where(np.logical_or(band == 1, band == 2))[0]

band = seg_rgb[slice_y, :, :]  # single row at that height (H=1, W, 3)

# --- Get drivable pixels (green [0,255,0] and yellow [255,255,0]) ---
road_mask = np.all(band == [0, 255, 0], axis=1)
lane_mask = np.all(band == [255, 255, 0], axis=1)
drivable_mask = np.logical_or(road_mask, lane_mask)

# Get x indices of drivable pixels
drivable_xs = np.where(drivable_mask)[0]
road_centroids = []
num_points = 8  # desired number of centroids

if len(drivable_xs) > 0:
    x_min = drivable_xs.min()
    x_max = drivable_xs.max()
    xs = np.linspace(x_min, x_max, num_points, dtype=int)
    for cx in xs:
        road_centroids.append((cx, slice_y))
        # cv2.circle(color_mask, (cx, slice_y), 4, (255, 0, 0), -1)

# --- Obstacle masks ---
obstacle_mask = cv2.inRange(seg_rgb, (0, 0, 255), (0, 0, 255))
mag_mask = cv2.inRange(seg_rgb, (255, 0, 255), (255, 0, 255))
obstacle_mask = cv2.bitwise_or(obstacle_mask, mag_mask)

# --- Parameters ---
lookahead_pixels = int(h * 0.18)
corridor_half_width = max(30, int(w * 0.06))
lane_preference = "left"  # or "right" (set manually or dynamically)

# --- Fallback if no road centroids ---
if len(road_centroids) == 0:
    safest_point = (int(w // 2), int(h * 0.6))
else:
    scores = []
    areas = []

    if len(road_centroids) > 2:
        del road_centroids[0]
        del road_centroids[-1]

    sorted_c = sorted(road_centroids, key=lambda c: c[0])
    num = len(sorted_c)
    if lane_preference == "left":
        pref_x = sorted_c[num // 4][0]
    elif lane_preference == "right":
        pref_x = sorted_c[3 * num // 4][0]
    else:
        pref_x = sorted_c[num // 2][0]

    for (cx, cy) in road_centroids:
        y_forward = max(0, cy - lookahead_pixels)
        x_min = max(0, cx - corridor_half_width)
        x_max = min(w - 1, cx + corridor_half_width)

        if x_max <= x_min:
            x_min = max(0, cx - 5)
            x_max = min(w - 1, cx + 5)

        rect = obstacle_mask[y_forward:cy + 1, x_min:x_max + 1]
        obstacle_area = int(cv2.countNonZero(rect))

        pref_distance = abs(cx - pref_x)
        pref_penalty = pref_distance / (w / 4.0)


        score = obstacle_area + (pref_penalty * 100)
        scores.append(score)
        areas.append(obstacle_area)

    if len(scores) > 0:
        best_idx = int(np.argmin(scores))
        safest_point = road_centroids[best_idx]
    else:
        safest_point = road_centroids[len(road_centroids) // 2]

sx, sy = safest_point
cv2.circle(color_mask, (sx, sy), 15, (0, 255, 255), -1)  # chosen waypoint

# for i, (cx, cy) in enumerate(road_centroids):
#     y_forward = max(0, cy - lookahead_pixels)
#     x_min = max(0, cx - corridor_half_width)
#     x_max = min(w - 1, cx + corridor_half_width)
#     cv2.rectangle(color_mask, (x_min, y_forward), (x_max, cy), (200, 200, 200), 1)
#     cv2.circle(color_mask, (cx, cy), 3, (255, 0, 0), -1)
#     if i < len(areas):
#         txt = f"{areas[i]}"
#         cv2.putText(color_mask, txt, (cx - 10, cy - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# print("Scores:", [round(s, 2) for s in scores])
print(f"Chosen waypoint ({lane_preference} pref):", safest_point)

# --- Display results ---
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.imshow(orig)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(color_mask)
plt.title("Segmented + Centroids + Safest Point")
plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(pred, cmap='gray')
# plt.title("Prediction Map (Class IDs)")
# plt.axis("off")

plt.show()
