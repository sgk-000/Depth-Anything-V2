import cv2
import torch
import numpy as np
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

video_root_path = '/home/anasys/dist_videos/'
video_name = 'motion_event_id_665983_motion_id_257639275_1688170025988_1688170050009.mp4'
video_path = video_root_path + video_name
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f'Error: Cannot open video {video_path}')
    exit()

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
fps = 13
size = (720, 1280)
video_writer = cv2.VideoWriter(f"/home/anasys/results/{video_name}", fourcc, fps, size)

cmap = matplotlib.colormaps.get_cmap('Spectral_r')


print("video loop start")
while True:
    ret, raw_img = cap.read()
    if not ret:
        break
    H, W, _ = raw_img.shape
    front_img = raw_img[:H//2, :, :]
    print(f"{front_img.shape=}")
    depth = model.infer_image(front_img) # HxW raw depth map in numpy
    print("inference end")

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    split_region = np.ones((front_img.shape[0], 50, 3), dtype=np.uint8) * 255

    print(f"{front_img.shape=}, {split_region.shape=}, {depth.shape=}")
    combined_result = cv2.hconcat([front_img, split_region, depth])
    
    video_writer.write(combined_result)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
video_writer.release()