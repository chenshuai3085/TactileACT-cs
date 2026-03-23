import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.nn import functional as F
import h5py
from tqdm import tqdm
from typing import List, Tuple, Optional
import os
from torchvision.transforms import Normalize

# 导入预训练脚本中的核心类/函数
from clip_pretraining_bounce import (
    modified_resnet18,
    ClipProjectionHead,
    BounceClipDataset,
    compute_proprio_stats,
    BounceSchema  # 补充导入BounceSchema
)

"""
适配BounceClipDataset的t-SNE可视化脚本
核心修改：对齐预训练的数据集格式、模型命名、相机配置
"""

def plot_tsne(all_image_vectors:np.ndarray, tac_vectors:np.ndarray, timestamps):
    """
    可视化图像和触觉（tac）的latent向量
    all_image_vectors: (n_cam, episode_len, clip_dim)
    tac_vectors: (episode_len, clip_dim)
    timestamps: 时间步数组
    """
    n_cam = all_image_vectors.shape[0]
    episode_len = len(timestamps)
    tsne = TSNE(n_components=2, random_state=10) 

    # 拼接所有向量用于t-SNE
    all_latent_vectors = np.concatenate(
        [all_image_vectors[i] for i in range(n_cam)] + [tac_vectors], 
        axis=0
    )

    # t-SNE降维
    embedded = tsne.fit_transform(all_latent_vectors)

    # 拆分图像/触觉的embedding
    image_embedings = [embedded[i*episode_len:(i+1)*episode_len] for i in range(n_cam)]
    tac_embedded = embedded[n_cam*episode_len:]

    # 绘图（颜色表示时间步，不同相机用不同标记）
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '*', 'D', 'P', 'H']
    for i, image_embedded in enumerate(image_embedings):
        plt.scatter(
            image_embedded[:, 0], image_embedded[:, 1], 
            c=timestamps/len(timestamps), cmap='viridis', 
            marker=markers[i], s=50, alpha=0.25, 
            label=f'Image Vectors ({camera_names[i]})'
        )
    plt.scatter(
        tac_embedded[:, 0], tac_embedded[:, 1], 
        c=timestamps/len(timestamps), cmap='viridis', 
        marker='x', s=50, alpha=0.25, 
        label='Tactile Vectors'
    )

    plt.title('t-SNE Visualization of Latent Vectors')
    plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Timestamp')
    plt.grid(True)
    plt.tight_layout() 


def plot_run_similarity(all_vectors, name):
    """绘制向量的余弦相似度矩阵"""
    episode_len = all_vectors.shape[0]
    similarity = np.zeros((episode_len, episode_len))
    for i in range(episode_len):
        for j in range(episode_len):
            # 余弦相似度计算（避免除零错误）
            norm_i = np.linalg.norm(all_vectors[i])
            norm_j = np.linalg.norm(all_vectors[j])
            if norm_i == 0 or norm_j == 0:
                similarity[i, j] = 0.0
            else:
                normalized_i = all_vectors[i] / norm_i
                normalized_j = all_vectors[j] / norm_j
                similarity[i, j] = np.dot(normalized_i, normalized_j)

    plt.figure()
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar()
    plt.title(f'Similarity Matrix for {name}')
    plt.xlabel('Timestep')
    plt.ylabel('Timestep')
    plt.tight_layout()


# -------------------------- 核心配置（已填你的路径） --------------------------
if __name__ == "__main__":
    # ====================== 1. 你的本地配置（已填） ======================
    SAVE_DIR = "/home/chenshuai/data/dataset/bounce/clip_models/loop2"
    DATASET_DIR = "/home/chenshuai/data/dataset/bounce"
    START_EPISODE = 90          
    NUM_EPISODES = 89           
    CAMERA_NAMES = "global,wrist"
    TAC_SIDE = "left"           
    TAC_KEY = "img"             
    PROPRIO_KEY = "proprio_eef" 
    CLIP_DIM = 512              
    FEATURES_PER_GROUP = 16     
    EPISODE_IDXS = [0, 3]
    EPOCH = 1499  # 手动指定epoch编号，设为None则自动选最新的

    # ====================== 2. 固定配置 ======================
    encoder_pretrained = True
    projection_head_pretrained = True
    use_projection_head = True
    use_act = False  

    # 解析相机名称
    camera_names = tuple([s.strip() for s in CAMERA_NAMES.split(",") if s.strip()])
    n_cameras = len(camera_names)

    # 设备
    device = torch.device("cpu")  # t-SNE可视化只做推理，用CPU即可

    # ====================== 3. 加载预训练模型 ======================
    vision_encoder = modified_resnet18(features_per_group=FEATURES_PER_GROUP).to(device)
    vision_projection = ClipProjectionHead(out_dim=CLIP_DIM).to(device)

    tac_encoder = modified_resnet18(features_per_group=FEATURES_PER_GROUP).to(device)
    tac_projection = ClipProjectionHead(
        out_dim=CLIP_DIM, 
        conditioning_dim=3  
    ).to(device)

    # 加载预训练权重
    if EPOCH is not None:
        target_epoch = EPOCH
    else:
        try:
            target_epoch = max(
                [int(f.split("_")[1]) for f in os.listdir(SAVE_DIR) if "vision_encoder.pth" in f]
            )
        except ValueError:
            print("未找到预训练权重文件，请检查SAVE_DIR路径！")
            exit(1)

    # 确保该epoch下4个权重文件都存在
    weight_files = {
        "vision_encoder": os.path.join(SAVE_DIR, f"epoch_{target_epoch}_vision_encoder.pth"),
        "vision_projection": os.path.join(SAVE_DIR, f"epoch_{target_epoch}_vision_projection.pth"),
        "tac_encoder": os.path.join(SAVE_DIR, f"epoch_{target_epoch}_tac_encoder.pth"),
        "tac_projection": os.path.join(SAVE_DIR, f"epoch_{target_epoch}_tac_projection.pth"),
    }
    for name, path in weight_files.items():
        if not os.path.exists(path):
            print(f"错误：epoch {target_epoch} 缺少权重文件 {name}: {path}")
            exit(1)
    print(f"加载 epoch {target_epoch} 的权重")

    vision_encoder.load_state_dict(
        torch.load(weight_files["vision_encoder"], map_location=device)
    )
    vision_projection.load_state_dict(
        torch.load(weight_files["vision_projection"], map_location=device)
    )
    tac_encoder.load_state_dict(
        torch.load(weight_files["tac_encoder"], map_location=device)
    )
    tac_projection.load_state_dict(
        torch.load(weight_files["tac_projection"], map_location=device)
    )

    # 模型设为评估模式
    vision_encoder.eval()
    vision_projection.eval()
    tac_encoder.eval()
    tac_projection.eval()

    # ====================== 4. 加载数据集 ======================
    episode_ids = list(range(START_EPISODE, START_EPISODE + NUM_EPISODES))
    proprio_mean, proprio_std = compute_proprio_stats(DATASET_DIR, episode_ids, PROPRIO_KEY)
    
    # 修复schema构建（直接用BounceSchema类）
    schema = BounceSchema(
        camera_names=camera_names,
        tac_side=TAC_SIDE,
        tac_key=TAC_KEY,
        proprio_key=PROPRIO_KEY
    )

    dataset = BounceClipDataset(
        episode_ids=episode_ids,
        dataset_dir=DATASET_DIR,
        schema=schema,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        min_distance=1,  
        n_images=1       
    )

    # ====================== 5. 图像归一化 ======================
    rgb_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # ====================== 6. 提取向量并可视化 ======================
    for idx in EPISODE_IDXS:
        episode_id = episode_ids[idx]
        episode_path = os.path.join(DATASET_DIR, f"episode_{episode_id}.hdf5")
        
        # 检查文件是否存在
        if not os.path.exists(episode_path):
            print(f"警告：Episode文件 {episode_path} 不存在，跳过！")
            continue
        
        # 读取episode长度（修复混入的注释文字）
        with h5py.File(episode_path, "r") as f:
            episode_len = f["observations"]["images"][camera_names[0]].shape[0]

        # 限制episode长度（避免内存溢出）
        max_len = 200  # 可根据显存调整
        episode_len = min(episode_len, max_len)

        # 初始化向量存储
        all_image_vectors = np.empty((n_cameras, episode_len, CLIP_DIM))
        tac_vectors = np.empty((episode_len, CLIP_DIM))
        timestamps = []

        # 逐时间步提取向量
        with torch.no_grad():
            for t in tqdm(range(episode_len), desc=f"Processing Episode {episode_id}"):
                timestamps.append(t)
                # 读取图像数据
                for cam_idx, cam_name in enumerate(camera_names):
                    with h5py.File(episode_path, "r") as f:
                        img = f["observations"]["images"][cam_name][t]
                    # 预处理
                    img = torch.tensor(img, dtype=torch.float32) / 255.0
                    img = torch.einsum("hwc->chw", img)  # 简化einsum写法
                    img = rgb_normalize(img).to(device)
                    # 提取特征
                    img_feat = vision_encoder(img.unsqueeze(0))
                    img_vec = vision_projection(img_feat)
                    all_image_vectors[cam_idx, t, :] = img_vec.detach().cpu().numpy().squeeze()

                # 读取触觉数据
                with h5py.File(episode_path, "r") as f:
                    tac_data = f["observations"]["tac"][TAC_SIDE][TAC_KEY][t]
                    proprio = f["observations"][PROPRIO_KEY][t].astype(np.float32)
                # 预处理触觉数据
                if TAC_KEY == "img":
                    tac = torch.tensor(tac_data, dtype=torch.float32) / 255.0
                    tac = torch.einsum("hwc->chw", tac)
                    tac = rgb_normalize(tac).to(device)
                elif TAC_KEY == "depth":
                    tac = torch.tensor(tac_data, dtype=torch.float32).unsqueeze(0).to(device)
                # 预处理proprio
                proprio = (proprio - proprio_mean) / proprio_std
                proprio = torch.tensor(proprio[:3], dtype=torch.float32).to(device)
                # 提取触觉特征
                tac_feat = tac_encoder(tac.unsqueeze(0))
                tac_vec = tac_projection(tac_feat, proprio.unsqueeze(0))
                tac_vectors[t, :] = tac_vec.detach().cpu().numpy().squeeze()

        timestamps = np.array(timestamps)

        # 可视化并保存（文件名带epoch方便对比不同版本）
        plot_run_similarity(tac_vectors, f'Tactile (Episode {episode_id}, Epoch {target_epoch})')
        plt.savefig(os.path.join(SAVE_DIR, f"ep{target_epoch}_episode_{episode_id}_sim_tactile.png"), dpi=300, bbox_inches='tight')

        for i, cam_name in enumerate(camera_names):
            plot_run_similarity(all_image_vectors[i], f'Camera {cam_name} (Episode {episode_id}, Epoch {target_epoch})')
            plt.savefig(os.path.join(SAVE_DIR, f"ep{target_epoch}_episode_{episode_id}_sim_{cam_name}.png"), dpi=300, bbox_inches='tight')

        plot_tsne(all_image_vectors, tac_vectors, timestamps)
        plt.title(f't-SNE for Episode {episode_id} (Epoch {target_epoch}, Clip Dim: {CLIP_DIM})')
        plt.savefig(os.path.join(SAVE_DIR, f"ep{target_epoch}_episode_{episode_id}_tsne.png"), dpi=300, bbox_inches='tight')

        plt.show()

    plt.show()
