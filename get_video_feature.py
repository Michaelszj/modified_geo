import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed
import argparse
from tqdm import tqdm
        
def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        sd_path = f"{feature_base}_sd.pt"
        dino_path = f"{feature_base}_dino.pt"

    # extract stable diffusion features
    if img_path is not None and os.path.exists(sd_path):
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to('cuda')
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

    # extract dinov2 features
    if img_path is not None and os.path.exists(dino_path):
        features_dino = torch.load(dino_path)
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            features_dino
        ], dim=1)
    
    desc = aggre_net(desc_gathered) # 1, 768, 60, 60
    # normalize the descriptors
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='./kinetics_dino')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    set_seed(42)
    num_patches = 60
    sd_model = sd_aug = extractor_vit = None
    aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
    aggre_net.load_pretrained_weights(torch.load('results_spair/best_856.PTH'))
    # This block may take ~4 minutes to run
    # If you only want to visualize the features of dataset images that you have already pre-extracted, you can skip this block

    sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')

    
    # folders = sorted([int(name) for name in os.listdir(args.video_path)])
    # current_path = os.path.join(args.video_path, str(folders[args.idx]))
    current_path = args.video_path
    images_path = os.path.join(current_path,'video')
    print('processing video:', images_path)
    output_path = os.path.join(current_path,'dino_embeddings')
    os.makedirs(output_path, exist_ok=True)
    image_files = sorted([file for file in os.listdir(images_path) if file[-3:] == 'jpg'])
    
    
    features = []
    for i in tqdm(range(len(image_files))):
        # import pdb; pdb.set_trace()
        img_path = os.path.join(images_path, image_files[i])
        img = Image.open(img_path).convert('RGB').resize((480,480),Image.LANCZOS)
        feature = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=img)
        features.append(feature.detach().cpu())
        
    features = torch.cat(features, dim=0)
    torch.save(features, os.path.join(output_path, 'geo_embed_video.pt'))
    