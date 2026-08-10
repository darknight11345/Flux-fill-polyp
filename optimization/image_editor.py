import os
import blobfile as bf
import pickle
from pathlib import Path
#from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

#from utils_visualize.metrics_accumulator import MetricsAccumulator
#from utils_visualize.video import save_video

from numpy import random
#from optimization.augmentations import ImageAugmentations

from PIL import Image
import cv2
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from torchvision import models
#from optimization.losses import range_loss, d_clip_loss, get_features, zecon_loss_direct
# import lpips
import numpy as np
#from src.vqc_core import *
#from model_vit.loss_vit import Loss_vit
#from guided_diffusion.guided_diffusion import dist_util, logger
'''from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils_visualize.visualization import show_tensor_image, show_editied_masked_image
'''
from pathlib import Path
#from id_loss import IDLoss
import datetime
from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer


import transformers
from accelerate import Accelerator
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast


import logging 
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxFillPipeline,
    FluxTransformer2DModel,
)
from diffusers.utils import load_image
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
import pickle
import blobfile as bf
import matplotlib.pyplot as plt
from torchvision import transforms

if is_wandb_available():
    import wandb

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",)

logger = logging.getLogger(__name__)

class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        #self.model_path = self.args.model_path
        self.model_name = "black-forest-labs/FLUX.1-Fill-dev"
        #self.output_path=args.output_path
        #os.makedirs(self.args.output_path, exist_ok=True)

        
        #logger.info("Loading model...")

        logger.info(f"Loading model: {self.model_name}")
        #logger.info(f"Using device: {self.device}")

        if self.args.seed is not None:
            logger.info(f"Setting seed:{self.args.seed}")
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        self.generator=torch.Generator(device=self.device).manual_seed(self.args.seed)

        self.pipe = FluxFillPipeline.from_pretrained(self.model_name,torch_dtype=torch.bfloat16)          
           
        #self.pipe.load_lora_weights(os.path.dirname(self.model_path),weight_name=os.path.basename(self.model_path),)
        

        #self.pipe.enable_model_cpu_offload()       

        
    def load_cluster_lora(self, lora_path):
 
  
        logger.info(hasattr(self.pipe, "unload_lora_weights"))
        logger.info(f" the lora_path is : {lora_path}")

        logger.info(f" the self.args.model_path is : {self.args.model_path}")
        base_dir = datetime.datetime.now().strftime("diffgen-%Y-%m-%d-%H-%M")

        self.model_path = lora_path
        
        logger.info(f" the self.model_path is : {self.model_path}")
     
        if self.args.cluster_path:
            cluster_index =  int(self.model_path.split("/")[-2].split("-")[1])
            base_dir = f"cluster{cluster_index}-{base_dir}"
            logger.info(f"base dir : {base_dir}")


        self.ranked_results_path = Path(self.args.output_path, base_dir)
        self.root_dir = Path(__file__).parent.parent.as_posix()
        
        logger.info(f"Root dir: {self.root_dir}")

        os.makedirs(self.ranked_results_path, exist_ok=True)
        self.pipe.unload_lora_weights()

        self.pipe.load_lora_weights(os.path.dirname(lora_path),weight_name=os.path.basename(lora_path))
        
        logger.info(f"loading lora weights from {lora_path}")
        self.pipe.to(self.device)
        
        logger.info("lora adapters........")
        logger.info(self.pipe.get_active_adapters())
        
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        #with open(f"{self.root_dir}/model_vit/config.yaml", "r") as ff:
            #config = yaml.safe_load(ff)

        #cfg = config
        """
        lambda_ssim = l_ssim
        lambda_contra_ssim = l_cont
        lambda_dir_cls = l_sem
        lambda_trg = l_sty

        Want to replace lambda_ssim, lambda_contra_ssim with zecon loss
        """
        
        ''' T commented
        self.VIT_LOSS = Loss_vit(cfg, lambda_ssim=self.args.lambda_ssim,lambda_dir_cls=self.args.lambda_dir_cls,lambda_contra_ssim=self.args.lambda_contra_ssim,lambda_trg=args.lambda_trg).eval()
        
        self.cm = ColorMatcher()

        # self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

        if self.args.lambda_vgg > 0:
            self.vgg = models.vgg19(pretrained=True).features
            self.vgg.to(self.device)
            self.vgg.eval().requires_grad_(False)
        
        self.vgg_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        '''

    def noisy_aug(self,t,clean_latent):

        noise=torch.randn_like(clean_latent)

        return self.scheduler.add_noise(
            clean_latent,
            noise,
            t
        )
    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep
    
    def zecon_loss(self,x_in,y_in,t,prompt_embeds,pooled_prompt_embeds,text_ids,latent_image_ids, guidance):
                    
        loss = zecon_loss_direct(self,x_in,y_in,t,prompt_embeds,pooled_prompt_embeds,text_ids,latent_image_ids, guidance)
        return loss.mean()
    
    def vgg_loss(self,x_in, y_in):
        content_features = get_features(self.vgg_normalize(x_in), self.vgg)
        target_features = get_features(self.vgg_normalize(y_in), self.vgg)
        loss = 0

        loss += torch.mean((target_features['conv1_1'] - content_features['conv1_1']) ** 2)
        loss += torch.mean((target_features['conv2_1'] - content_features['conv2_1']) ** 2)
        # loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        # loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        return loss.mean()
    
    def cnt_mse_loss(self, x_in, y_in):
        loss = mse_loss(x_in, y_in)
        return loss.mean()

    def _list_image_files_recursively(self, data_dir):
        results = []
        for entry in sorted(bf.listdir(data_dir)):
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                results.append(full_path)
            elif bf.isdir(full_path):
                results.extend(self._list_image_files_recursively(full_path))
        return results
    
    def _load_cluster(self, cluster_path):
        # Model file name is cluster_i_modelN.pt
        cluster_index = int(self.model_path.split("/")[-2].split("-")[1])
        logger.info(f"Loading cluster {cluster_index} from {cluster_path}")
        with open(cluster_path, "rb") as f:
            cluster = pickle.load(f)
        return cluster[cluster_index]
    
    def _get_target_image_and_mask(self, img_paths, it=None, exclude_path=None):
        if exclude_path is not None:
            img_paths = [p for p in img_paths if p != exclude_path]
        len_img_paths = len(img_paths)
        logger.info(f"0th image 1st: {img_paths[0]}")        
        rand_file = img_paths[it % len_img_paths] if it is not None else random.choice(img_paths)
        logger.info(f"rand_file 1st: {rand_file}")
        # rand_file of the form .../guided_diffusion/data
        # Replace . with self.root_dir
        #rand_file = bf.join(self.root_dir, rand_file[2:]) if self.args.cluster_path else rand_file
        logger.info(f"rand_file 2nd: {rand_file}")
        mask_path = bf.join(bf.dirname(bf.dirname(rand_file)), 'masks', bf.basename(rand_file))
        logger.info(f"mask_path: {mask_path}")
        #self.target_image_pil = load_image(rand_file)
        #self.target_mask_pil = load_image(mask_path)
        
        self.target_image_pil = Image.open(rand_file).convert("RGB")
        self.target_mask_pil = Image.open(mask_path).convert("L")
        size=1024
        #self.target_image_pil = self.target_image_pil.resize(size, Image.NEAREST)
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.target_image_pil=train_resize(self.target_image_pil)

        print(f"the size of self.target_image_pil : {self.target_image_pil.size}")
        self.target_mask_pil = self.target_mask_pil.resize(self.target_image_pil.size, Image.NEAREST)
        
        # Dilate mask
        if self.args.inpainting:
        
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask_arr = np.array(self.target_mask_pil)
            mask_arr = np.where(mask_arr > 0, 1, 0).astype(np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
            self.target_mask_pil = Image.fromarray(mask_arr.astype(np.float64))        
        
            '''
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask_arr = np.array(self.target_mask_pil)
            mask_arr = mask_arr.astype(np.float32) / 255.0
            mask_arr[mask_arr < 0.5] = 0
            mask_arr[mask_arr >= 0.5] = 1
            #mask_arr = np.where(mask_arr > 0, 1, 0).astype(np.uint8)
            #mask_arr = np.where(mask_arr >= 128, 255, 0).astype(np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
            #self.target_mask_pil = Image.fromarray(mask_arr.astype(np.float64))
            self.target_mask_pil = Image.fromarray(mask_arr)
            '''
        #self.target_image = (TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1))
        #self.target_mask = (TF.to_tensor(self.target_mask_pil).to(self.device).unsqueeze(0).mul(2).sub(1))
        #self.target_image = self.target_image.repeat(self.args.batch_size, 1, 1, 1)
        #self.target_mask = self.target_mask.repeat(self.args.batch_size, 1, 1, 1)
        
        logger.info(f"shape of self.target_image_pil, self.target_mask_pil is {self.target_image_pil}, {self.target_mask_pil}")
        
        self.target_image_pil.save("debug_image.png")
        self.target_mask_pil.save("debug_mask.png")
        
        return self.target_image_pil, self.target_mask_pil
        
    def _get_init_image_and_mask(self, img_paths, it=None, exclude_path=None):
        if exclude_path is not None:
            img_paths = [p for p in img_paths if p != exclude_path]
        len_img_paths = len(img_paths)
        rand_file = img_paths[it % len_img_paths] if it is not None else random.choice(img_paths)
        rand_file = bf.join(self.root_dir, rand_file[2:]) if self.args.cluster_path else rand_file
        mask_path = bf.join(bf.dirname(bf.dirname(rand_file)), 'masks', bf.basename(rand_file))

        self.init_image_pil = Image.open(rand_file).convert("RGB")
        self.init_mask_pil = Image.open(mask_path).convert("L")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)
        self.init_mask_pil = self.init_mask_pil.resize(self.image_size, Image.LANCZOS)

        # Dilate mask
        if self.args.inpainting:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask_arr = np.array(self.init_mask_pil)
            mask_arr = np.where(mask_arr > 0, 1, 0).astype(np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
            self.target_mask_pil = Image.fromarray(mask_arr.astype(np.float64))

        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        # Tile to batch size
        self.init_image = self.init_image.repeat(self.args.batch_size, 1, 1, 1)
        self.init_mask = (
            TF.to_tensor(self.init_mask_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        self.init_mask = self.init_mask.repeat(self.args.batch_size, 1, 1, 1)

        return rand_file, mask_path

    def _save(self, all_images):#(self, all_images, styled_images):
        # Apply np.array to all values in dict
        img_dict = {k: np.array(v) for k, v in all_images.items()}
        arr = next(iter(img_dict.values()))
        img_dict_shape = list(arr.shape)
        img_dict_shape[0] = img_dict_shape[0] * len(img_dict)
        # arr = arr[: args.num_samples]
        logger.info(f"Shape of img_dict: {img_dict_shape}")
        '''
        styled_dict = {k: np.array(v) for k, v in styled_images.items()}
        if styled_dict == {}: # If no styling
            styled_dict = img_dict
        styled_arr = next(iter(styled_dict.values()))
        styled_dict_shape = list(styled_arr.shape)
        styled_dict_shape[0] = styled_dict_shape[0] * len(styled_dict)
        logger.info(f"Shape of styled_dict: {styled_dict_shape}")
        # arr = np.array(all_images)
        # styled_arr = np.array(styled_images)
        '''
        shape_str = "x".join([str(x) for x in img_dict_shape])
        out_path = os.path.join(self.ranked_results_path, f"samples_{shape_str}.pkl")
        #styled_shape_str = "x".join([str(x) for x in styled_dict_shape])
        #styled_out_path = os.path.join(self.ranked_results_path, f"styled_samples_{styled_shape_str}.pkl")

        logger.info(f"saving to {out_path}")
        # np.savez(out_path, arr)
        # np.savez(styled_out_path, styled_arr)
        # Save dicts as pkl
        with open(out_path, 'wb') as f:
            pickle.dump(img_dict, f)
        #with open(styled_out_path, 'wb') as f:
            #pickle.dump(styled_dict, f)
    def flux_gen_sample(self,height,width,target_image,target_mask,):
        
        logger.info(f"image size: {target_image.size}")
        logger.info(f"mask size: {target_mask.size}")
        mask_arr = np.array(target_mask)
        logger.info(f"unique mask values: {np.unique(mask_arr)}")
        logger.info(f"masked ratio: {(mask_arr > 0).mean()}")
        plt.figure(figsize=(5,5))
        plt.imshow(target_mask, cmap="gray")
        plt.axis("off")
        plt.show()
        plt.imshow(target_image)
        plt.show()        
        result = self.pipe(
                image=target_image,
                mask_image=target_mask,
                prompt="",
                height=height,
                width=width,
                guidance_scale=30,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=self.generator
        ).images[0]

        logger.info(f"Result size: {result.size}")           
        result.save(f"flux_inpaint_{height}_{width}_lora.png")

        return result

    def sample_image(self):
        shapes = (self.args.batch_size, self.vae.config.latent_channels, self.args.model_output_size, self.args.model_output_size)
        logger.info(f"the shape is: {shapes}")
        self.init_image = torch.zeros(shapes).to(self.device)
        self.image_size = (self.args.model_output_size, self.args.model_output_size)
        it = 0
        if self.args.init_image is not None:
            self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
            self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.init_image = (
                TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1))
        
        self.target_image = None
        if self.args.target_image is not None:
            self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
            self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.target_image = (
                TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
            )
        elif self.args.find_target_image and (self.args.cluster_path or self.args.data_dir): # find target image
            logger.info("finding the target image..........")
            all_files = self._load_cluster(self.args.cluster_path) if self.args.cluster_path else self._list_image_files_recursively(self.args.data_dir)
            
            style_img_path = None

            if False and self.args.init_image is not None:  #### T: this will never run, then why?
                best_loss = self.VIT_LOSS.calculate_global_ssim_loss(self.init_image, self.target_image) + self.VIT_LOSS.calculate_contra_ssim_loss(self.init_image, self.target_image)
                best_tgt_img_pil = self.target_image_pil
                
                for path in all_files:
                    target_image_pil = Image.open(path).convert("RGB")
                    target_image_pil = target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
                    target_image = (
                        TF.to_tensor(target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
                    )
                    ssim_loss = self.VIT_LOSS.calculate_global_ssim_loss(self.init_image, target_image)
                    cont_ssim_loss = self.VIT_LOSS.calculate_contra_ssim_loss(self.init_image, target_image)
                    # Save target_image with lowest ssim + contrastive ssim
                    if ssim_loss + cont_ssim_loss < best_loss:
                        best_loss = ssim_loss + cont_ssim_loss
                        best_tgt_img_pil = target_image_pil
                self.target_image_pil = best_tgt_img_pil
                self.target_image = (
                    TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
                )
       
        self.prev = self.init_image.detach()
        self.flag_resample=False
        #total_steps = 1000 - self.args.skip_timesteps - 1 this is used for codn_fn

        def cond_fn(x, t, y=None):
            # if self.args.prompt == "":
            #     return torch.zeros_like(x)
            self.flag_resample=False
            with torch.enable_grad():
                frac_cont=1.0
   
                pred_x0 = x.detach().requires_grad_()
                t = self.unscale_timestep(t)
               
                latents = pred_x0 / self.vae.config.scaling_factor

                if self.vae.config.shift_factor is not None:
                    latents = latents + self.vae.config.shift_factor

                x_in = self.vae.decode(latents).sample
                
                loss = torch.tensor(0) # default value

                if self.init_image.eq(0).all(): # reconstruction
                    if self.target_image is not None:
                        loss = loss + mse_loss(
                            x_in[:, :3, ...],
                            self.target_image
                        ) * self.args.l2_trg_lambda
                else: # styling
                    if self.args.use_noise_aug_all:
                        #x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                        x_in = self.noisy_aug(
                            t[0],
                            pred_x0
                        )

                        x_in = self.vae.decode(
                            x_in / self.vae.config.scaling_factor
                        ).sample
                    else:
                        x_in = x_in
                        #x_in = out["pred_xstart"]
                    # self.init_image = (B,4,H,W)
                    x_in3 = x_in[:, :3, ...]
                    # init_image_batch = torch.tile(self.init_image[:3, ...], dims=(self.args.batch_size, 1, 1, 1))
                    # zecon_init_image_batch = torch.tile(self.init_image, dims=(self.args.batch_size, 1, 1, 1))
                    # self.prev = torch.tile(self.prev[:3, ...], dims=(self.args.batch_size, 1, 1, 1))
                    init_image_batch = self.init_image[:, :3, ...]
                    zecon_init_image_batch = self.init_image
                    self.prev = self.prev[:, :3, ...]

                    if self.args.vit_lambda != 0:     
                        # self.init_image is x_src  
                        if t[0].item()>self.args.diff_iter : # directional cls
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in3, init_image_batch,self.prev,use_dir=True,frac_cont=frac_cont,target = self.target_image)
                        else:
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in3,init_image_batch,self.prev,use_dir=False,frac_cont=frac_cont,target = self.target_image)
                        loss = loss + vit_loss

                    if self.args.range_lambda != 0:
                        #r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                        r_loss = range_loss(pred_x0).sum() * self.args.range_lambda
                        loss = loss + r_loss
                        self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                    if self.target_image is not None:
                        loss = loss + mse_loss(x_in3, self.target_image) * self.args.l2_trg_lambda

                    self.prev = x_in3.detach().clone()

                    # ------------------  New Losses ------------------
                    #fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]             

                    if not self.args.use_noise_aug_all:
                        x_in = ((1-sigma)*pred_x0+sigma*x)
                    
                    current_sigma = sigma.flatten()[0].item()


                    if self.args.lambda_zecon != 0:
                        #y_t = self.diffusion.q_sample(zecon_init_image_batch,t)
                        #y_in = zecon_init_image_batch * fac + y_t * (1 - fac)
                        noise_ref = torch.randn_like(zecon_init_image_batch)
                        init_latent = vae.encode(self.init_image).latent_dist.sample()

                        init_latent = (init_latent - self.vae.config.shift_factor)*self.vae.config.scaling_factor
                        y_t = ((1-sigma)*init_latent+sigma*noise_ref) #noisy reference latent
                        
                        #zecon_loss = self.zecon_loss(x_in, y_in,t) * self.args.lambda_zecon
                        x_in=pred_x0
                        zecon_loss = self.zecon_loss(
                                        x_in,
                                        y_t,
                                        t,
                                        prompt_embeds,
                                        pooled_prompt_embeds,
                                        text_ids,
                                        latent_image_ids,
                                        guidance
                                    ) * self.args.lambda_zecon
                        loss = loss + zecon_loss
                        self.metrics_accumulator.update_metric("zecon_loss", zecon_loss.item())
                    
                    if self.args.lambda_vgg != 0 and current_sigma < 0.8:  #t[0].item() < 800:
                        '''
                        y_t = self.diffusion.q_sample(init_image_batch,t)
                        y_in = init_image_batch * fac + y_t * (1 - fac)
                        '''
                        noise_ref = torch.randn_like(init_latent)
                        y_t = ((1-sigma)*init_latent+sigma*noise_ref)
                        y_in = self.vae.decode(y_t / self.vae.config.scaling_factor).sample
                        
                        vgg_loss = self.vgg_loss(x_in3, y_in) * self.args.lambda_vgg
                        loss = loss + vgg_loss
                        self.metrics_accumulator.update_metric("vgg_loss", vgg_loss.item())
                    if self.args.lambda_mse != 0 and current_sigma < 0.7: #t[0].item() < 700:
                        #y_t = self.diffusion.q_sample(init_image_batch, t)
                        #y_in = init_image_batch * fac + y_t * (1 - fac)
                        noise_ref = torch.randn_like(init_latent)
                        y_t = ((1-sigma)*init_latent+sigma*noise_ref)
                        y_in = self.vae.decode(y_t / self.vae.config.scaling_factor).sample

                        cnt_mse_loss = self.cnt_mse_loss(x_in3, y_in) * self.args.lambda_mse
                        loss = loss + cnt_mse_loss
                        self.metrics_accumulator.update_metric("cnt_mse_loss", cnt_mse_loss.item())

                    # ------------------  New Losses End ------------------
                    
                    if self.args.use_range_restart:
                        if t[0].item() < total_steps:
                            if r_loss>0.01:
                                    self.flag_resample =True
            return (-torch.autograd.grad(loss, x)[0] if not loss.eq(0).all() else loss), self.flag_resample

        
        def cond_fn_old(x, t, y=None):
            # if self.args.prompt == "":
            #     return torch.zeros_like(x)
            self.flag_resample=False
            with torch.enable_grad():
                frac_cont=1.0
   
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                velocity = self.transformer(
                    hidden_states=x,
                    timestep=t/1000,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance
                )[0]


                sigma = get_sigmas(
                    t,
                    n_dim=x.ndim,
                    dtype=x.dtype
                )
                
                pred_x0 = x - sigma * velocity #generated latent 
                
                latents = pred_x0 / self.vae.config.scaling_factor

                if self.vae.config.shift_factor is not None:
                    latents = latents + self.vae.config.shift_factor

                x_in = self.vae.decode(latents).sample
                
                loss = torch.tensor(0) # default value

                if self.init_image.eq(0).all(): # reconstruction
                    '''                
                    noise = torch.randn_like(x)
                    nonzero_mask = (
                        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                    )  # no noise when t == 0
                    x_in = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                    '''
                    if self.target_image is not None:
                        loss = loss + mse_loss(
                            x_in[:, :3, ...],
                            self.target_image
                        ) * self.args.l2_trg_lambda
                else: # styling
                    if self.args.use_noise_aug_all:
                        #x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                        x_in = self.noisy_aug(
                            t[0],
                            pred_x0
                        )

                        x_in = self.vae.decode(
                            x_in / self.vae.config.scaling_factor
                        ).sample
                    else:
                        x_in = x_in
                        #x_in = out["pred_xstart"]
                    # self.init_image = (B,4,H,W)
                    x_in3 = x_in[:, :3, ...]
                    # init_image_batch = torch.tile(self.init_image[:3, ...], dims=(self.args.batch_size, 1, 1, 1))
                    # zecon_init_image_batch = torch.tile(self.init_image, dims=(self.args.batch_size, 1, 1, 1))
                    # self.prev = torch.tile(self.prev[:3, ...], dims=(self.args.batch_size, 1, 1, 1))
                    init_image_batch = self.init_image[:, :3, ...]
                    zecon_init_image_batch = self.init_image
                    self.prev = self.prev[:, :3, ...]

                    if self.args.vit_lambda != 0:     
                        # self.init_image is x_src  
                        if t[0].item()>self.args.diff_iter : # directional cls
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in3, init_image_batch,self.prev,use_dir=True,frac_cont=frac_cont,target = self.target_image)
                        else:
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in3,init_image_batch,self.prev,use_dir=False,frac_cont=frac_cont,target = self.target_image)
                        loss = loss + vit_loss

                    if self.args.range_lambda != 0:
                        #r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                        r_loss = range_loss(pred_x0).sum() * self.args.range_lambda
                        loss = loss + r_loss
                        self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                    if self.target_image is not None:
                        loss = loss + mse_loss(x_in3, self.target_image) * self.args.l2_trg_lambda

                    self.prev = x_in3.detach().clone()

                    # ------------------  New Losses ------------------
                    #fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]             

                    if not self.args.use_noise_aug_all:
                        x_in = ((1-sigma)*pred_x0+sigma*x)
                    
                    current_sigma = sigma.flatten()[0].item()


                    if self.args.lambda_zecon != 0:
                        #y_t = self.diffusion.q_sample(zecon_init_image_batch,t)
                        #y_in = zecon_init_image_batch * fac + y_t * (1 - fac)
                        noise_ref = torch.randn_like(zecon_init_image_batch)
                        init_latent = vae.encode(self.init_image).latent_dist.sample()

                        init_latent = (init_latent - self.vae.config.shift_factor)*self.vae.config.scaling_factor
                        y_t = ((1-sigma)*init_latent+sigma*noise_ref) #noisy reference latent
                        
                        #zecon_loss = self.zecon_loss(x_in, y_in,t) * self.args.lambda_zecon
                        x_in=pred_x0
                        zecon_loss = self.zecon_loss(
                                        x_in,
                                        y_t,
                                        t,
                                        prompt_embeds,
                                        pooled_prompt_embeds,
                                        text_ids,
                                        latent_image_ids,
                                        guidance
                                    ) * self.args.lambda_zecon
                        loss = loss + zecon_loss
                        self.metrics_accumulator.update_metric("zecon_loss", zecon_loss.item())
                    
                    if self.args.lambda_vgg != 0 and current_sigma < 0.8:  #t[0].item() < 800:
                        '''
                        y_t = self.diffusion.q_sample(init_image_batch,t)
                        y_in = init_image_batch * fac + y_t * (1 - fac)
                        '''
                        noise_ref = torch.randn_like(init_latent)
                        y_t = ((1-sigma)*init_latent+sigma*noise_ref)
                        y_in = self.vae.decode(y_t / self.vae.config.scaling_factor).sample
                        
                        vgg_loss = self.vgg_loss(x_in3, y_in) * self.args.lambda_vgg
                        loss = loss + vgg_loss
                        self.metrics_accumulator.update_metric("vgg_loss", vgg_loss.item())
                    if self.args.lambda_mse != 0 and current_sigma < 0.7: #t[0].item() < 700:
                        #y_t = self.diffusion.q_sample(init_image_batch, t)
                        #y_in = init_image_batch * fac + y_t * (1 - fac)
                        noise_ref = torch.randn_like(init_latent)
                        y_t = ((1-sigma)*init_latent+sigma*noise_ref)
                        y_in = self.vae.decode(y_t / self.vae.config.scaling_factor).sample

                        cnt_mse_loss = self.cnt_mse_loss(x_in3, y_in) * self.args.lambda_mse
                        loss = loss + cnt_mse_loss
                        self.metrics_accumulator.update_metric("cnt_mse_loss", cnt_mse_loss.item())

                    # ------------------  New Losses End ------------------
                    
                    if self.args.use_range_restart:
                        if t[0].item() < total_steps:
                            if r_loss>0.01:
                                    self.flag_resample =True
            return (-torch.autograd.grad(loss, x)[0] if not loss.eq(0).all() else loss), self.flag_resample        
        # [-1, 1] -> [0, 255]
        def preprocess(sample): # sample is image tensor
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            return sample
        
        def refine_mask(sample_batch):
            mask = sample_batch[:, 3, ...]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
            mask = mask.cpu().numpy()
            for i in range(mask.shape[0]):
                mask[i] = cv2.morphologyEx(mask[i], cv2.MORPH_OPEN, kernel)
            mask = torch.from_numpy(mask).to(sample_batch.device)
            sample_batch[:, 3, ...] = mask
            return sample_batch

        def prepare_mask_and_masked_image(image, mask):
            image = np.array(image.convert("RGB"))
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

            mask = np.array(mask.convert("L"))
            mask = mask.astype(np.float32) / 255.0
            mask = mask[None, None]
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)

            masked_image = image * (mask < 0.5)
            
            return mask, masked_image
    
        @torch.no_grad()
        def flux_sample_loop(
            self,
            shape,
            target_image,
            target_mask,
            guidance,):
            
            return_dict = True
            

            pil_image = Image.open(target_image)
            mask_image = Image.open(target_mask)
        
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask_image)
            sigma = self.scheduler.sigmas[0]

            init_latent = self.vae.encode(self.pil_image).latent_dist.sample()
            init_latent = (init_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            init_latent = init_latent.to(dtype=weight_dtype)
            
            latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                init_latent.shape[0],
                init_latent.shape[2] // 2,
                init_latent.shape[3] // 2,
                accerlator.device,
                init_latent.dtype,
            )
            logger.info(f"latent_image_ids shape {latent_image_ids.shape}")
            
            noise = torch.randn_like(init_latent)
            #sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            latents = ((1-sigma)*init_latent+sigma*noise)
            packed_latents = FluxFillPipeline._pack_latents(
                    latents,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
            )                
              
            masked_image_latents = vae.encode(masked_image.to(dtype=weight_dtype)).latent_dist.sample()
            masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                #logger.info("masked image latents", masked_image_latents.shape)
                
            masked_image_latents = FluxFillPipeline._pack_latents(
                    masked_image_latents,
                    batch_size=init_latent.shape[0],
                    num_channels_latents=init_latent.shape[1],
                    height=init_latent.shape[2],
                    width=init_latent.shape[3],
            )            

            # 5.resize mask to latents shape we we concatenate the mask to the latents
            mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
            mask = mask.view(
                    init_latent.shape[0], init_latent.shape[2], vae_scale_factor, init_latent.shape[3], vae_scale_factor
            )  # batch_size, height, 8, width, 8
            mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
            mask = mask.reshape(
                    init_latent.shape[0], vae_scale_factor * vae_scale_factor, init_latent.shape[2], init_latent.shape[3]
            )  # ba
                #print("mask ", mask.shape)
                #print("packed masked image latents", masked_image_latents.shape)
                #print("model input", init_latent)
                #print("model inputhshae", init_latent.shape)
            mask = FluxFillPipeline._pack_latents(
                    mask,
                    batch_size=init_latent.shape[0],
                    num_channels_latents=vae_scale_factor*vae_scale_factor,
                    height=init_latent.shape[2],
                    width=init_latent.shape[3],
            )
                #print("packed mask ", mask.shape)                
            masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
            logger.info(f" masked_image_latents {masked_image_latents.shape}")
            

            
            self.scheduler.set_timesteps(self.args.num_sampling_steps,device=self.device) # this is number of inference steps
                


            for t in self.scheduler.timesteps:

                timestep = t.expand(latents.shape[0]).to(latents.dtype)


                transformer_input = torch.cat((packed_latents, masked_image_latents), dim=2)    
                prompt_embeds = torch.zeros_like(prompt_embeds)
                pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
                velocity = self.transformer(
                    hidden_states=transformer_input,
                    timestep=t/1000,
                    guidance=guidance,
                    prompt="",
                    img_ids=latent_image_ids,
                )[0]
                latents = self.scheduler.step(
                    velocity,
                    t,
                    latents
                )[0]
                
            unpacked_latents = FluxFillPipeline._unpack_latents(
                    velocity,
                    height=height,
                    width=width,
                    vae_scale_factor=self.vae.config.scaling_factor,
            )
            unpacked_latents = (unpacked_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor   
            image = self.vae.decode(unpacked_latents)[0]
            
            logger.info(unpacked_latents.shape)
            logger.info(image.shape)
            if not return_dict:
                return (image,) 
            return FluxPipelineOutput(images=image)                

        def flux_style_loop(
            self,
            shape,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            init_image,
            guidance,):
            
            
            latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                init_latent.shape[0],
                init_latent.shape[2] // 2,
                init_latent.shape[3] // 2,
                accerlator.device,
                init_latent.dtype,
            )
            logger.info("latent_image_ids shape ", latent_image_ids.shape)
            
            noise = torch.randn_like(init_latent)
            sigma = self.scheduler.sigmas[0]
            #sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            latents = ((1-sigma)*init_latent+sigma*noise)
                       
            self.scheduler.set_timesteps(self.args.num_sampling_steps,device=self.device) # this is number of inference steps

            for t in self.scheduler.timesteps:

                timestep = t.expand(latents.shape[0])
                packed_latents = FluxFillPipeline._pack_latents(
                    latents,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )

                transformer_input = packed_latents  
                
                velocity = self.transformer(
                    hidden_states=transformer_input,
                    timestep=t/1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                )[0]
                
                velocity = FluxFillPipeline._unpack_latents(
                    velocity,
                    height=height,
                    width=width,
                    vae_scale_factor=8,
                )
                sigma = t / 1000
                
                pred_x0 = latents - sigma * velocity
                
                cond_fn(pred_x0,)
                
                latents = self.scheduler.step(
                    velocity,
                    t,
                    latents
                ).prev_sample
                
        
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor
            ).sample
            
            logger.info(latents.shape)
            logger.info(image.shape)
            return latents               

        

        
        all_images = {}
        #styled_images = {}
        total_style_steps = 1000 - self.args.style_skip_timesteps  #80
        save_image_interval = total_style_steps // 5
        num_samples = self.args.num_samples * len(all_files) if self.args.sample_per_image else self.args.num_samples
        logger.info(f"len(all_files) no of image in cluster index: {len(all_files)}")
        logger.info(f"num_samples {num_samples}")
        logger.info(f"Sampling {num_samples * self.args.batch_size} images")
        while it < 1: #num_samples
            # Sets target_image and target_mask
            target_image, target_mask = self._get_target_image_and_mask(all_files, it=it)
            it += 1
            logger.info(f"Style image {style_img_path}")

            if not self.args.style_aug:
                '''
                samples = self.flux_sample_loop(
                    shape=shape,
                    target_image=target_image, 
                    target_mask=target_mask,
                    guidance=None
                )
                '''
                samples = self.flux_gen_sample(
                    #shapes=shapes,
                    height=546,
                    width=626,
                    target_image=target_image, 
                    target_mask=target_mask,
                )
                
                if self.flag_resample:
                    continue
                samples = torch.stack([refine_mask(sample[i]) for i in samples]).squeeze(0)
                
                logger.info(f"sample.shape: {sample.shape}") # [batch_size, 4, 256, 256] for unet  , [batch_size, 16, 256, 256] for flux 
            # NOTE: If we are styling, we want to predict x_0, so return sample["pred_xstart"], otherwise, we sample, so return sample["sample"]
            else:
                src_image_path, src_mask = self._get_init_image_and_mask(all_files, style_img_path)
                logger.info(f"Source image {src_image_path} with mask {src_mask}")
                samples = torch.cat([self.init_image, self.init_mask], dim=1)
                # Tile to batch size
                samples = samples.repeat(self.args.batch_size, 1, 1, 1)

            if self.args.style or self.args.style_aug:
                logger.info("Styling samples...")
                self.init_image = samples
                self.prev = self.init_image.detach()

                
                styled_samples = self.flux_style_loop(
                    shape=shape,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    init_image=self.init_image,
                    guidance=guidance
                )
                
                for j, styled_sample in enumerate(styled_samples):
                    should_save_image = j % save_image_interval == 0 or j == total_style_steps - 1
                    if should_save_image:
                        styled_img = styled_sample["pred_xstart"]
                        styled_samples = preprocess(styled_img)
                        styled_image = styled_samples.cpu().numpy() 
                        # Last in batch, shape (W, H, 4)
                if self.args.use_colormatch and self.init_image is not None:
                    for img in styled_image:
                        src_image = Normalizer(img[..., :3]).type_norm()
                        arr_pil = np.asarray(self.target_image_pil)
   
                        trg_image = Normalizer(arr_pil).type_norm()
                        img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                        img_res = Normalizer(img_res).uint8_norm()
                        img = np.concatenate([img_res, img[..., -1:]], axis=-1)
                        # logger.info("Styled image shape colormatch", img.shape) # W, H, 4
                        # styled_images.extend([styled_image])
                        curr = styled_images.get(style_img_path, [])
                        curr.extend([img])
                        styled_images[style_img_path] = curr
                    logger.info(f"Styled image shape colormatch {styled_images[style_img_path][-1].shape}") # W, H, 4
                else:
                    curr = styled_images.get(style_img_path, [])
                    curr.extend([img for img in styled_image])
                    styled_images[style_img_path] = curr   

            samples = preprocess(samples)
            # all_images.extend([sample.cpu().numpy() for sample in samples])
            # Add to dict
            # Empty list if key doesn't exist
            curr = all_images.get(style_img_path, [])
            curr.extend([sample.cpu().numpy() for sample in samples])
            # image, mask = curr[-1][..., :3], curr[-1][..., -1:]
            # import matplotlib.pyplot as plt
            # plt.imshow(np.array(image))
            # plt.savefig(f"image_{it}.png")
            # plt.imshow(np.array(mask), cmap="gray")
            # plt.savefig(f"mask_{it}.png")
            all_images[style_img_path] = curr

            if it % 50 == 0:
                logger.info(f"Saving {it} images...")
                self._save(all_images, styled_images)

        if it % 50 != 0: # prevent saving twice
            logger.info(f"Saving {it} images...")        
            self._save(all_images, styled_images)