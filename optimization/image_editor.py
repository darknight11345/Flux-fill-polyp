import os
import blobfile as bf
import pickle
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils_visualize.metrics_accumulator import MetricsAccumulator
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
from optimization.losses import range_loss, d_clip_loss, get_features, zecon_loss_direct
# import lpips
import numpy as np
#from src.vqc_core import *
from model_vit.loss_vit import Loss_vit
#from guided_diffusion.guided_diffusion import dist_util, logger
'''from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils_visualize.visualization import show_tensor_image, show_editied_masked_image
'''
from pathlib import Path
from id_loss import IDLoss
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
from PIL import ImageChops

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
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
import yaml
import inspect
from torchvision.transforms.functional import to_pil_image

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
        
        self.max_sequence_length=512
        
        self.generator=torch.Generator(device=self.device).manual_seed(self.args.seed)

        self.pipe = FluxFillPipeline.from_pretrained(self.model_name,torch_dtype=torch.bfloat16)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_name, subfolder="scheduler")        
           
        #self.pipe.load_lora_weights(os.path.dirname(self.model_path),weight_name=os.path.basename(self.model_path),)
        # Load the tokenizers
        tokenizer_one = CLIPTokenizer.from_pretrained(
            self.model_name,
            subfolder="tokenizer",
            revision=args.revision, # T: need to add this in arguments.py
        )
        tokenizer_two = T5TokenizerFast.from_pretrained(
            self.model_name,
            subfolder="tokenizer_2",
            revision=args.revision,
        )
        
        def import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
            text_encoder_config = PretrainedConfig.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, revision=revision
            )
            model_class = text_encoder_config.architectures[0]
            if model_class == "CLIPTextModel":
                from transformers import CLIPTextModel

                return CLIPTextModel
            elif model_class == "T5EncoderModel":
                from transformers import T5EncoderModel

                return T5EncoderModel
            else:
                raise ValueError(f"{model_class} is not supported.")
        
        
        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.model_name, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.model_name, args.revision, subfolder="text_encoder_2"
        )
        
        text_encoder_one = text_encoder_cls_one.from_pretrained(
                self.model_name, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
                self.model_name, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
        )    

        
        self.tokenizers = [tokenizer_one, tokenizer_two]
        self.text_encoders = [text_encoder_one, text_encoder_two]
        
        

    
        
        #self.pipe.enable_model_cpu_offload()
        
    def tokenize_prompt(self, tokenizer, prompt, max_sequence_length):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids
        
    def _encode_prompt_with_t5(
        self,
        text_encoder,
        tokenizer,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _encode_prompt_with_clip(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,):
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds


    def encode_prompt(
        self,
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids_list=None,
        ):
        
        #logger.info(f"text_encoders[0]: {text_encoders[0]}")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        dtype = text_encoders[0].dtype

        pooled_prompt_embeds = self._encode_prompt_with_clip(
            text_encoder=text_encoders[0],
            tokenizer=tokenizers[0],
            prompt=prompt,
            device=device if device is not None else text_encoders[0].device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
        )

        prompt_embeds = self._encode_prompt_with_t5(
            text_encoder=text_encoders[1],
            tokenizer=tokenizers[1],
            max_sequence_length=max_sequence_length,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device if device is not None else text_encoders[1].device,
            text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids
    

    def compute_text_embeddings(self,prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                text_encoders, tokenizers, prompt, self.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(self.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)
            text_ids = text_ids.to(self.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids  
            
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
        '''
        logger.info(type(self.transformer))
        logger.info(self.transformer.transformer_blocks)
        logger.info(self.transformer.single_transformer_blocks)
        block = self.transformer.transformer_blocks[0]

        logger.info(type(block))
        logger.info(block.__class__.__module__)
        logger.info(inspect.getfile(block.__class__))
        '''
        self.zecon_features = []

        def zecon_hook(module, inputs, output):
            # FluxTransformerBlock output:
            # output[0] = encoder/text hidden states
            # output[1] = image hidden states
            #logger.info(f"HOOK output[0]: {output[0].shape}")
            #logger.info(f"HOOK output[1]: {output[1].shape}")
            self.zecon_features.append(output[1])
            
        for layer_idx in [0, 4, 8, 12, 18]:
            self.transformer.transformer_blocks[layer_idx].register_forward_hook(zecon_hook)
        #self.zecon_hook_handle = (self.transformer.transformer_blocks[0].register_forward_hook(zecon_hook))        

        self.vae = self.pipe.vae
        #self.scheduler = self.pipe.scheduler
        #self.text_encoder = self.pipe.text_encoder
        #self.tokenizer = self.pipe.tokenizer
        self.vae_config_shift_factor = self.vae.config.shift_factor
        self.vae_config_scaling_factor = self.vae.config.scaling_factor
        self.vae_config_block_out_channels = self.vae.config.block_out_channels
        instance_prompt = ""
        #logger.info(f"use_dynamic_shifting: "f"{self.scheduler.config.use_dynamic_shifting}")
        #logger.info(f"Scheduler config: {self.scheduler.config}")
        #logger.info(f"text_encoders len: {len(self.text_encoders)}")
        
        with open(f"{self.root_dir}/model_vit/config.yaml", "r") as ff:
            config = yaml.safe_load(ff)

        cfg = config
        
        self.VIT_LOSS = Loss_vit(cfg, lambda_ssim=self.args.lambda_ssim,lambda_dir_cls=self.args.lambda_dir_cls,lambda_contra_ssim=self.args.lambda_contra_ssim,lambda_trg=self.args.lambda_trg).eval()
            
        self.cm = ColorMatcher()

            # self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

        if self.args.lambda_vgg > 0:
            self.vgg = models.vgg19(pretrained=True).features
            self.vgg.to(self.device)
            self.vgg.eval().requires_grad_(False)
            
        self.vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
        

        with torch.no_grad():
            self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = self.compute_text_embeddings(
            instance_prompt, self.text_encoders, self.tokenizers)




    
    def noisy_aug(self, sigma, x, x_hat):
        x_mix = x_hat * sigma + x * (1 - sigma)
        return x_mix
    
    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep
    
    def zecon_loss(self,x,x_in,y_in,t,prompt_embeds,pooled_prompt_embeds,text_ids, guidance):
                    
        loss = zecon_loss_direct(self,x,x_in,y_in, torch.zeros_like(t,device=self.device),prompt_embeds,pooled_prompt_embeds,text_ids, guidance)
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
        
        size=self.args.model_output_size
        #self.target_image_pil = self.target_image_pil.resize(size, Image.NEAREST)
        train_resize = transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.target_image_pil=train_resize(self.target_image_pil)

        print(f"the size of self.target_image_pil : {self.target_image_pil.size}")
        self.target_mask_pil = self.target_mask_pil.resize(self.target_image_pil.size, Image.NEAREST)
        self.target_mask_pil_undilated=self.target_mask_pil
        
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
            
        self.target_image = (TF.to_tensor(self.target_image_pil).unsqueeze(0).to(device=self.device,dtype=torch.float32)) * 2 - 1 
        self.target_mask = (TF.to_tensor(self.target_mask_pil).unsqueeze(0).to(device=self.device,dtype=torch.float32))
        logger.info(f"shape of self.target_image: {self.target_image.shape}")
        #self.target_image = self.target_image.repeat(self.args.batch_size, 1, 1, 1)
        #self.target_mask = self.target_mask.repeat(self.args.batch_size, 1, 1, 1)
        
        logger.info(f"shape of self.target_image_pil, self.target_mask_pil is {self.target_image_pil}, {self.target_mask_pil}")
        
        self.target_image_pil.save("debug_image.png")
        self.target_mask_pil.save("debug_mask.png")
        
        return self.target_image_pil, self.target_mask_pil, self.target_mask_pil_undilated
        
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

    def _save(self, all_images, styled_images):
        # Apply np.array to all values in dict
        img_dict = {k: np.array(v) for k, v in all_images.items()}
        arr = next(iter(img_dict.values()))
        img_dict_shape = list(arr.shape)
        img_dict_shape[0] = img_dict_shape[0] * len(img_dict)
        # arr = arr[: args.num_samples]
        logger.info(f"Shape of img_dict: {img_dict_shape}")
        
        styled_dict = {k: np.array(v) for k, v in styled_images.items()}
        if styled_dict == {}: # If no styling
            styled_dict = img_dict
        styled_arr = next(iter(styled_dict.values()))
        styled_dict_shape = list(styled_arr.shape)
        styled_dict_shape[0] = styled_dict_shape[0] * len(styled_dict)
        logger.info(f"Shape of styled_dict: {styled_dict_shape}")
        # arr = np.array(all_images)
        # styled_arr = np.array(styled_images)
        
        shape_str = "x".join([str(x) for x in img_dict_shape])
        out_path = os.path.join(self.ranked_results_path, f"samples_{shape_str}.pkl")
        styled_shape_str = "x".join([str(x) for x in styled_dict_shape])
        styled_out_path = os.path.join(self.ranked_results_path, f"styled_samples_{styled_shape_str}.pkl")

        logger.info(f"saving to {out_path}")
        # np.savez(out_path, arr)
        # np.savez(styled_out_path, styled_arr)
        # Save dicts as pkl
        with open(out_path, 'wb') as f:
            pickle.dump(img_dict, f)
        with open(styled_out_path, 'wb') as f:
            pickle.dump(styled_dict, f)

    
    def flux_gen_sample(self,height,width,target_image,target_mask,):
        
        '''
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
        '''
        result = self.pipe(
                image=target_image,
                mask_image=target_mask,
                prompt="",
                height=height,
                width=width,
                guidance_scale=30,
                num_inference_steps=50,
                max_sequence_length=self.max_sequence_length,
                generator=self.generator
        ).images[0]

        #logger.info(f"Result size: {result.size}")  
        
        
        #result.save(f"flux_inpaint_{height}_{width}_lora.png")
        
        return result
        

        #return result

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
        total_steps = 1000 - self.args.skip_timesteps - 1 #this is used for codn_fn

        def cond_fn(x0, sigma, t, x, y=None):
            # if self.args.prompt == "":
            #     return torch.zeros_like(x)
            self.flag_resample=False
            '''
            logger.info(f"x0 shape: {x0.shape}")
            logger.info(f"x shape: {x.shape}")
            logger.info(f"t: {t}")
            logger.info(f"diff_iter: {self.args.diff_iter}")
            '''
            
            
            with torch.enable_grad():
                frac_cont=1.0
                
                x = x.detach().requires_grad_() # T: noisy input latent which used as input to transformer

                #t = self.unscale_timestep(t)
               
                init_image_cond_fn=F.to_tensor(self.init_image).unsqueeze(0).to(device=self.device,dtype=torch.float32) # T: generated sample in inference
                init_image_cond_fn = init_image_cond_fn * 2 - 1 #T: to convert from the range of 0 to 1, to -1 to 1 for VIT loss
                loss = torch.tensor(0) # default value
                if init_image_cond_fn.eq(0).all(): # reconstruction
                    if self.target_image is not None:
                        loss = loss + mse_loss(
                            x_in[:, :3, ...],
                            self.target_image
                        ) * self.args.l2_trg_lambda
                else: # styling
                    if self.args.use_noise_aug_all:
                        #x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                        #logger.info(f"t={t.flatten()[0].item():.6f}, "f"sigma={sigma.flatten()[0].item():.12f}, "f"1-sigma={(1-sigma).flatten()[0].item():.12f}")
                        x_in_latent = self.noisy_aug(sigma,x,x0)
                        x_in_latent = x_in_latent.to(dtype=self.vae.dtype)
                        '''
                        test_input_loss = x_in_latent.float().mean()

                        test_input_grad = torch.autograd.grad(
                            test_input_loss,
                            x,
                            retain_graph=True
                        )[0]

                        logger.info(
                            f"x_in_latent -> x grad norm: "
                            f"{test_input_grad.float().norm().item():.12e}"
                        )
                        logger.info(
                            f"x requires_grad={x.requires_grad}, "
                            f"x_in_latent requires_grad={x_in_latent.requires_grad}, "
                            f"x_in_latent grad_fn={x_in_latent.grad_fn}"
                        )
                        logger.info(f"x_in_latent shape: {x_in_latent.shape}") 
                        '''                        
                        x_in = self.vae.decode(x_in_latent / self.vae.config.scaling_factor + self.vae.config.shift_factor,return_dict=False,)[0]
                        
                        #logger.info(f"x_in decoded shape: {x_in.shape}")
                        #logger.info(f"x_in  dtype: {x_in.dtype}")
                    else:
                        x_in = x0
                        #x_in = out["pred_xstart"]
                    # self.init_image = (B,4,H,W)
                    x0_decoded = self.vae.decode(x0 / self.vae.config.scaling_factor + self.vae.config.shift_factor,return_dict=False,)[0]
                    #logger.info(f"pred x0 shape: {x0.shape}")

                    x_in3 = x_in[:, :3, ...].float() 
                    # init_image_batch = torch.tile(self.init_image[:3, ...], dims=(self.args.batch_size, 1, 1, 1))
                    # zecon_init_image_batch = torch.tile(self.init_image, dims=(self.args.batch_size, 1, 1, 1))
                    # self.prev = torch.tile(self.prev[:3, ...], dims=(self.args.batch_size, 1, 1, 1))
                    init_image_batch = init_image_cond_fn[:, :3, ...]
                    zecon_init_image_batch = init_image_cond_fn.to(dtype=self.transformer.dtype)
                    prev_cond_fn=F.to_tensor(self.prev).unsqueeze(0).to(device=self.device,dtype=torch.float32)    # T: self.prev = self.init_image
                    prev_cond_fn = prev_cond_fn * 2 - 1 #T: to convert from the range of 0 to 1, to -1 to 1 for VIT loss

                    #ogger.info(f"prev_cond_fn shape: {prev_cond_fn.shape}")      
                    # Compute DINO feature of previous image ONCE
                    #self.VIT_LOSS.set_prev_image(prev_cond_fn)                    
                    #self.prev = self.prev[:, :3, ...]
                    #logger.info(f"x_in3 range: {x_in3.min().item():.4f} to {x_in3.max().item():.4f}")
                    #logger.info(f"init range: {init_image_batch.min().item():.4f} to {init_image_batch.max().item():.4f}")
                    #logger.info(f"prev range: {prev_cond_fn.min().item():.4f} to {prev_cond_fn.max().item():.4f}")

                    if self.args.vit_lambda != 0:     
                        # self.init_image is x_src  
                        if t[0].item()>self.args.diff_iter : # directional cls
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in3, init_image_batch,prev_cond_fn,use_dir=True,frac_cont=frac_cont,target = self.target_image)
                        else:
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in3,init_image_batch,prev_cond_fn,use_dir=False,frac_cont=frac_cont,target = self.target_image)
                        loss = loss + vit_loss

                    if self.args.range_lambda != 0:
                        #r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                        r_loss = range_loss(x0).sum() * self.args.range_lambda
                        loss = loss + r_loss
                        self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                    if self.target_image is not None:
                        loss = loss + mse_loss(x_in3, self.target_image) * self.args.l2_trg_lambda

                    self.prev = F.to_pil_image(x_in3[0].detach().cpu().clamp(-1, 1),mode="RGB")

                    # ------------------  New Losses ------------------
                    #fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]             

                    if not self.args.use_noise_aug_all:
                        x_in = ((1-sigma)*x0+sigma*x)
                    
                    current_sigma = sigma.flatten()[0].item()


                    if self.args.lambda_zecon != 0:
                        #y_t = self.diffusion.q_sample(zecon_init_image_batch,t)
                        #y_in = zecon_init_image_batch * fac + y_t * (1 - fac)
                        
                        init_latent = self.vae.encode(zecon_init_image_batch.to(device=self.device,dtype=self.vae.dtype)).latent_dist.sample()
                        init_latent = (init_latent - self.vae.config.shift_factor)*self.vae.config.scaling_factor
                        noise_ref = torch.randn_like(init_latent)
                        y_in = ((1-sigma)*init_latent+sigma*noise_ref) #noisy reference latent
                        
                        #zecon_loss = self.zecon_loss(x_in, y_in,t) * self.args.lambda_zecon
                        
                        zecon_loss = self.zecon_loss(
                                        x,
                                        x_in_latent,
                                        y_in,
                                        t/1000,
                                        self.prompt_embeds,
                                        self.pooled_prompt_embeds,
                                        self.text_ids,                                        
                                        guidance=30
                                    ) * self.args.lambda_zecon
                        '''
                        logger.info(
                            f"ZECon loss = {zecon_loss.item():.12e}"
                        )
                        logger.info(
                            f"ZECon requires_grad = {zecon_loss.requires_grad}"
                        )
                        logger.info(
                            f"ZECon grad_fn = {zecon_loss.grad_fn}"
                        )

                        zecon_grad = torch.autograd.grad(
                            zecon_loss,
                            x,
                            retain_graph=True,
                            allow_unused=True
                        )[0]

                        if zecon_grad is None:
                            logger.info("!!! ZeCon gradient wrt x is NONE !!!")
                        else:
                            logger.info(
                                f"ZeCon grad norm = "
                                f"{zecon_grad.float().norm().item():.12e}"
                            )
                            logger.info(
                                f"ZeCon grad max = "
                                f"{zecon_grad.float().abs().max().item():.12e}"
                            )
                        '''
                        loss = loss + zecon_loss
                        self.metrics_accumulator.update_metric("zecon_loss", zecon_loss.item())
                    
                    if self.args.lambda_vgg != 0 and current_sigma < 0.8:  #t[0].item() < 800:
                        '''
                        y_t = self.diffusion.q_sample(init_image_batch,t)
                        y_in = init_image_batch * fac + y_t * (1 - fac)
                        '''
                        #logger.info("running vgg_loss")
                        noise_ref = torch.randn_like(init_image_cond_fn)
                        y_in = ((1-sigma)*init_image_cond_fn+sigma*noise_ref)
                        #y_in = self.vae.decode(y_t / self.vae.config.scaling_factor).sample
                        
                        vgg_loss = self.vgg_loss(x_in3, y_in) * self.args.lambda_vgg
                        loss = loss + vgg_loss
                        self.metrics_accumulator.update_metric("vgg_loss", vgg_loss.item())
                    if self.args.lambda_mse != 0 and current_sigma < 0.7: #t[0].item() < 700:
                        #y_t = self.diffusion.q_sample(init_image_batch, t)
                        #y_in = init_image_batch * fac + y_t * (1 - fac)
                        #logger.info("running cnt_mse_loss")
                        noise_ref = torch.randn_like(init_image_cond_fn)
                        y_in = ((1-sigma)*init_image_cond_fn+sigma*noise_ref)
                        #y_in = self.vae.decode(y_t / self.vae.config.scaling_factor).sample

                        cnt_mse_loss = self.cnt_mse_loss(x_in3, y_in) * self.args.lambda_mse
                        loss = loss + cnt_mse_loss
                        self.metrics_accumulator.update_metric("cnt_mse_loss", cnt_mse_loss.item())

                    # ------------------  New Losses End ------------------
                    
                    if self.args.use_range_restart:
                        if t[0].item() < total_steps:
                            if r_loss>0.01:
                                    self.flag_resample =True
                                    
            if loss.eq(0).all():
                #logger.info("!!! TOTAL LOSS IS ZERO !!!")
                grad = torch.zeros_like(x)
            else:
                #logger.info(f"TOTAL LOSS = {loss.item():.12e}")
                #logger.info(f"LOSS REQUIRES GRAD = {loss.requires_grad}")
                #logger.info(f"LOSS GRAD FN = {loss.grad_fn}")

                grad = torch.autograd.grad(loss,x,retain_graph=False,allow_unused=False)[0]

                #logger.info(f"GRAD FP32 NORM = {grad.float().norm().item():.12e}")
                #logger.info(f"GRAD FP32 MAX = {grad.float().abs().max().item():.12e}")

            return -grad, self.flag_resample
            #return (-torch.autograd.grad(loss, x)[0] if not loss.eq(0).all() else loss), self.flag_resample

        
        # [-1, 1] -> [0, 255]
        def preprocess(sample): # sample is image tensor
            if sample.dim() == 3:
                sample = sample.unsqueeze(0)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            logger.info(f"the shape of styled sample in preprocess is {sample.shape}")
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
            image = torch.from_numpy(image).to(device=self.device,dtype=torch.float32)
            image = image / 127.5 - 1.0
            logger.info(f"image_pil: {image.shape}") #torch.Size([1, 3, 512, 512])

            # mask is already a torch tensor in [0, 1]
            mask = mask.to(device=self.device,dtype=torch.float32,)

            # For your zero mask:
            # mask = 0 everywhere
            # therefore masked_image = image everywhere
            masked_image = image * (mask < 0.5).to(image.dtype)
            
            return mask, masked_image
    

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
            schedule_timesteps = self.scheduler.timesteps.to(self.device)
            #logger.info(f"schedule_timesteps: {schedule_timesteps}")            
            timesteps = timesteps.to(self.device)
            if timesteps.ndim == 0:
                timesteps = timesteps.unsqueeze(0)
            step_indices = [(schedule_timesteps == t).nonzero(as_tuple=True)[0].item() for t in timesteps]
            #logger.info(f"step_indices: {step_indices}") 
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma


       
        def flux_style_loop(
            self,
            init_image,
            guidance=30):
            
            image_pil=init_image
            # ---------------------------------------------------------
            # 1. Encode Stage-1 generated image
            # ---------------------------------------------------------            
            image = exif_transpose(image_pil)
            if not image.mode == "RGB":
                image = image.convert("RGB")
                
            #logger.info(f"image_pil: {image.shape}")
            #logger.info(f"image_pil: {image_pil.height}")
            #logger.info(f"image_pil: {image_pil.width}")
                
            train_resize = transforms.Resize(self.args.model_output_size, interpolation=transforms.InterpolationMode.BILINEAR)
            image = train_resize(image)
            
            train_crop = transforms.CenterCrop(self.args.model_output_size) if self.args.center_crop else transforms.RandomCrop(self.args.model_output_size)
            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )        
            y1, x1, h, w = train_crop.get_params(image, (self.args.model_output_size, self.args.model_output_size))
            image = crop(image, y1, x1, h, w)
            
            image_pixel_values = train_transforms(image)
            image_pixel_values = image_pixel_values.unsqueeze(0) ## T: to add batch dimension
            # Move to VAE device and dtype
            image_pixel_values = image_pixel_values.to(device=self.device,dtype=self.vae.dtype,)
            model_input = self.vae.encode(image_pixel_values).latent_dist.sample()
            model_input = (model_input - self.vae_config_shift_factor) * self.vae_config_scaling_factor
            model_input = model_input.to(dtype=self.transformer.dtype)
            #logger.info(f"model_input shape: {model_input.shape}")
            

            
            image_seq_len = model_input.shape[2] * model_input.shape[3] // 4
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps,device=self.device, mu=mu)
            all_timesteps = self.scheduler.timesteps
            all_sigmas = self.scheduler.sigmas
            '''
            logger.info(f"len timesteps = {len(all_timesteps)}")
            logger.info(f"len sigmas = {len(all_sigmas)}")
            logger.info(f"timesteps[:5] = {all_timesteps[:5]}")
            logger.info(f"sigmas[:5] = {all_sigmas[:5]}")
            
            logger.info(f"scheduler.config.num_train_timesteps: {self.scheduler.config.num_train_timesteps}")
            logger.info(f"scheduler timesteps = "f"{len(self.scheduler.timesteps)}")
            '''
            self.vae_scale_factor = 2 ** (len(self.vae_config_block_out_channels) -1)
            #logger.info(f"vae config block channels: {self.vae_config_block_out_channels}")
            #logger.info(f"vae scale factor: {self.vae_scale_factor}")
            vae_scale_factor=self.vae_scale_factor
                    
            self.latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                self.device,
                model_input.dtype,
            )
            '''
            logger.info(f"latent_image_ids shape : {self.latent_image_ids.shape}")
            logger.info(f"self.transformer.dtype : {self.transformer.dtype}")
            logger.info(f"image_pixel_values:{image_pixel_values.shape}")
            logger.info(f"model_input: {model_input.shape}")
            '''
            
            
            
            # ---------------------------------------------------------
            # 2. Prepare Fill conditioning
            #
            # For styling, we do NOT want to generate a new mask.
            #
            # mask = 0 means:
            # "the whole image is known/conditioning image"
            # ---------------------------------------------------------            
            mask = torch.zeros(
                (1,1,image_pil.height,image_pil.width),
                device=self.device,
                dtype=torch.float32,
            )
            #logger.info(f"mask:{mask.shape}")
            mask, masked_image = prepare_mask_and_masked_image(image, mask)
            mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
            mask = mask.view(model_input.shape[0], model_input.shape[2], vae_scale_factor, model_input.shape[3], vae_scale_factor)  # batch_size, height, 8, width, 8
            mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
            mask = mask.reshape(model_input.shape[0], vae_scale_factor * vae_scale_factor, model_input.shape[2], model_input.shape[3])        

                #print("packed masked image latents", masked_image_latents.shape)
                #print("model input", model_input)
                #print("model inputhshae", model_input.shape)
            mask = FluxFillPipeline._pack_latents(
                    mask,
                    batch_size=model_input.shape[0],
                    num_channels_latents=vae_scale_factor*vae_scale_factor,
                    height=model_input.shape[2],
                    width=model_input.shape[3],
            )
            #print("packed mask ", mask.shape)
            masked_image = masked_image.to(device=self.device,dtype=self.vae.dtype,)
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
            masked_image_latents = (masked_image_latents - self.vae_config_shift_factor) * self.vae_config_scaling_factor
                
                
            masked_image_latents = FluxFillPipeline._pack_latents(
                    masked_image_latents,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
            )            
            mask = mask.to(device=self.device,dtype=self.transformer.dtype) 
            masked_image_latents = masked_image_latents.to(device=self.device,dtype=self.transformer.dtype)            
            masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
            #logger.info(f"masked_image_latents dtype: {masked_image_latents.dtype}")
            #logger.info(f"mask dtype: {mask.dtype}")
            #logger.info(f"transformer dtype: {self.transformer.dtype}")
            #print("concat masked image latents", masked_image_latents.shape)   
            
                
            

            # ---------------------------------------------------------
            # 3. Prepare text conditioning --  implemented in class imageeditor
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # 4. FLUX scheduler --- implemented FlowMatchEulerDiscreteScheduler in class imageeditor
            # ---------------------------------------------------------


            # ---------------------------------------------------------
            # 5. Start from a noised version of Stage-1 image
            #
            # This is the FLUX equivalent of the UNet styling:
            #
            #     img = q_sample(init_image, timestep)
            # ---------------------------------------------------------

            noise = torch.randn_like(model_input)
            #timesteps = self.scheduler.timesteps
            skip_timesteps = self.args.style_skip_timesteps
            #print(self.scheduler.timesteps[:5])
            #print(self.scheduler.timesteps[-5:])
            #print(len(self.scheduler.timesteps))
            #start_index = ((self.scheduler.timesteps - self.args.style_skip_timesteps).abs()).argmin()
            #timesteps = timesteps[:len(timesteps) - skip_timesteps].to(device=model_input.device)
            num_steps = len(all_timesteps) - skip_timesteps

            timesteps = all_timesteps[skip_timesteps:]
            sigmas = all_sigmas[skip_timesteps:]
            start_t = timesteps[0]
            start_sigma = sigmas[0]

            #logger.info(f"timesteps[0]: {timesteps[0]}")
            #logger.info(f"timesteps[1]: {timesteps[1]}")
            sigmas_fp32 = self.scheduler.sigmas.to(device=self.device,dtype=torch.float32)
            sigmas_bf16 = sigmas_fp32.to(torch.bfloat16)

            #logger.info(f"timesteps: {timesteps[:5]}")
            sigma_x0 = start_sigma.to(device=model_input.device)#, dtype=torch.float32,)
            sigma_x0=sigma_x0.to(dtype=model_input.dtype,)
            #sigma_x0 = get_sigmas(timesteps[0].unsqueeze(0), n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = ((1-sigma_x0)*model_input+sigma_x0*noise)

            #logger.info(f"masked_image_latents: {masked_image_latents.shape}")
            

            guidance = torch.tensor([guidance], device=self.device)
            guidance = guidance.expand(model_input.shape[0])
            j=0  
                   
            # this is to log for test purpose
            packed_latents_test = FluxFillPipeline._pack_latents(
                                    model_input,
                                    batch_size=model_input.shape[0],
                                    num_channels_latents=model_input.shape[1],
                                    height=model_input.shape[2],
                                    width=model_input.shape[3],
                                )
            logger.info(
                f"Initial latent before styling loop without noise: min={packed_latents_test.min().item():.4f}, "
                f"max={packed_latents_test.max().item():.4f}, "
                f"mean={packed_latents_test.mean().item():.4f}, "
                f"std={packed_latents_test.std().item():.4f}"
            )
            with torch.no_grad():
                test_image = self.vae.decode(
                    model_input / self.vae_config_scaling_factor
                    + self.vae_config_shift_factor,
                    return_dict=False,
                )[0]
                   
            decoded = test_image.clamp(-1, 1)
            decoded = ((decoded + 1) / 2 * 255).to(torch.uint8)
            decoded = decoded[0].permute(1, 2, 0).cpu().numpy()

            Image.fromarray(decoded).save(f"debug_decoded_generated_sample.png")

            #test_image_pil.save("debug_decoded_generated_sample.png")
            logger.info(
                f"Decoded image before styling: "
                f"shape={test_image.shape}, "
                f"min={test_image.min().item():.4f}, "
                f"max={test_image.max().item():.4f}, "
                f"mean={test_image.mean().item():.4f}, "
                f"std={test_image.std().item():.4f}"
            )
            
            
            for step_idx, (t, sigma) in enumerate(zip(timesteps, sigmas)):
            
                with torch.no_grad():
            
                    #sigma = get_sigmas(t.unsqueeze(0), n_dim=model_input.ndim, dtype=model_input.dtype)
                    sigma = sigma.reshape(1).to(device=self.device)
                    #dtype=torch.float32,)
                    sigma = sigma.to(dtype=model_input.dtype,)
                    #sigma_fp32 = sigmas_fp32[step_idx].flatten()
                    #sigma_bf16 = sigma_fp32.to(torch.bfloat16)

                    #logger.info(f"sigma FP32: {sigma_fp32.flatten().tolist()}")

                    #logger.info(f"sigma BF16: {sigma_bf16.flatten().tolist()}")

                    #timestep = t.to(self.device).expand(noisy_model_input.shape[0])
                    
                    #timestep = (t.reshape(1).to(device=self.device, dtype=model_input.dtype).expand(noisy_model_input.shape[0]))
                    timestep = (t.reshape(1).to(device=self.device).expand(noisy_model_input.shape[0]))
                    logger.info(f"step={step_idx}, "f"t={t.item():.3f}, "f"sigma={sigma.item():.9f}, "f"sigma_bf16={sigma.to(torch.bfloat16).item():.9f}, "f"1-sigma={(1-sigma).item():.9f}")
                    
                    #logger.info(f" the timestep and len is : {timestep} , {len(timestep)}")
                    
                    packed_latents = FluxFillPipeline._pack_latents(
                        noisy_model_input,
                        batch_size=noisy_model_input.shape[0],
                        num_channels_latents=noisy_model_input.shape[1],
                        height=noisy_model_input.shape[2],
                        width=noisy_model_input.shape[3],
                    )

                    transformer_input = torch.cat((packed_latents, masked_image_latents), dim=2) 
                    #logger.info(f"packed noisy: {packed_latents.shape}")
                    #logger.info(f"transformer input: {transformer_input.shape}")                
                    
                    velocity = self.transformer(
                        hidden_states=transformer_input,
                        timestep=timestep/1000,
                        guidance=guidance,
                        pooled_projections=self.pooled_prompt_embeds,
                        encoder_hidden_states=self.prompt_embeds,
                        txt_ids=self.text_ids,
                        img_ids=self.latent_image_ids,
                    )[0]
                    
                    velocity = FluxFillPipeline._unpack_latents(
                        velocity,
                        height=model_input.shape[2] * vae_scale_factor,
                        width=model_input.shape[3] * vae_scale_factor,
                        vae_scale_factor=vae_scale_factor,
                    )
                    
                    #sigma = t / 1000
                    
                    pred_x0 = noisy_model_input - sigma * velocity
                    
                    
                    
                    #noisy_model_input = self.scheduler.step(velocity,t,noisy_model_input)[0]
                    
                    #noisy_model_input = (noisy_model_input / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    #decoded_x0 = self.vae.decode(pred_x0 / self.vae.config.scaling_factor+ self.vae.config.shift_factor,return_dict=False,)[0]
                        
                    grad, flag_resample=cond_fn(pred_x0,sigma,timestep,noisy_model_input)
                    
                    guided_velocity = velocity + 0.01 * grad # self.args.guidance_scale = 0.01

                    noisy_model_input = self.scheduler.step(
                        guided_velocity,
                        timestep,
                        noisy_model_input,
                    ).prev_sample
                    #logger.info(f"velocity norm = {velocity.float().norm().item():.6e}")
                    #logger.info(f"grad norm     = {grad.float().norm().item():.6e}")
                    #logger.info(f"guided delta  = "f"{(0.01 * grad).float().norm().item():.6e}") #self.args.guidance_scale
                    #logger.info(f"grad shape: {grad.shape}")
                    #logger.info(f"grad dtype: {grad.dtype}")
                    #logger.info(f"grad norm: {grad.norm().item():.6f}")
                    
                    '''
                    # T: below code is just to check the decoded image, debugging code
                    decoded = self.vae.decode(pred_x0, return_dict=False)[0]
                    
                    decoded = decoded.clamp(-1, 1)
                    decoded = ((decoded + 1) / 2 * 255).to(torch.uint8)
                    decoded = decoded[0].permute(1, 2, 0).cpu().numpy()

                    Image.fromarray(decoded).save(f"styled_step_{t}.png")
                    '''
                    #j=j+1
                    #if j==5:                       
                        #break
                
        
            logger.info(
                f"Final styled latent: shape={noisy_model_input.shape}, "
                f"dtype={noisy_model_input.dtype}, "
                f"min={noisy_model_input.min().item():.4f}, "
                f"max={noisy_model_input.max().item():.4f}, "
                f"mean={noisy_model_input.mean().item():.4f}, "
                f"std={noisy_model_input.std().item():.4f}"
            )
            with torch.no_grad():
                style_image = self.vae.decode(
                    noisy_model_input / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor,
                    return_dict=False,
                )[0]
            logger.info(
                f"Decoded styled image: "
                f"shape={style_image.shape}, "
                f"min={style_image.min().item():.4f}, "
                f"max={style_image.max().item():.4f}, "
                f"mean={style_image.mean().item():.4f}, "
                f"std={style_image.std().item():.4f}"
            )
            
            #logger.info(f"decoded.shape: {decoded.shape}")
            return style_image               

        

        
        all_images = {}
        styled_images = {}
        total_style_steps = 1000 - self.args.style_skip_timesteps  #80
        save_image_interval = total_style_steps // 5
        num_samples = self.args.num_samples * len(all_files) if self.args.sample_per_image else self.args.num_samples
        logger.info(f"len(all_files) no of image in cluster index: {len(all_files)}")
        logger.info(f"num_samples {num_samples}")
        logger.info(f"Sampling {num_samples * self.args.batch_size} images")
        while it < 1: #num_samples
            # Sets target_image and target_mask
            target_image, target_mask, target_mask_pil_undilated = self._get_target_image_and_mask(all_files, it=it)
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
                    height=self.args.model_output_size,
                    width=self.args.model_output_size,
                    target_image=target_image, 
                    target_mask=target_mask,
                )
                '''
                if self.flag_resample:
                    continue
                samples = torch.stack([refine_mask(sample[i]) for i in samples]).squeeze(0)
                '''
                
                logger.info(f"sample.shape: {samples.size}") # [batch_size, 4, 256, 256] for unet  , [batch_size, 16, 256, 256] for flux 
                logger.info(f"type={type(samples)}, mode={samples.mode}, size={samples.size}")
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
                #self.prev = self.init_image.detach()
                self.prev = self.init_image

                
                styled_samples = flux_style_loop(
                    self,
                    self.init_image
                )
                
                for j, styled_sample in enumerate(styled_samples):
                    should_save_image = j % save_image_interval == 0 or j == total_style_steps - 1
                    if should_save_image:
                        styled_img = styled_sample
                        styled_samples_processed = preprocess(styled_img)
                        styled_image = styled_samples_processed.cpu().numpy() 
                        # Last in batch, shape (W, H, 4)
                if self.args.use_colormatch and self.init_image is not None:
                    for img in styled_image:
                        logger.info(f"img shape after styling: {img.shape}")
                        src_image = Normalizer(img[..., :3]).type_norm()
                        arr_pil = np.asarray(self.target_image_pil)
   
                        trg_image = Normalizer(arr_pil).type_norm()
                        img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                        img_res = Normalizer(img_res).uint8_norm()
                        mask_for_styled = np.asarray(self.target_mask_pil_undilated)
                        if mask_for_styled.ndim == 2:
                            mask_for_styled = mask_for_styled[..., None]

                        img = np.concatenate([img_res, mask_for_styled],axis=-1)
                        logger.info(f"Styled image shape after concat of mask {img.shape}") # W, H, 4
                        # styled_images.extend([styled_image])
                        curr = styled_images.get(style_img_path, [])
                        curr.extend([img])
                        styled_images[style_img_path] = curr
                    logger.info(f"Styled image shape colormatch {styled_images[style_img_path][-1].shape}") # W, H, 4
                else:
                    curr = styled_images.get(style_img_path, [])
                    curr.extend([img for img in styled_image])
                    styled_images[style_img_path] = curr   

            samples_tensor = F.to_tensor(samples).unsqueeze(0)
            samples_tensor = samples_tensor * 2 - 1

            samples_processed = preprocess(samples_tensor)
            #samples = preprocess(samples)
            # all_images.extend([sample.cpu().numpy() for sample in samples])
            # Add to dict
            # Empty list if key doesn't exist
            logger.info(f"style_img_path before preprocessingthe generated samples: {style_img_path}") 
            curr = all_images.get(style_img_path, [])
            curr.extend([sample.cpu().numpy() for sample in samples_processed])
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
        logger.info("generating samples completed")
        if it % 50 != 0: # prevent saving twice
            logger.info(f"Saving {it} images...")        
            self._save(all_images, styled_images)