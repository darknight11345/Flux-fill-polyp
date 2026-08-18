import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from diffusers import FluxFillPipeline
import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",)

logger = logging.getLogger(__name__)

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance
def get_features(image, model, layers=None):
    
    if layers is None:
        layers = {'0': 'conv1_1', 
                  '2': 'conv1_2', 
                  '5': 'conv2_1',  
                  '7': 'conv2_2',
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device=self.device,dtype=torch.float32)
    image = image / 127.5 - 1.0
    #logger.info(f"image_pil: {image.shape}") #torch.Size([1, 3, 512, 512])

    # mask is already a torch tensor in [0, 1]
    mask = mask.to(device=self.device,dtype=torch.float32,)

            # For your zero mask:
            # mask = 0 everywhere
            # therefore masked_image = image everywhere
    masked_image = image * (mask < 0.5).to(image.dtype)
            
    return mask, masked_image
    
def zecon_loss_direct(self,x,x_in,y_in,t,prompt_embeds,pooled_prompt_embeds,text_ids, guidance):
    
    total_loss = 0
    
    # choose FLUX transformer layers
    #nce_layers = [0,4,8,12,18]
    num_patches = 256
    l2norm = Normalize(2)
    '''
    logger.info(f"x_in {x_in.shape}")
    logger.info(f"y_in {y_in.shape}")
    logger.info(f"transformer input channels: {self.transformer.config.in_channels}")
    logger.info(f"x_embedder: {self.transformer.x_embedder}")
    '''
    
    vae_scale_factor=self.vae_scale_factor
    
    mask = torch.zeros((1,1,self.args.model_output_size,self.args.model_output_size),device=self.device,dtype=self.transformer.dtype,)    
    #mask, masked_image = prepare_mask_and_masked_image(image, mask)
    mask = mask.to(device=self.device,dtype=self.transformer.dtype,)
    mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
    mask = mask.view(x_in.shape[0], x_in.shape[2], vae_scale_factor, x_in.shape[3], vae_scale_factor)  # batch_size, height, 8, width, 8
    mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    mask = mask.reshape(x_in.shape[0], vae_scale_factor * vae_scale_factor, x_in.shape[2], x_in.shape[3])
    mask_x_in = FluxFillPipeline._pack_latents(
                    mask,
                    batch_size=x_in.shape[0],
                    num_channels_latents=vae_scale_factor*vae_scale_factor,
                    height=x_in.shape[2],
                    width=x_in.shape[3],
    )   
    #logger.info(f"mask {mask_x_in.shape} {mask_x_in.dtype}")
    
    assert x_in.shape == y_in.shape, (f"x_in {x_in.shape} != y_in {y_in.shape}")

    latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                x_in.shape[0],
                x_in.shape[2] // 2,
                x_in.shape[3] // 2,
                self.device,
                x_in.dtype,
    )
    packed_latents_x_in = FluxFillPipeline._pack_latents(
                        x_in,
                        batch_size=x_in.shape[0],
                        num_channels_latents=x_in.shape[1],
                        height=x_in.shape[2],
                        width=x_in.shape[3],
    )
    packed_masked_image_x_in = FluxFillPipeline._pack_latents(
        x_in,
        batch_size=x_in.shape[0],
        num_channels_latents=x_in.shape[1],
        height=x_in.shape[2],
        width=x_in.shape[3],
    )

    #logger.info(f"mask_x_in {mask_x_in.shape} {mask_x_in.dtype}")

    masked_image_latents_x_in = torch.cat((packed_masked_image_x_in, mask_x_in), dim=-1)
    transformer_input_x_in = torch.cat((packed_latents_x_in, masked_image_latents_x_in), dim=2) 
    #logger.info(f"transformer_input_x_in {transformer_input_x_in.shape} {transformer_input_x_in.dtype}")
    guidance = torch.tensor([guidance], device=self.device)
    guidance = guidance.expand(x_in.shape[0])
    
    self.zecon_features = []
    '''
    test_packed = packed_latents_x_in.float().mean()

    test_packed_grad = torch.autograd.grad(
        test_packed,
        x,
        retain_graph=True,
        allow_unused=True
    )[0]

    if test_packed_grad is None:
        logger.info("packed_latents_x_in -> x gradient = NONE")
    else:
        logger.info(
            f"packed_latents_x_in -> x grad norm: "
            f"{test_packed_grad.float().norm().item():.12e}"
        )
    
    test_transformer_input = transformer_input_x_in.float().mean()

    test_transformer_grad = torch.autograd.grad(
        test_transformer_input,
        x,
        retain_graph=True,
        allow_unused=True
    )[0]

    if test_transformer_grad is None:
        logger.info("transformer_input_x_in -> x gradient = NONE")
    else:
        logger.info(
            f"transformer_input_x_in -> x grad norm: "
            f"{test_transformer_grad.float().norm().item():.12e}"
        )
    '''
    _ = self.transformer(
        hidden_states=transformer_input_x_in,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        guidance=guidance,
    )

    feat_q = self.zecon_features
    '''
    for i, feat in enumerate(feat_q):
        logger.info(
            f"Q[{i}]: shape={feat.shape}, "
            f"requires_grad={feat.requires_grad}, "
            f"grad_fn={feat.grad_fn}"
        )
    test_q_loss = feat_q[0].float().mean()

    test_q_grad = torch.autograd.grad(
        test_q_loss,
        x,
        retain_graph=True
    )[0]

    logger.info(
        f"Q feature -> x grad norm: "
        f"{test_q_grad.float().norm().item():.12e}"
    )

    logger.info(f"feat_q shape {feat_q[0].shape}")
    '''
    
    ########### y_in ###############
    
    mask_y_in = FluxFillPipeline._pack_latents(
                    mask,
                    batch_size=y_in.shape[0],
                    num_channels_latents=vae_scale_factor*vae_scale_factor,
                    height=y_in.shape[2],
                    width=y_in.shape[3],
    )   
    #logger.info(f"mask {mask_y_in.shape}")
    


    latent_image_ids_y_in = FluxFillPipeline._prepare_latent_image_ids(
                y_in.shape[0],
                y_in.shape[2] // 2,
                y_in.shape[3] // 2,
                self.device,
                y_in.dtype,
    )
    packed_latents_y_in = FluxFillPipeline._pack_latents(
                        y_in,
                        batch_size=y_in.shape[0],
                        num_channels_latents=y_in.shape[1],
                        height=y_in.shape[2],
                        width=y_in.shape[3],
    )
    packed_masked_image_y_in = FluxFillPipeline._pack_latents(
        y_in,
        batch_size=y_in.shape[0],
        num_channels_latents=y_in.shape[1],
        height=y_in.shape[2],
        width=y_in.shape[3],
    )
    masked_image_latents_y_in = torch.cat((packed_masked_image_y_in, mask_y_in), dim=-1)
    transformer_input_y_in = torch.cat((packed_latents_y_in, masked_image_latents_y_in), dim=2) 
    

    
    
    self.zecon_features = []

    _ = self.transformer(
        hidden_states=transformer_input_y_in,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        guidance=guidance,
    )

    feat_k = self.zecon_features
    '''
    for i, feat in enumerate(feat_k):
        logger.info(
            f"K[{i}]: shape={feat.shape}, "
            f"requires_grad={feat.requires_grad}, "
            f"grad_fn={feat.grad_fn}"
        )
        
    logger.info(f"feat_k shape {feat_k[0].shape}")
    '''

    patch_ids = []
    feat_k_pool = []
    feat_q_pool = []
    '''
    for i, (q, k) in enumerate(zip(feat_q, feat_k)):
        logger.info(f"{i},q: {q.shape} k: {k.shape} ")
    '''
    # -------- reference image features --------
    for feat_id, feat in enumerate(feat_k):

        # FLUX feature:
        # [B, tokens, channels]
        feat_reshape = feat

        num_tokens = feat_reshape.shape[1]
        patch_id = torch.randperm(num_tokens,device=feat.device)
        patch_id = patch_id[:min(num_patches,num_tokens)]
        x_sample = feat_reshape[:, patch_id, :]
        # [B,P,C] -> [B*P,C]
        x_sample = x_sample.flatten(0,1)
        patch_ids.append(patch_id)
        x_sample = l2norm(x_sample)
        feat_k_pool.append(x_sample)

    # -------- generated image features --------
    for feat_id, feat in enumerate(feat_q):

        feat_reshape = feat
        patch_id = patch_ids[feat_id]
        x_sample = feat_reshape[:,patch_id,:]
        x_sample = x_sample.flatten(0,1)
        x_sample = l2norm(x_sample)
        feat_q_pool.append(x_sample)


    for f_q,f_k in zip(feat_q_pool,feat_k_pool):

        loss = PatchNCELoss(f_q,f_k,batch_size=x_in.shape[0])
        #logger.info(f"PatchNCE loss: {loss.mean().item():.12e}")
        '''
        # TEST 2
        test_grad_fq = torch.autograd.grad(
            loss.mean(),
            f_q,
            retain_graph=True,
            allow_unused=True
        )[0]

        if test_grad_fq is None:
            logger.info("!!! PatchNCE -> f_q gradient is NONE !!!")
        else:
            logger.info(
                f"PatchNCE -> f_q grad norm: "
                f"{test_grad_fq.float().norm().item():.12e}"
            )
            logger.info(
                f"PatchNCE -> f_q grad max: "
                f"{test_grad_fq.float().abs().max().item():.12e}"
            )
        '''
        total_loss += loss.mean()

    return total_loss / len(feat_q_pool)

def PatchNCELoss(feat_q, feat_k, batch_size=1, nce_T = 0.07):
    # feat_q : n_patch x 512
    # feat_q : n_patch x 512
    batch_size = batch_size
    nce_T = nce_T
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    mask_dtype = torch.bool

    num_patches = feat_q.shape[0]
    dim = feat_q.shape[1]
    feat_k = feat_k.detach()
    
    # pos logit 
    l_pos = torch.bmm(
        feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
    l_pos = l_pos.view(num_patches, 1)

    # reshape features to batch size
    feat_q = feat_q.view(batch_size, -1, dim)
    feat_k = feat_k.view(batch_size, -1, dim)
    npatches = feat_q.size(1)
    l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

    # diagonal entries are similarity between same features, and hence meaningless.
    # just fill the diagonal with very small number, which is exp(-10) and almost zero
    diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[None, :, :]
    l_neg_curbatch.masked_fill_(diagonal, -10.0)
    l_neg = l_neg_curbatch.view(-1, npatches)

    out = torch.cat((l_pos, l_neg), dim=1) / nce_T

    loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                    device=feat_q.device))

    return loss

def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])