from pyexpat import model
import open_clip
from open_clip import tokenizer

import os, sys
sys.path.append("./latent-diffusion")
sys.path.append("./taming-transformers")
from taming.models import vqgan

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange
tqdm_auto_model = __import__("tqdm.auto", fromlist=[None]) 
sys.modules['tqdm'] = tqdm_auto_model
from einops import rearrange
from torchvision.utils import make_grid
import gc
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import shortuuid
import streamlit as st
st.set_page_config(layout="wide")


def generate_id(length=8):
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(length)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.half()
    model.eval()
    return model


def setup_global_config():
    global model
    global device
    global clip_model
    global preprocess
    global text_features

    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    text = tokenizer.tokenize(["NSFW", "adult content", "porn", "naked people","genitalia","penis","vagina"])
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    config = OmegaConf.load("/home/ubuntu/yha/LatentDiffusion/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml") 
    model = load_model_from_config(config, "/home/ubuntu/yha/LatentDiffusion/models/latent_diffusion_txt2img_f8_large.ckpt")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


def run(opt):
    torch.cuda.empty_cache()
    gc.collect()
    if opt.plms:
        opt.ddim_eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with model.ema_scope():
                uc = None
                if opt.scale > 0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.H//8, opt.W//8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image_vector = Image.fromarray(x_sample.astype(np.uint8))
                        image = preprocess(image_vector).unsqueeze(0)
                        image_features = clip_model.encode_image(image)
                        sims = image_features @ text_features.T
                        
                        if(sims.max()<nsfw_scale):
                            image_vector.save(os.path.join(sample_path, f"{base_count:04}.png"))
                        else:
                            raise Exception('Potential NSFW content was detected on your outputs. Try again with different prompts. If you feel your prompt was not supposed to give NSFW outputs, this may be due to a bias in the model')
                        base_count += 1
                    all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))
    st.image(Image.fromarray(grid.astype(np.uint8)))

def main():
    import argparse
    
    global nsfw_scale

    st.markdown(
        '<h1 style="text-align: center; color: white;">Latent Diffusion</h1>',
        unsafe_allow_html=True,
    )
    Prompt = st.text_input("What do you want to draw?")
    Steps = st.sidebar.number_input("Steps", value=50)
    ETA = st.sidebar.number_input("ETA", value=0)
    Iterations = st.sidebar.number_input("Iterations", value=2)
    Width = st.sidebar.number_input("Width", value=256) #@param{type:"integer"}
    Height = st.sidebar.number_input("Height", value=256) #@param{type:"integer"}
    Samples_in_parallel = st.sidebar.number_input("Samples_in_parallel", value=3)#@param{type:"integer"}
    Diversity_scale = st.sidebar.number_input("Diversity_scale", value=5.0) #@param {type:"number"}
    PLMS_sampling = st.sidebar.selectbox("PLMS_sampling", options=[False, True]) #@param {type:"boolean"}
    nsfw_scale = st.sidebar.number_input("nsfw_scale", value=18)

    args = argparse.Namespace(
        prompt = Prompt, 
        outdir=f'/home/ubuntu/yha/LatentDiffusion/outputs/{generate_id()}',
        ddim_steps = Steps,
        ddim_eta = ETA,
        n_iter = Iterations,
        W=Width,
        H=Height,
        n_samples=Samples_in_parallel,
        scale=Diversity_scale,
        plms=PLMS_sampling
    )
    lcol, mid, rcol = st.columns(3)
    RUN = lcol.button("RUN")
    if RUN:
        setup_global_config()
        run(args)

if __name__ == "__main__":
    
    main()
    