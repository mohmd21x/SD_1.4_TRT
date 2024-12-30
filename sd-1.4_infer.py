import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers import LMSDiscreteScheduler
from torch import autocast
import argparse
import time
import os
import re
from typing import List
import numpy as np
import torch
from torch.cuda import nvtx
from polygraphy.logger import G_LOGGER
from utilities import Engine

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="Super Mario learning to fly in an airport, Painting by Leonardo Da Vinci",
        help="input prompt",
    )
    parser.add_argument(
        "--trt_unet_save_path",
        default="./unet.engine",
        type=str,
        help="TensorRT unet saved path",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--img_size", default=(512, 512), help="Unet input image size (h,w)"
    )
    parser.add_argument(
        "--max_seq_length", default=64, help="Maximum sequence length of input text"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Running benchmark by average num iteration",
    )
    parser.add_argument(
        "--n_iters", default=50, help="Running benchmark by average num iteration"
    )

    return parser.parse_args()


G_LOGGER.module_severity = G_LOGGER.ERROR

class TrtUnet():
    def __init__(self, model_name: str, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stream = None
        self.model_name = model_name
        self.configs = None

        self.profile_idx = 0
        self.model_path = model_path

        self.engine_vram_req = 0
        self.refitted_keys = set()

        self.engine = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        nvtx.range_push("forward")
        feed_dict = {
            "sample": x.float(),
            "timestep": timesteps.float(),
            "encoder_hidden_states": context.float(),
        }
        if "y" in kwargs:
            feed_dict["y"] = kwargs["y"].float()

        tmp = torch.empty(
            self.engine_vram_req, dtype=torch.uint8, device='cuda'
        )
        self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)['noise_pred']
        print(out.size)
        nvtx.range_pop()
        return out

    def apply_loras(self, refit_dict: dict):
        if not self.refitted_keys.issubset(set(refit_dict.keys())):
            # Need to ensure that weights that have been modified before and are not present anymore are reset.
            self.refitted_keys = set()
            self.switch_engine()

        self.engine.refit_from_dict(refit_dict, is_fp16=True)
        self.refitted_keys = set(refit_dict.keys())

    def switch_engine(self):
        #self.loaded_config = self.configs[self.profile_idx]
        self.engine.reset(self.model_path)
        self.activate()

    def activate(self):
        #self.loaded_config = self.configs[self.profile_idx]
        if self.engine is None:
            self.engine = Engine(
                self.model_path
            )
        self.engine.load()
        print(f"\nLoaded Profile: {self.profile_idx}")
        #print(self.engine)
        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(True)

    def deactivate(self):
        del self.engine






class TrtDiffusionModel:
    def __init__(self, args):
        self.device = torch.device("cuda")
        self.unet = TrtUnet(model_name='unet', model_path='./unet_trt_v10.engine')
        self.unet.activate()
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="vae"
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

    def predict(
        self, prompts, num_inference_steps=50, height=512, width=512, max_seq_length=64
    ):
        guidance_scale = 7.5
        batch_size = 1
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn((batch_size, 4, height // 8, width // 8)).to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * self.scheduler.sigmas[0]
        with torch.inference_mode(), autocast("cuda"):
            for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # predict the noise residual
                print(latent_model_input.shape)
                print(t.shape)
                print(text_embeddings.shape)
                
                inputs = [
                    latent_model_input,
                    torch.tensor([t]).to(self.device),
                    text_embeddings,
                ]
                noise_pred = self.unet.forward(latent_model_input,
                    torch.tensor([t]).to(self.device),
                    text_embeddings,
                                    )
                
                noise_pred = torch.reshape(noise_pred, (batch_size * 2, 4, 64, 64))

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.cuda(), t, latents)[
                    "prev_sample"
                ]

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
        return image


if __name__ == "__main__":
    args = get_args()
    model = TrtDiffusionModel(args)
    if args.benchmark:
        n_iters = args.n_iters
        # warm up
        for i in range(3):
            image = model.predict(
                prompts=args.prompt,
                num_inference_steps=50,
                height=args.img_size[0],
                width=args.img_size[1],
                max_seq_length=args.max_seq_length,
            )
    else:
        n_iters = 1

    start = time.time()
    for i in tqdm(range(n_iters)):
        image = model.predict(
            prompts=args.prompt,
            num_inference_steps=50,
            height=args.img_size[0],
            width=args.img_size[1],
            max_seq_length=args.max_seq_length,
        )
    end = time.time()
    if args.benchmark:
        print("Average inference time is: ", (end - start) / n_iters)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save("image_generated.png")
