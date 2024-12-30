import os
import re
from typing import List

import numpy as np
import torch
from torch.cuda import nvtx
from polygraphy.logger import G_LOGGER
#import gradio as gr

#from modules import script_callbacks, sd_unet, devices, scripts, shared

#import ui_trt
from utilities import Engine
#from model_manager import TRT_MODEL_DIR, modelmanager
#from datastructures import ModelType
#from scripts.lora import apply_loras

G_LOGGER.module_severity = G_LOGGER.ERROR



class TrtUnet():
    def __init__(self, model_name: str, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stream = None
        self.model_name = model_name
        self.configs = None

        self.profile_idx = 0
        self.model_path = None

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
            "timesteps": timesteps.float(),
            "encoder_hidden_states": context.float(),
        }
        if "y" in kwargs:
            feed_dict["y"] = kwargs["y"].float()

        tmp = torch.empty(
            self.engine_vram_req, dtype=torch.uint8, device=devices.device
        )
        self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["latent"]

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
        print(self.engine)
        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(True)

    def deactivate(self):
        del self.engine