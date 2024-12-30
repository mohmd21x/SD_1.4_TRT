import torch
import tensorrt as trt
import os, sys, argparse 
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from time import time

onnx_model = "./models/unet/1/unet.onnx"
engine_filename = "./unet_trt_v10.engine"
def convert_model():
    batch_size = 1
    height = 512
    width = 512
    latents_shape = (batch_size*2, 4, height // 8, width // 8)
    embed_shape = (batch_size*2,64,768)
    timestep_shape = (batch_size,)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    TRT_BUILDER = trt.Builder(TRT_LOGGER)

    network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    parse_success = onnx_parser.parse_from_file(onnx_model)
    for idx in range(onnx_parser.num_errors):
        print(onnx_parser.get_error(idx))
    if not parse_success:
        sys.exit('ONNX model parsing failed')
    print("Load Onnx model done")

    config = TRT_BUILDER.create_builder_config()
    profile = TRT_BUILDER.create_optimization_profile() 
    profile.set_shape("sample", latents_shape, latents_shape, latents_shape) 
    profile.set_shape("encoder_hidden_states", embed_shape, embed_shape, embed_shape) 
    profile.set_shape("timestep", timestep_shape, timestep_shape, timestep_shape) 
    config.add_optimization_profile(profile)

    config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = TRT_BUILDER.build_serialized_network(network, config)

    ## save TRT engine
    with open(engine_filename, 'wb') as f:
        f.write(serialized_engine)
    print(f'Engine is saved to {engine_filename}')

convert_model()