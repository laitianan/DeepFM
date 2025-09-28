# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 11:02:25 2025

@author: admin
"""

# tensorrt_inference.py
import tensorrt as trt
import numpy as np
import os
import time
from typing import Dict, List
from PIL import Image


class TensorRTInference:
    def __init__(self, engine_path):
        print(f'Loading engine from: {engine_path}')
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # 获取输入输出绑定信息
        self.input_bindings = []
        self.output_bindings = []
        self.binding_names = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            is_input = self.engine.binding_is_input(i)

            print(f"Binding {i}: {'Input' if is_input else 'Output'}, Name={name}, Dtype={dtype}, Shape={shape}")

            self.binding_names.append(name)
            if is_input:
                self.input_bindings.append(i)
            else:
                self.output_bindings.append(i)

        # 初始化缓冲区
        self.input_buffers = {}
        self.output_buffers = {}
        self.current_shapes = {}

    def load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path), f"Engine file not found: {engine_file_path}"
        print(f"Reading engine from file: {engine_file_path}")

        # 创建记录器
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_file_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    def __call__(self, inputs: Dict[str, np.ndarray], time_buffer=None):
        # 准备输入
        input_arrays = []
        bindings = []

        # 处理输入
        for binding_idx in self.input_bindings:
            name = self.engine.get_binding_name(binding_idx)
            input_data = inputs[name]

            # 检查数据类型
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))

            if input_data.dtype != dtype:
                input_data = input_data.astype(dtype)

            # 检查形状变化
            if name not in self.current_shapes or self.current_shapes[name] != input_data.shape:
                self.current_shapes[name] = input_data.shape
                self.context.set_binding_shape(binding_idx, input_data.shape)

            # 确保内存连续
            input_data = np.ascontiguousarray(input_data)
            input_arrays.append(input_data)
            bindings.append(input_data.ctypes.data)

        # 准备输出
        outputs = {}
        for binding_idx in self.output_bindings:
            name = self.engine.get_binding_name(binding_idx)
            shape = self.context.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))

            # 创建输出缓冲区
            output_buffer = np.empty(shape, dtype=dtype)
            outputs[name] = output_buffer
            bindings.append(output_buffer.ctypes.data)

        # 执行推理
        start_time = time.time()
        self.context.execute_v2(bindings=bindings)
        inference_time = time.time() - start_time

        if time_buffer is not None:
            time_buffer.append(inference_time)

        return outputs


# 使用示例
if __name__ == "__main__":
    
    
   
    # 创建模型实例 (使用FP16引擎)
    model = TensorRTInference("fm_model.engine")
   
    batch_size = 2
    sequence_length = 10
    
    feature_indices = np.array([[179, 421, 345,  78, 386, 272,  80,  39,  45,  24],
           [ 44, 476, 112, 377, 462, 215, 453, 185, 122,  58]])
    
    # 特征值（浮点数，通常为1.0或归一化值）
    feature_values = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    
    print(f"输入数据形状 - 索引: {feature_indices.shape}, 值: {feature_values.shape}")
    
    # 3. 执行推理
    print("执行推理...")

    data={"feature_indices":feature_indices,"feature_values":feature_values}
    
    inference_times = []

    outputs = model(data, time_buffer=inference_times)
    print(outputs)
