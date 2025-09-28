import tensorrt as trt
import os

def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16_mode=True, max_workspace_size=2048):
    """
    使用Python API将ONNX模型转换为TensorRT引擎
    
    Args:
        onnx_path: ONNX模型文件路径
        engine_path: 输出的TensorRT引擎路径
        fp16_mode: 是否启用FP16精度
        max_workspace_size: 最大工作空间大小(MB)
    """
    # 创建日志记录器
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 创建构建器
    builder = trt.Builder(logger)
    
    # 创建网络定义（必须启用EXPLICIT_BATCH以支持动态形状）
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建ONNX解析器
    parser = trt.OnnxParser(network, logger)
    
    # 加载并解析ONNX模型
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("ONNX模型解析成功！")
    
    # 检查网络输入，确定实际的输入张量名称和形状
    print("网络输入信息：")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"输入 {i}: 名称='{input_tensor.name}', 形状={input_tensor.shape}")
    
    # 创建构建配置
    config = builder.create_builder_config()
    
    # 设置工作空间大小
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size * (1 << 20))
    
    # 创建优化配置文件
    profile = builder.create_optimization_profile()
    
    # 关键修改：根据实际模型输入配置动态形状
    # 假设您的FM模型有两个输入：feature_indices和feature_values
    # 需要为每个输入分别设置动态形状范围
    
    # 获取实际的输入名称（而不是硬编码）
    input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    print(f"检测到的输入名称: {input_names}")
    
    # 为每个输入配置动态形状范围
    for input_name in input_names:
        # 获取输入的原始形状信息
        input_tensor = None
        for i in range(network.num_inputs):
            if network.get_input(i).name == input_name:
                input_tensor = network.get_input(i)
                break
        
        if input_tensor is None:
            print(f"警告: 未找到输入张量 '{input_name}'")
            continue
            
        original_shape = input_tensor.shape
        print(f"配置输入 '{input_name}' 的动态形状，原始形状: {original_shape}")
        
        # 根据输入类型设置不同的动态形状范围
        if "indices" in input_name.lower():
            # 特征索引输入：通常是整数类型，形状为 [batch_size, sequence_length]
            min_shape = (1, 5)      # 最小批次=1，最小序列长度=5
            opt_shape = (32, 10)    # 最优批次=32，最优序列长度=10
            max_shape = (64, 500)   # 最大批次=64，最大序列长度=500
        elif "values" in input_name.lower():
            # 特征值输入：通常是浮点类型，形状与indices相同
            min_shape = (1, 5)      # 与indices保持相同的动态维度
            opt_shape = (32, 10)
            max_shape = (64, 500)
        else:
            # 默认配置（适用于其他输入）
            min_shape = (1, 5)
            opt_shape = (32, 10)
            max_shape = (64, 500)
        
        # 设置动态形状范围
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        print(f"  设置动态形状: min={min_shape}, opt={opt_shape}, max={max_shape}")
    
    # 将优化配置文件添加到构建配置中
    config.add_optimization_profile(profile)
    
    # 启用FP16精度（如果支持）
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("已启用FP16精度模式")
    else:
        print("使用FP32精度模式")
    
    # 构建引擎
    print("开始构建TensorRT引擎（这可能需要几分钟...）")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build TensorRT engine")
        return False
    
    # 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"✅ TensorRT引擎已成功保存至: {engine_path}")
    
    # 输出引擎信息
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    print("引擎构建完成！详细信息：")
    print(f"  输入数量: {engine.num_bindings}")
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        print(f"  {'输入' if is_input else '输出'} {i}: {name}, 类型={dtype}, 形状={shape}")
    
    return True

# 使用示例
if __name__ == "__main__":
    success = convert_onnx_to_tensorrt("fm_model.onnx", "fm_model.engine", fp16_mode=True)
    if success:
        print("🎉 模型转换成功！")
    else:
        print("❌ 模型转换失败")