import torch


from FM import FactorizationMachine
device="cpu"
# 示例：仅加载模型参数用于推理（推荐方式）[3](@ref)
model = FactorizationMachine(num_features=500, factor_dim=10).to(device)
model.load_state_dict(torch.load('fm_model_state_dict.pth', map_location=device))
model.eval()  # 设置为评估模式[2](@ref)
print("模型参数加载完成，可用于推理")

def convert_to_onnx(model, onnx_path, num_features=500, max_sequence_length=10):
    """
    将FM模型转换为ONNX格式
    """
    model.eval()
    
    # 创建示例输入（模拟稀疏特征）
    dummy_feature_indices = torch.randint(0, num_features, (1, max_sequence_length), dtype=torch.long)
    dummy_feature_values = torch.ones((1, max_sequence_length), dtype=torch.float32)
    
    # 定义输入和输出的名称
    input_names = ['feature_indices', 'feature_values']
    output_names = ['output']
    
    # 动态轴（支持批量大小和序列长度变化）
    dynamic_axes = {
        'feature_indices': {0: 'batch_size', 1: 'sequence_length'},
        'feature_values': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size'}
    }
    
    # 导出为ONNX
    torch.onnx.export(
        model,
        (dummy_feature_indices, dummy_feature_values),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        verbose=True
    )
    
    print(f"模型已成功导出为ONNX格式: {onnx_path}")

# 转换为ONNX
convert_to_onnx(model, "fm_model.onnx")

# 验证ONNX模型
def verify_onnx_model(onnx_path):
    """验证ONNX模型的有效性"""
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证成功")
        
        # 测试ONNX推理
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path)
        
        # 准备测试输入
        test_indices = torch.randint(0, 500, (2, 5), dtype=torch.long).numpy()
        test_values = torch.ones((2, 5), dtype=torch.float32).numpy()
        
        # ONNX推理
        ort_inputs = {
            'feature_indices': test_indices,
            'feature_values': test_values
        }
        ort_outs = ort_session.run(None, ort_inputs)
        
        print("✓ ONNX推理测试成功")
        print(f"推理结果形状: {ort_outs[0].shape}")
        print(f"示例输出: {ort_outs[0][:2]}")
        
    except Exception as e:
        print(f"ONNX验证失败: {e}")

# 验证导出的模型
verify_onnx_model("fm_model.onnx")