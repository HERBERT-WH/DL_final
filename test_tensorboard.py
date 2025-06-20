import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import time

# 模拟NeRF网络结构
class TestNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=4, skips=[4], input_ch_views=3, use_viewdirs=False):
        super(TestNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # 位置编码的全连接层
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # 视角相关的层
        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

def test_tensorboard():
    """
    测试TensorBoard的各种功能
    """
    print("开始TensorBoard测试...")
    
    # 创建日志目录
    log_dir = './logs/tensorboard_test'
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # 创建测试模型
    model = TestNeRF(D=4, W=128, input_ch=63, output_ch=4, input_ch_views=27, use_viewdirs=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    print("模型创建成功，开始测试...")
    
    # 模拟训练数据
    batch_size = 1024
    input_dim = 63 + 27  # input_ch + input_ch_views
    
    try:
        # 测试1: 基本标量记录
        print("测试1: 标量记录...")
        for i in range(10):
            loss = np.random.random() * 0.1
            psnr = 20 + np.random.random() * 10
            lr = 5e-4 * (0.1 ** (i / 1000))
            
            writer.add_scalar('Loss/train', loss, i)
            writer.add_scalar('PSNR/train', psnr, i)
            writer.add_scalar('Learning_rate', lr, i)
        print("✓ 标量记录测试通过")
        
        # 测试2: 文本记录
        print("测试2: 文本记录...")
        writer.add_text('Model', str(model))
        writer.add_text('Config', 'Test configuration for NeRF model')
        print("✓ 文本记录测试通过")
        
        # 测试3: 图像记录
        print("测试3: 图像记录...")
        # 模拟渲染结果
        fake_rgb = torch.rand(400, 400, 3)  # H, W, C
        fake_disp = torch.rand(400, 400)    # H, W
        fake_acc = torch.rand(400, 400)     # H, W
        
        writer.add_image('Render/rgb', fake_rgb.permute(2,0,1), 0)  # C, H, W
        writer.add_image('Render/disp', fake_disp.unsqueeze(0), 0)  # 1, H, W
        writer.add_image('Render/acc', fake_acc.unsqueeze(0), 0)    # 1, H, W
        print("✓ 图像记录测试通过")
        
        # 测试4: 参数分布记录（正常情况）
        print("测试4: 参数分布记录（正常情况）...")
        # 进行一次前向传播和反向传播以生成梯度
        x = torch.randn(batch_size, input_dim)
        target = torch.randn(batch_size, 4)
        
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # 记录参数分布
        for name, param in model.named_parameters():
            if param.data.numel() > 0 and not torch.isnan(param.data).any() and not torch.isinf(param.data).any():
                writer.add_histogram(f'Parameters/{name}', param.data, 0)
            
            if param.grad is not None and param.grad.numel() > 0:
                if not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
                    writer.add_histogram(f'Gradients/{name}', param.grad.data, 0)
        print("✓ 参数分布记录测试通过")
        
        # 测试5: 异常情况处理
        print("测试5: 异常情况处理...")
        
        # 创建包含异常值的参数
        test_params = {
            'normal_param': torch.randn(10, 10),
            'empty_param': torch.tensor([]),
            'nan_param': torch.full((5, 5), float('nan')),
            'inf_param': torch.full((5, 5), float('inf')),
            'zero_param': torch.zeros(3, 3)
        }
        
        for name, param in test_params.items():
            try:
                # 使用安全检查
                if param.numel() > 0 and not torch.isnan(param).any() and not torch.isinf(param).any():
                    writer.add_histogram(f'Test/{name}', param, 0)
                    print(f"  ✓ {name}: 成功记录")
                else:
                    print(f"  - {name}: 跳过（包含无效值或为空）")
            except Exception as e:
                print(f"  ✗ {name}: 错误 - {e}")
        
        print("✓ 异常情况处理测试通过")
        
        # 测试6: 网络图记录
        print("测试6: 网络图记录...")
        try:
            dummy_input = torch.randn(1, input_dim)
            writer.add_graph(model, dummy_input)
            print("✓ 网络图记录测试通过")
        except Exception as e:
            print(f"- 网络图记录跳过: {e}")
        
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        return False
    
    finally:
        # 关闭writer
        writer.close()
    
    print(f"\n测试完成！TensorBoard日志已保存到: {log_dir}")
    print("可以使用以下命令查看结果:")
    print(f"tensorboard --logdir {log_dir}")
    
    return True

def test_original_error_case():
    """
    专门测试原始错误情况
    """
    print("\n测试原始错误情况...")
    
    log_dir = './logs/error_test'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    try:
        # 模拟可能导致错误的情况
        print("创建可能导致错误的参数...")
        
        # 情况1: 空张量
        empty_tensor = torch.tensor([])
        print(f"空张量大小: {empty_tensor.numel()}")
        
        # 情况2: 包含NaN的张量
        nan_tensor = torch.tensor([1.0, 2.0, float('nan'), 4.0])
        print(f"NaN张量: {nan_tensor}")
        
        # 情况3: 包含inf的张量
        inf_tensor = torch.tensor([1.0, 2.0, float('inf'), 4.0])
        print(f"Inf张量: {inf_tensor}")
        
        # 不安全的方式（会导致错误）
        print("测试不安全的记录方式...")
        try:
            writer.add_histogram('Unsafe/empty', empty_tensor, 0)
            print("✗ 空张量记录应该失败但没有失败")
        except Exception as e:
            print(f"✓ 空张量记录正确失败: {e}")
        
        try:
            writer.add_histogram('Unsafe/nan', nan_tensor, 0)
            print("✓ NaN张量记录成功（TensorBoard可能会处理）")
        except Exception as e:
            print(f"- NaN张量记录失败: {e}")
        
        # 安全的方式
        print("测试安全的记录方式...")
        test_tensors = {
            'empty': empty_tensor,
            'nan': nan_tensor,
            'inf': inf_tensor,
            'normal': torch.randn(10)
        }
        
        for name, tensor in test_tensors.items():
            if tensor.numel() > 0 and not torch.isnan(tensor).any() and not torch.isinf(tensor).any():
                writer.add_histogram(f'Safe/{name}', tensor, 0)
                print(f"✓ {name}: 安全记录成功")
            else:
                print(f"- {name}: 跳过无效张量")
        
    except Exception as e:
        print(f"✗ 错误测试过程中出现异常: {e}")
    
    finally:
        writer.close()
    
    print("错误情况测试完成")

if __name__ == '__main__':
    print("=" * 50)
    print("TensorBoard 功能测试")
    print("=" * 50)
    
    # 主要功能测试
    success = test_tensorboard()
    
    # 错误情况测试
    test_original_error_case()
    
    if success:
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("请运行以下命令查看TensorBoard结果:")
        print("tensorboard --logdir ./logs")
        print("然后在浏览器中打开 http://localhost:6006")
        print("=" * 50)
    else:
        print("\n测试失败，请检查错误信息") 