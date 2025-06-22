import os
import glob
import cv2
import argparse

def countFile(dir):
    """递归计算文件夹中的文件数量"""
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp

def downsample_images(input_dir, output_dir, factor=8, file_format='jpg'):
    """
    对图像进行下采样处理
    
    Args:
        input_dir (str): 输入图像文件夹路径
        output_dir (str): 输出图像文件夹路径  
        factor (int): 下采样倍数，支持4或8
        file_format (str): 图像格式，如'jpg', 'png'等
    """
    
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在！")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
        image_files.extend(glob.glob(os.path.join(input_dir, f'*.{ext.upper()}')))
    
    if not image_files:
        print(f"警告：在 {input_dir} 中未找到图像文件！")
        return
    
    image_files.sort()  # 按文件名排序
    total_files = len(image_files)
    
    print(f"找到 {total_files} 张图像")
    print(f"下采样倍数: {factor}x")
    print(f"输出目录: {output_dir}")
    
    for i, file_path in enumerate(image_files):
        try:
            # 读取原始图像
            original_image = cv2.imread(file_path)
            if original_image is None:
                print(f"警告：无法读取图像 {file_path}")
                continue
            
            # 下采样处理
            img_downsampled = original_image
            if factor == 4:
                img_downsampled = cv2.pyrDown(original_image)
                img_downsampled = cv2.pyrDown(img_downsampled)
            elif factor == 8:
                img_downsampled = cv2.pyrDown(original_image)
                img_downsampled = cv2.pyrDown(img_downsampled)
                img_downsampled = cv2.pyrDown(img_downsampled)
            else:
                print(f"警告：不支持的下采样倍数 {factor}，跳过处理")
                continue
            
            # 生成输出文件名
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            
            # 保持原有编号格式或重新编号
            if name.isdigit() or ('_' in name and name.split('_')[-1].isdigit()):
                # 保持原有命名
                output_filename = f"{name}.{file_format}"
            else:
                # 重新编号
                output_filename = f"{i+1:03d}.{file_format}"
            
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存下采样后的图像
            cv2.imwrite(output_path, img_downsampled)
            
            print(f"处理完成 ({i+1}/{total_files}): {filename} -> {output_filename}")
            
        except Exception as e:
            print(f"处理图像 {file_path} 时出错: {e}")
    
    print(f"\n下采样完成！处理了 {total_files} 张图像")
    print(f"结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='图像下采样工具')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入图像文件夹路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出图像文件夹路径')
    parser.add_argument('--factor', '-f', type=int, choices=[4, 8], default=8,
                        help='下采样倍数 (4 或 8)')
    parser.add_argument('--format', type=str, default='jpg',
                        help='输出图像格式 (jpg, png等)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("NeRF 图像下采样工具")
    print("=" * 50)
    
    downsample_images(args.input, args.output, args.factor, args.format)

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认示例
    import sys
    if len(sys.argv) == 1:
        print("=" * 50)
        print("NeRF 图像下采样工具 - 示例模式")
        print("=" * 50)
        print("使用示例路径进行演示...")
        
        # 默认示例路径（可以修改这些路径）
        input_dir = "data/nerf_llff_data/llfftest/images"
        output_dir = "data/nerf_llff_data/llfftest/images_8"
        factor = 8
        
        if os.path.exists(input_dir):
            downsample_images(input_dir, output_dir, factor)
        else:
            print(f"示例输入目录 {input_dir} 不存在")
            print("\n请使用命令行参数指定路径：")
            print("python data.py --input your_input_path --output your_output_path --factor 8")
    else:
        main()
