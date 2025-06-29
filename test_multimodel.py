import os
import asyncio
from typing import List
from chunk_processor import extract_image_content, init_vl_model
from PIL import Image
import numpy as np

def sync_test_image_description(image_path: str):
    """同步测试单张图片描述生成"""
    print(f"\n测试图片: {os.path.basename(image_path)}")
    try:
        # 注意：这里直接调用同步函数
        description = extract_image_content(image_path)
        print("生成描述:")
        print(description)
        
        # 显示图片基本信息
        with Image.open(image_path) as img:
            print(f"\n图片信息: 格式={img.format}, 大小={img.size}, 模式={img.mode}")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")

async def test_multiple_images(image_paths: List[str]):
    """测试多张图片批量处理"""
    print(f"\n批量测试 {len(image_paths)} 张图片...")
    for img_path in image_paths:
        sync_test_image_description(img_path)

def test_special_cases():
    """测试特殊场景"""
    print("\n测试特殊场景...")
    
    # 1. 测试不存在的图片
    sync_test_image_description("non_existent.jpg")
    
    # 2. 测试中等大小图片(减小尺寸)
    medium_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    medium_img_path = "medium_image.jpg"
    Image.fromarray(medium_img).save(medium_img_path)
    sync_test_image_description(medium_img_path)
    os.remove(medium_img_path)
    
    # 3. 测试无效图片文件
    with open("invalid.jpg", "wb") as f:
        f.write(b"invalid image data")
    sync_test_image_description("invalid.jpg")
    os.remove("invalid.jpg")

def generate_test_images():
    """生成测试用图片"""
    os.makedirs("test_images", exist_ok=True)
    
    # 1. 简单几何图形
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    img1[50:250, 50:250] = [255, 0, 0]  # 红色方块
    Image.fromarray(img1).save("test_images/red_square.jpg")
    
    # 2. 文字图片
    img2 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(Image.fromarray(img2))
    font = ImageFont.load_default()
    draw.text((50, 80), "多模态模型测试", fill="black", font=font)
    Image.fromarray(img2).save("test_images/text_image.jpg")
    
    # 3. 实际图片测试
    return [
        "test_images/red_square.jpg",
        "test_images/text_image.jpg",
        "/root/ty/output/05_机器人工程设计专项赛/auto/images/0b8dd788911401ab24360c3426e065541bca5b93f4422ab44e3535385a2ff35a.jpg",
        "/root/ty/output/05_机器人工程设计专项赛/auto/images/0e1b9c95298877b775772d158cbaaf7ca882415b1707c91e44d0b82b76070bf6.jpg"
    ]

def main():
    # 初始化模型
    init_vl_model()
    
    # 生成测试图片并获取路径
    test_images = generate_test_images()
    
    # 执行测试
    print("="*50)
    print("开始多模态模型测试")
    print("="*50)
    
    # 测试单张图片
    sync_test_image_description(test_images[0])
    
    # 测试多张图片
    asyncio.run(test_multiple_images(test_images))
    
    # 测试特殊场景
    test_special_cases()
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
