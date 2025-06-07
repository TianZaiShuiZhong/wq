import argparse
import json
import os
from src.process import init_detector, process_img

def process_single_image(detector, image_path, output_path):
    """处理单张图片"""
    detections = process_img(image_path)
    
    # 保存结果
    result = {os.path.basename(image_path): detections}
    with open(output_path.replace('.jpg', '.txt'), 'w') as f:
        json.dump(result, f, indent=2)

    # 可视化结果
    detector.visualize(image_path, detections, output_path)

def process_folder(detector, input_folder, output_folder, confidence):
    """处理整个文件夹"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    results = {}
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"result_{filename}")
            
            detections = process_img(image_path)
            results[filename] = detections
            
            # 可视化结果
            detector.visualize(image_path, detections, output_path)
    
    # 保存所有结果到一个TXT文件
    output_txt = os.path.join(output_folder, 'detections.txt')
    with open(output_txt, 'w') as f:
        # 生成txt输出
        # 按照大小排序
        f.write('{\n')
        for i, (filename, detections) in enumerate(results.items()):
            f.write(f'    "{filename}": [')
            for j, det in enumerate(detections):
                f.write('{' + 
                    f'"x": {det["x"]}, "y": {det["y"]}, ' +
                    f'"w": {det["w"]}, "h": {det["h"]}, ' +
                    # f'"Confidence": {det["confidence"]}' + 
                    '}')
                if j < len(detections)-1:
                    f.write(', ')
            f.write(']')
            if i < len(results)-1:
                f.write(',')
            f.write('\n')
        f.write('}\n')

def main():
    parser = argparse.ArgumentParser(description='网球检测')
    parser.add_argument('--image', help='输入图片路径')
    parser.add_argument('--folder', help='输入文件夹路径')
    parser.add_argument('--output', required=True, help='输出路径')
    parser.add_argument('--model', default='src/best.onnx', help='模型路径')
    parser.add_argument('--confidence', type=float, default=0.05, help='检测置信度阈值')
    args = parser.parse_args()

    # 初始化检测器
    detector = init_detector(args.model, confidence=args.confidence)

    if args.image:
        process_single_image(detector, args.image, args.output)
    elif args.folder:
        process_folder(detector, args.folder, args.output, args.confidence)
    else:
        print("请指定 --image 或 --folder 参数")

if __name__ == '__main__':
    main()
