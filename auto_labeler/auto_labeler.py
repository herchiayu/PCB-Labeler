#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCB 自動標記程式 - 優化版
功能：
1. 訓練 YOLOv8 模型
2. 自動標記 PCB 圖片並產生 YOLO 格式的 txt 檔案
3. 簡潔的檔案管理，只保留必要檔案
 
目錄結構：
auto_labeler/
├── auto_labeler.py
├── weights/
│   └── yolov8_pcb_best.pt
└── data/
    ├── original/       # 原始圖片
    ├── labeled/        # 人工標記
    │   └── classes.txt
    └── auto_labeled/   # 自動標記結果
"""
 
import cv2
import os
import numpy as np
import shutil
import yaml
import random
import time
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
 
# ==================== 配置參數 ====================
# 訓練參數
TRAINING_CONFIG = {
    'EPOCHS': 100,
    'BATCH_SIZE': 16,
    'IMAGE_SIZE': 640,
    'VALIDATION_SPLIT': 0.2,
    'DEVICE': "cpu",
    'MODEL_SIZE': "yolov8n.pt",
    'LEARNING_RATE': 0.01,
    'WARMUP_EPOCHS': 3,
    'PATIENCE': 50,
    'SAVE_PERIOD': 10,
    'AUGMENT': True,
    'MOSAIC': 1.0,
    'MIXUP': 0.0,
    'SAVE_PLOTS': False,
    'VERBOSE': False
}
 
# 檢測參數
CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.4
 
# 執行選項
EXECUTION_CONFIG = {
    'SAVE_WEIGHTS': True,       # 儲存權重檔案
    'SAVE_AUTO_LABELS': True,   # 儲存自動標記txt檔案
    'CLEANUP_TEMP': True,       # 清理暫存檔案
}
 
# ==================== 路徑配置 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"
 
# 子目錄
ORIGINAL_DIR = DATA_DIR / "image"
LABELED_DIR = DATA_DIR / "label"
AUTO_LABELED_DIR = DATA_DIR / "auto_labeled"
 
# 檔案路徑
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
CLASSES_PATH = LABELED_DIR / "classes.txt"
PRETRAINED_MODEL_PATH = WEIGHTS_DIR / TRAINING_CONFIG['MODEL_SIZE']
 
# ==================== 工具函數 ====================
def get_device():
    """自動檢測可用的設備"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"🎯 檢測到 {device_count} 個CUDA設備")
            return "0"  # 使用第一個GPU
        else:
            print("🎯 未檢測到CUDA設備，使用CPU")
            return "cpu"
    except ImportError:
        print("🎯 PyTorch未安裝，使用CPU")
        return "cpu"
 
def check_ultralytics():
    """檢查並安裝 ultralytics"""
    try:
        import ultralytics
        return True
    except ImportError:
        print("⚠️  正在安裝 ultralytics...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            print("❌ ultralytics 安裝失敗")
            return False
 
def setup_directories():
    """建立必要的目錄結構"""
    print("📂 檢查目錄結構...")
   
    directories = [ORIGINAL_DIR, LABELED_DIR, AUTO_LABELED_DIR, WEIGHTS_DIR]
   
    for directory in directories:
        if directory.exists():
            print(f"   ✅ {directory.name}: {directory}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   📁 建立: {directory}")
 
def load_classes():
    """載入類別檔案"""
    if not CLASSES_PATH.exists():
        print(f"❌ 找不到類別檔案: {CLASSES_PATH}")
        return None
   
    try:
        with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"📋 載入 {len(classes)} 個類別")
        return classes
    except Exception as e:
        print(f"❌ 載入類別檔案失敗: {e}")
        return None
 
def safe_remove_directory(dir_path):
    """安全刪除目錄"""
    if not dir_path.exists():
        return True
    try:
        import stat
        for root, dirs, files in os.walk(dir_path):
            for d in dirs:
                os.chmod(os.path.join(root, d), stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            for f in files:
                os.chmod(os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(dir_path)
        return True
    except Exception as e:
        print(f"   ⚠️ 無法刪除目錄 {dir_path}: {e}")
        return False
 
# ==================== 訓練功能 ====================
def download_pretrained_model():
    """下載預訓練模型"""
    print(f"📥 檢查預訓練模型: {TRAINING_CONFIG['MODEL_SIZE']}")
   
    if PRETRAINED_MODEL_PATH.exists():
        print(f"   ✅ 預訓練模型已存在")
        return str(PRETRAINED_MODEL_PATH)
   
    try:
        from ultralytics import YOLO
        print(f"   📥 下載 {TRAINING_CONFIG['MODEL_SIZE']}...")
       
        # 下載模型
        temp_model = YOLO(TRAINING_CONFIG['MODEL_SIZE'])
       
        # 複製到指定位置
        if Path(TRAINING_CONFIG['MODEL_SIZE']).exists():
            shutil.copy2(TRAINING_CONFIG['MODEL_SIZE'], PRETRAINED_MODEL_PATH)
            print(f"   ✅ 模型已儲存到: {PRETRAINED_MODEL_PATH}")
            return str(PRETRAINED_MODEL_PATH)
        else:
            print(f"   ⚠️ 使用線上版本")
            return TRAINING_CONFIG['MODEL_SIZE']
    except Exception as e:
        print(f"   ⚠️ 下載失敗: {e}，使用線上版本")
        return TRAINING_CONFIG['MODEL_SIZE']
 
def prepare_training_data(classes):
    """準備訓練資料集"""
    print("\n📦 準備訓練資料...")
   
    # 使用臨時目錄
    temp_dir = Path(tempfile.gettempdir()) / f"pcb_training_{int(time.time())}"
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
   
    for split in ['train', 'val']:
        (temp_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (temp_dir / split / "labels").mkdir(parents=True, exist_ok=True)
   
    # 找到標記檔案
    label_files = [f for f in LABELED_DIR.glob("*.txt") if f.name != "classes.txt"]
    if not label_files:
        print("❌ 找不到標記檔案")
        return None
   
    # 檢查對應圖片
    valid_pairs = []
    for label_file in label_files:
        for ext in ['.jpg', '.png', '.jpeg']:
            image_path = ORIGINAL_DIR / (label_file.stem + ext)
            if image_path.exists():
                valid_pairs.append((image_path, label_file))
                break
   
    if len(valid_pairs) < 2:
        print("❌ 訓練資料不足")
        return None
   
    print(f"   📊 找到 {len(valid_pairs)} 組資料")
   
    # 分割資料
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * (1 - TRAINING_CONFIG['VALIDATION_SPLIT']))
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
   
    print(f"   訓練: {len(train_pairs)}, 驗證: {len(val_pairs)}")
   
    # 複製檔案
    for image_path, label_path in train_pairs:
        shutil.copy2(image_path, train_dir / "images" / image_path.name)
        shutil.copy2(label_path, train_dir / "labels" / label_path.name)
   
    for image_path, label_path in val_pairs:
        shutil.copy2(image_path, val_dir / "images" / image_path.name)
        shutil.copy2(label_path, val_dir / "labels" / label_path.name)
   
    # 建立配置檔案
    dataset_config = {
        'train': str(train_dir / "images"),
        'val': str(val_dir / "images"),
        'nc': len(classes),
        'names': classes
    }
   
    config_path = temp_dir / "dataset.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
   
    print(f"   ✅ 資料準備完成")
    return config_path, temp_dir
 
def train_model(dataset_config, temp_dir):
    """訓練 YOLO 模型"""
    print(f"\n🚀 開始訓練...")
   
    try:
        from ultralytics import YOLO
       
        # 準備模型
        model_path = download_pretrained_model()
        model = YOLO(model_path)
       
        # 自動檢測設備
        device = get_device()
        print(f"🎯 使用設備: {device}")
       
        # 訓練輸出目錄
        output_dir = Path(tempfile.gettempdir()) / f"pcb_output_{int(time.time())}"
       
        # 開始訓練
        results = model.train(
            data=str(dataset_config),
            epochs=TRAINING_CONFIG['EPOCHS'],
            imgsz=TRAINING_CONFIG['IMAGE_SIZE'],
            batch=TRAINING_CONFIG['BATCH_SIZE'],
            device=device,
            lr0=TRAINING_CONFIG['LEARNING_RATE'],
            warmup_epochs=TRAINING_CONFIG['WARMUP_EPOCHS'],
            patience=TRAINING_CONFIG['PATIENCE'],
            save_period=TRAINING_CONFIG['SAVE_PERIOD'],
            augment=TRAINING_CONFIG['AUGMENT'],
            mosaic=TRAINING_CONFIG['MOSAIC'],
            mixup=TRAINING_CONFIG['MIXUP'],
            project=str(output_dir),
            name='training',
            exist_ok=True,
            save=True,
            plots=TRAINING_CONFIG['SAVE_PLOTS'],
            verbose=TRAINING_CONFIG['VERBOSE']
        )
       
        print("✅ 訓練完成!")
        return output_dir / "training"
       
    except Exception as e:
        print(f"❌ 訓練失敗: {e}")
        return None
 
def save_weights(train_output_dir):
    """儲存訓練權重"""
    print(f"\n💾 儲存權重...")
   
    if not train_output_dir or not train_output_dir.exists():
        return False
   
    if EXECUTION_CONFIG['SAVE_WEIGHTS']:
        WEIGHTS_DIR.mkdir(exist_ok=True)
       
        best_weight = train_output_dir / "weights" / "best.pt"
        if best_weight.exists():
            shutil.copy2(best_weight, WEIGHTS_PATH)
            print(f"   ✅ 權重已儲存: {WEIGHTS_PATH.name}")
            return True
        else:
            print("   ❌ 找不到最佳權重")
            return False
    else:
        print("   ⏭️ 跳過權重儲存")
        return False
 
def cleanup_temp_files(*temp_dirs):
    """清理暫存檔案"""
    if EXECUTION_CONFIG['CLEANUP_TEMP']:
        print("   🧹 清理暫存檔案...")
        for temp_dir in temp_dirs:
            if temp_dir and temp_dir.exists():
                safe_remove_directory(temp_dir)
 
# ==================== 自動標記功能 ====================
def initialize_model():
    """初始化 YOLO 模型"""
    print("\n🧠 載入模型...")
   
    if not WEIGHTS_PATH.exists():
        print(f"❌ 權重檔案不存在: {WEIGHTS_PATH}")
        return None
   
    try:
        from ultralytics import YOLO
        model = YOLO(str(WEIGHTS_PATH))
        model.conf = CONFIDENCE_THRESHOLD
        model.iou = NMS_THRESHOLD
        print("   ✅ 模型載入成功")
        return model
    except Exception as e:
        print(f"   ❌ 模型載入失敗: {e}")
        return None
 
def auto_label_images(model, classes):
    """執行自動標記"""
    print("\n🔍 開始自動標記...")
   
    # 掃描圖片
    image_files = [f for f in ORIGINAL_DIR.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_files:
        print("❌ 找不到圖片檔案")
        return
   
    print(f"   📊 找到 {len(image_files)} 張圖片")
   
    processed = 0
    skipped = 0
   
    for i, image_path in enumerate(image_files, 1):
        print(f"📷 處理 [{i}/{len(image_files)}]: {image_path.name}")
       
        # 檢查是否已標記
        labeled_txt = LABELED_DIR / (image_path.stem + '.txt')
        auto_labeled_txt = AUTO_LABELED_DIR / (image_path.stem + '.txt')
       
        if labeled_txt.exists():
            print(f"   ⏭️ 跳過 - 已有人工標記")
            skipped += 1
            continue
           
        if EXECUTION_CONFIG['SAVE_AUTO_LABELS'] and auto_labeled_txt.exists():
            print(f"   ⏭️ 跳過 - 已有自動標記")
            skipped += 1
            continue
       
        # 讀取圖片
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"   ❌ 無法讀取圖片")
            continue
       
        # 進行偵測
        try:
            results = model.predict(img, verbose=False)
            detections = []
           
            if len(results) > 0 and results[0].boxes is not None:
                boxes_data = results[0].boxes
                for j in range(len(boxes_data)):
                    box = boxes_data.xyxy[j].cpu().numpy()
                    conf = float(boxes_data.conf[j].cpu())
                    cls = int(boxes_data.cls[j].cpu())
                   
                    if conf >= CONFIDENCE_THRESHOLD and cls < len(classes):
                        # 轉換為 YOLO 格式
                        img_h, img_w = img.shape[:2]
                        x_center = (box[0] + box[2]) / 2 / img_w
                        y_center = (box[1] + box[3]) / 2 / img_h
                        width = (box[2] - box[0]) / img_w
                        height = (box[3] - box[1]) / img_h
                       
                        detections.append(f'{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')
                        print(f"      偵測到: {classes[cls]} (信心度: {conf:.3f})")
           
            # 儲存結果
            if EXECUTION_CONFIG['SAVE_AUTO_LABELS']:
                with open(auto_labeled_txt, 'w') as f:
                    for detection in detections:
                        f.write(detection + '\n')
                print(f"   ✅ 儲存 {len(detections)} 個標記")
            else:
                print(f"   📝 偵測到 {len(detections)} 個物件（未儲存）")
           
            processed += 1
           
        except Exception as e:
            print(f"   ❌ 處理失敗: {e}")
   
    # 顯示結果
    print(f"\n📊 標記完成:")
    print(f"   處理: {processed} 張")
    print(f"   跳過: {skipped} 張")
    if EXECUTION_CONFIG['SAVE_AUTO_LABELS']:
        print(f"   輸出: {AUTO_LABELED_DIR}")
 
# ==================== 設定功能 ====================
def show_settings():
    """顯示設定選單"""
    print("\n⚙️ 檔案儲存設定:")
    print("=" * 30)
   
    settings = [
        ("權重檔案", 'SAVE_WEIGHTS'),
        ("自動標記txt", 'SAVE_AUTO_LABELS'),
        ("暫存清理", 'CLEANUP_TEMP')
    ]
   
    for i, (name, key) in enumerate(settings, 1):
        status = "✅ 儲存" if EXECUTION_CONFIG[key] else "❌ 不儲存"
        print(f"   {i}. {name}: {status}")
   
    print("   4. 🔙 返回")
   
    while True:
        choice = input("\n選擇 (1-4): ").strip()
       
        if choice in ['1', '2', '3']:
            key = settings[int(choice)-1][1]
            EXECUTION_CONFIG[key] = not EXECUTION_CONFIG[key]
            status = "儲存" if EXECUTION_CONFIG[key] else "不儲存"
            print(f"   ✅ {settings[int(choice)-1][0]} 設為 {status}")
        elif choice == '4':
            break
        else:
            print("   ❌ 無效選項")
 
def show_statistics():
    """顯示資料統計"""
    print("\n📊 資料統計:")
   
    # 統計檔案數量
    original_count = len(list(ORIGINAL_DIR.glob("*.jpg")) + list(ORIGINAL_DIR.glob("*.png")))
    labeled_count = len([f for f in LABELED_DIR.glob("*.txt") if f.name != "classes.txt"])
    auto_labeled_count = len(list(AUTO_LABELED_DIR.glob("*.txt"))) if AUTO_LABELED_DIR.exists() else 0
   
    print(f"📸 原始圖片: {original_count} 張")
    print(f"🏷️ 人工標記: {labeled_count} 個")
    print(f"🤖 自動標記: {auto_labeled_count} 個")
    print(f"⚖️ 權重檔案: {'✅ 存在' if WEIGHTS_PATH.exists() else '❌ 不存在'}")
   
    if original_count > 0:
        coverage = ((labeled_count + auto_labeled_count) / original_count) * 100
        print(f"📈 標記覆蓋率: {coverage:.1f}%")
 
# ==================== 主程式 ====================
def show_menu():
    """顯示主選單"""
    print("\n" + "="*40)
    print("🔧 PCB 自動標記程式")
    print("="*40)
    print("1. 🚀 訓練模型")
    print("2. 🤖 自動標記")
    print("3. 🔄 訓練並標記")
    print("4. 📊 資料統計")
    print("5. ⚙️ 設定")
    print("6. 🚪 退出")
   
    return input("\n請選擇 (1-6): ").strip()
 
def main():
    """主程式"""
    print("🔧 PCB 自動標記程式 - 優化版")
    print("="*50)
    print("💡 設定: 只保留必要檔案")
    print("   ✅ 權重檔案 (供推論使用)")
    print("   ✅ 自動標記txt檔案")
    print("   ❌ 不產生圖片、圖表等額外檔案")
    print("="*50)
   
    # 初始化
    setup_directories()
    classes = load_classes()
    if not classes:
        print("❌ 無法載入類別檔案")
        return
   
    # 主迴圈
    while True:
        choice = show_menu()
       
        if choice == "1":
            # 訓練模型
            if not check_ultralytics():
                continue
           
            result = prepare_training_data(classes)
            if result:
                dataset_config, temp_dir = result
                train_output = train_model(dataset_config, temp_dir)
                if train_output:
                    save_weights(train_output)
                cleanup_temp_files(temp_dir, train_output.parent if train_output else None)
               
        elif choice == "2":
            # 自動標記
            if not WEIGHTS_PATH.exists():
                print("❌ 請先訓練模型")
                continue
           
            if not check_ultralytics():
                continue
           
            model = initialize_model()
            if model:
                auto_label_images(model, classes)
               
        elif choice == "3":
            # 訓練並標記
            if not check_ultralytics():
                continue
           
            # 訓練
            result = prepare_training_data(classes)
            if result:
                dataset_config, temp_dir = result
                train_output = train_model(dataset_config, temp_dir)
                if train_output and save_weights(train_output):
                    # 標記
                    model = initialize_model()
                    if model:
                        auto_label_images(model, classes)
                cleanup_temp_files(temp_dir, train_output.parent if train_output else None)
               
        elif choice == "4":
            # 統計
            show_statistics()
           
        elif choice == "5":
            # 設定
            show_settings()
           
        elif choice == "6":
            # 退出
            print("👋 再見！")
            break
           
        else:
            print("❌ 無效選項")
 
if __name__ == "__main__":
    main()
 