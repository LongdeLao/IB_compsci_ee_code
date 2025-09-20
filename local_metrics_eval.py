#!/usr/bin/env python3
"""
Local Model Metrics Evaluation Script
Evaluates locally downloaded YOLO and DETR models and generates comprehensive metrics
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import torch
from ultralytics import YOLO
import torchvision.transforms as T
from collections import defaultdict
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

class LocalMetricsEvaluator:
    def __init__(self, models_dir: str = "models", data_dir: str = "data", results_dir: str = "metrics_results"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Add detection visualization directory
        self.detection_viz_dir = Path(results_dir) / "detection_visualizations"
        self.detection_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models storage
        self.models = {}
        self.model_stats = {}
        self.all_image_results = []
        
        print("Local Metrics Evaluator initialized")
        print(f"Models directory: {self.models_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Results directory: {self.results_dir}")

    def load_models(self):
        """Load all available local models"""
        print("\n=== Loading Local Models ===")
        
        # YOLO models
        yolo_models = {
            'YOLOv8l': self.models_dir / 'yolov8l.pt',
            'YOLOv8x': self.models_dir / 'yolov8x.pt',
        }
        
        for name, model_path in yolo_models.items():
            if model_path.exists():
                try:
                    print(f"Loading {name} from {model_path}...")
                    model = YOLO(str(model_path))
                    self.models[name] = {
                        'model': model,
                        'type': 'yolo',
                        'loaded': True,
                        'path': str(model_path)
                    }
                    print(f"✅ {name} loaded successfully")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")

        # DETR models
        detr_models = {
            'DETR-ResNet50': self.models_dir / 'detr50.pth',
            'DETR-ResNet101': self.models_dir / 'detr101.pth'
        }
        
        for name, model_path in detr_models.items():
            try:
                print(f"Loading {name} from {model_path}...")
                
                if 'ResNet50' in name:
                    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
                else:
                    model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
                
                model.eval()
                self.models[name] = {
                    'model': model,
                    'type': 'detr',
                    'loaded': True,
                    'path': str(model_path) if model_path.exists() else 'pretrained'
                }
                print(f"✅ {name} loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")

        loaded_count = sum(1 for m in self.models.values() if m.get('loaded', False))
        print(f"\nLoaded {loaded_count} models successfully")
        
        for name, info in self.models.items():
            if info.get('loaded', False):
                print(f"  ✅ {name} ({info['type']}) - {info.get('path', 'unknown')}")

    def get_test_directories(self):
        """Get possible test directories"""
        return [
            self.data_dir / "VisDrone2019-DET-val" / "images",
            self.data_dir / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "images",
            self.data_dir / "VisDrone2019-DET-test-dev" / "images",
            self.data_dir / "VisDrone2019-DET-test-dev" / "VisDrone2019-DET-test-dev" / "images",
            Path("data"),
            Path("test_images")
        ]

    def find_test_images(self, max_images: int = 50):
        """Find test images from various possible locations"""
        possible_dirs = self.get_test_directories()
        
        image_files = []
        for test_dir in possible_dirs:
            if test_dir.exists():
                print(f"Found test images directory: {test_dir}")
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(list(test_dir.glob(ext)))
                if image_files:
                    break
        
        if not image_files:
            print("❌ No test images found")
            return []
        
        if len(image_files) > max_images:
            image_files = image_files[:max_images]
        
        print(f"Found {len(image_files)} test images")
        return image_files

    def run_yolo_inference(self, model, image_path):
        """Run YOLO inference on image"""
        try:
            results = model(str(image_path), verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        detections.append({
                            'bbox': box.tolist(),
                            'confidence': float(score),
                            'class': int(cls)
                        })
            
            return {
                'detections': detections,
                'detection_count': len(detections),
                'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
                'status': 'success'
            }
        except Exception as e:
            return {
                'detections': [],
                'detection_count': 0,
                'avg_confidence': 0,
                'status': 'error',
                'error': str(e)
            }

    def run_detr_inference(self, model, image_path):
        """Run DETR inference on image"""
        try:
            image = Image.open(image_path).convert('RGB')
            transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
            
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.5
            
            detections = []
            if keep.sum() > 0:
                boxes = outputs['pred_boxes'][0, keep].cpu()
                scores = probas[keep].max(-1).values.cpu()
                classes = probas[keep].max(-1).indices.cpu()
                
                h, w = image.size[1], image.size[0]
                for box, score, cls in zip(boxes, scores, classes):
                    cx, cy, bw, bh = box
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w
                    y2 = (cy + bh/2) * h
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(score),
                        'class': int(cls)
                    })
            
            return {
                'detections': detections,
                'detection_count': len(detections),
                'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
                'status': 'success'
            }
        except Exception as e:
            return {
                'detections': [],
                'detection_count': 0,
                'avg_confidence': 0,
                'status': 'error',
                'error': str(e)
            }

    def draw_yolo_detections(self, image_path: Path, detections: List[Dict], model_name: str) -> Image.Image:
        """Draw YOLO detection results on image"""
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#85C1E9'
        ]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = int(det['class'])
            confidence = det['confidence']
            
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            color = colors[class_id % len(colors)]
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            label = f"{class_name}: {confidence:.2f}"
            bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1-25), label, fill='white', font=font)
        
        return image

    def draw_detr_detections(self, image_path: Path, detections: List[Dict], model_name: str) -> Image.Image:
        """Draw DETR detection results on image"""
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        class_names = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
            'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#85C1E9'
        ]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = int(det['class'])
            confidence = det['confidence']
            
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            if class_name == 'N/A':
                class_name = f"class_{class_id}"
            color = colors[class_id % len(colors)]
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            label = f"{class_name}: {confidence:.2f}"
            bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1-25), label, fill='white', font=font)
        
        return image

    def process_single_image(self, image_path: Path):
        """Process a single image with all models"""
        image_name = image_path.name
        
        for model_name, model_info in self.models.items():
            if not model_info.get('loaded', False):
                continue
            
            # Initialize model stats if not exists
            if model_name not in self.model_stats:
                self.model_stats[model_name] = {
                    'image_results': {},
                    'total_detections': 0,
                    'successful_images': 0,
                    'failed_images': 0
                }
            
            # Run inference
            if model_info['type'] == 'yolo':
                result = self.run_yolo_inference(model_info['model'], image_path)
            elif model_info['type'] == 'detr':
                result = self.run_detr_inference(model_info['model'], image_path)
            else:
                continue
            
            # Store results
            self.model_stats[model_name]['image_results'][image_name] = result
            
            if result['status'] == 'success':
                self.model_stats[model_name]['successful_images'] += 1
                self.model_stats[model_name]['total_detections'] += result['detection_count']
            else:
                self.model_stats[model_name]['failed_images'] += 1

    def analyze_comprehensive_results(self):
        """Analyze comprehensive results"""
        print("\n=== Analyzing Comprehensive Results ===")
        
        for model_name, stats in self.model_stats.items():
            if stats['successful_images'] > 0:
                # Calculate averages
                stats['avg_detections_per_image'] = stats['total_detections'] / stats['successful_images']
                
                # Calculate confidence statistics
                confidences = []
                detection_counts = []
                
                for result in stats['image_results'].values():
                    if result['status'] == 'success':
                        confidences.append(result['avg_confidence'])
                        detection_counts.append(result['detection_count'])
                
                stats['avg_confidence'] = np.mean(confidences) if confidences else 0
                stats['detection_std'] = np.std(detection_counts) if detection_counts else 0
                stats['confidence_std'] = np.std(confidences) if confidences else 0
                stats['detection_range'] = (min(detection_counts), max(detection_counts)) if detection_counts else (0, 0)
                stats['confidence_range'] = (min(confidences), max(confidences)) if confidences else (0, 0)
                
                # Calculate performance consistency
                stats['performance_consistency'] = 1.0 / (1.0 + stats['detection_std']) if stats['detection_std'] > 0 else 1.0
                
                # Find best and worst images
                image_scores = []
                for img_name, result in stats['image_results'].items():
                    if result['status'] == 'success':
                        score = result['detection_count'] * result['avg_confidence']
                        image_scores.append((img_name, score, result))
                
                image_scores.sort(key=lambda x: x[1], reverse=True)
                stats['best_image'] = image_scores[0] if image_scores else None
                stats['worst_image'] = image_scores[-1] if image_scores else None
                
                # Count detection frequency
                detection_freq = defaultdict(int)
                for result in stats['image_results'].values():
                    if result['status'] == 'success':
                        detection_freq[result['detection_count']] += 1
                
                stats['most_common_detections'] = dict(list(detection_freq.items())[:3])

        self.print_comprehensive_summary()

    def print_comprehensive_summary(self):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE LOCAL MODEL EVALUATION SUMMARY")
        print("="*80)
        
        for model_name, stats in self.model_stats.items():
            print(f"\n📊 {model_name}")
            print(f"   📈 Basic Metrics:")
            print(f"      Successful images: {stats['successful_images']}")
            print(f"      Failed images: {stats['failed_images']}")
            success_rate = stats['successful_images']/(stats['successful_images']+stats['failed_images'])*100 if (stats['successful_images']+stats['failed_images']) > 0 else 0
            print(f"      Success rate: {success_rate:.1f}%")
            
            print(f"   🎯 Detection Metrics:")
            print(f"      Total detections: {stats['total_detections']}")
            print(f"      Avg detections per image: {stats.get('avg_detections_per_image', 0):.2f}")
            print(f"      Detection std: {stats.get('detection_std', 0):.2f}")
            print(f"      Detection range: {stats.get('detection_range', (0, 0))[0]} - {stats.get('detection_range', (0, 0))[1]}")
            
            print(f"   🎯 Confidence Metrics:")
            print(f"      Avg confidence: {stats.get('avg_confidence', 0):.3f}")
            print(f"      Confidence std: {stats.get('confidence_std', 0):.3f}")
            print(f"      Confidence range: {stats.get('confidence_range', (0, 0))[0]:.3f} - {stats.get('confidence_range', (0, 0))[1]:.3f}")
            
            print(f"   🏆 Performance Highlights:")
            if stats.get('best_image'):
                best = stats['best_image']
                print(f"      Best image: {best[0]} (score: {best[1]:.2f}, {best[2]['detection_count']} detections)")
            if stats.get('worst_image'):
                worst = stats['worst_image']
                print(f"      Worst image: {worst[0]} (score: {worst[1]:.2f}, {worst[2]['detection_count']} detections)")
            
            print(f"   📊 Advanced Metrics:")
            print(f"      Performance consistency: {stats.get('performance_consistency', 0):.3f}")
            print(f"      Detection rate: {stats.get('avg_detections_per_image', 0):.2f}")
            print(f"      Most common detections: {stats.get('most_common_detections', {})}")

    def save_detection_visualizations(self):
        """Save detection visualizations for top 3 best and worst images per model"""
        print("\n=== Saving Detection Visualizations ===")
        
        if not hasattr(self, 'model_stats') or not self.model_stats:
            print("⚠️ No model results available for visualization")
            return
        
        for model_name, stats in self.model_stats.items():
            print(f"\n📸 Processing visualizations for {model_name}...")
            
            image_results = stats.get('image_results', {})
            if not image_results:
                print(f"   ⚠️ No image results found for {model_name}")
                continue
            
            # Sort images by performance score (detection_count * avg_confidence)
            sorted_images = []
            for img_name, result in image_results.items():
                if result['status'] == 'success':
                    score = result['detection_count'] * result['avg_confidence']
                    sorted_images.append((img_name, result, score))
            
            sorted_images.sort(key=lambda x: x[2], reverse=True)
            
            if len(sorted_images) < 3:
                print(f"   ⚠️ Not enough successful results for {model_name}")
                continue
            
            # Get top 3 best and worst
            best_images = sorted_images[:3]
            worst_images = sorted_images[-3:]
            
            # Create model-specific directory
            model_viz_dir = self.detection_viz_dir / model_name
            model_viz_dir.mkdir(exist_ok=True)
            
            # Process best images
            for i, (image_name, result, score) in enumerate(best_images, 1):
                try:
                    image_path = None
                    # Find the actual image file
                    for test_dir in self.get_test_directories():
                        potential_path = test_dir / image_name
                        if potential_path.exists():
                            image_path = potential_path
                            break
                    
                    if image_path is None:
                        print(f"   ❌ Could not find image file: {image_name}")
                        continue
                    
                    # Draw detections based on model type
                    if any(yolo_name in model_name.lower() for yolo_name in ['yolo', 'yv']):
                        annotated_image = self.draw_yolo_detections(
                            image_path, result['detections'], model_name
                        )
                    else:  # DETR models
                        annotated_image = self.draw_detr_detections(
                            image_path, result['detections'], model_name
                        )
                    
                    # Save annotated image
                    output_path = model_viz_dir / f"best_{i}_{image_name.replace('.jpg', '.png')}"
                    annotated_image.save(output_path)
                    
                    print(f"   ✅ Saved best #{i}: {output_path.name} "
                          f"({result['detection_count']} detections, "
                          f"avg conf: {result['avg_confidence']:.3f})")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {image_name}: {str(e)}")
            
            # Process worst images
            for i, (image_name, result, score) in enumerate(worst_images, 1):
                try:
                    image_path = None
                    # Find the actual image file
                    for test_dir in self.get_test_directories():
                        potential_path = test_dir / image_name
                        if potential_path.exists():
                            image_path = potential_path
                            break
                    
                    if image_path is None:
                        print(f"   ❌ Could not find image file: {image_name}")
                        continue
                    
                    # Draw detections based on model type
                    if any(yolo_name in model_name.lower() for yolo_name in ['yolo', 'yv']):
                        annotated_image = self.draw_yolo_detections(
                            image_path, result['detections'], model_name
                        )
                    else:  # DETR models
                        annotated_image = self.draw_detr_detections(
                            image_path, result['detections'], model_name
                        )
                    
                    # Save annotated image
                    output_path = model_viz_dir / f"worst_{i}_{image_name.replace('.jpg', '.png')}"
                    annotated_image.save(output_path)
                    
                    print(f"   ✅ Saved worst #{i}: {output_path.name} "
                          f"({result['detection_count']} detections, "
                          f"avg conf: {result['avg_confidence']:.3f})")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {image_name}: {str(e)}")
        
        print(f"\n✅ Detection visualizations saved in: {self.detection_viz_dir}")

    def save_comprehensive_results(self):
        """Save comprehensive results to files"""
        print("\n=== Saving Comprehensive Results ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = self.results_dir / f"comprehensive_results_{timestamp}.json"
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json_data = {
            'timestamp': timestamp,
            'model_stats': convert_numpy(self.model_stats)
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Create CSV comparison
        csv_file = self.results_dir / f"model_metrics_comparison_{timestamp}.csv"
        comparison_data = []
        
        for model_name, stats in self.model_stats.items():
            comparison_data.append({
                'Model': model_name,
                'Successful_Images': stats['successful_images'],
                'Failed_Images': stats['failed_images'],
                'Total_Detections': stats['total_detections'],
                'Avg_Detections_Per_Image': stats.get('avg_detections_per_image', 0),
                'Avg_Confidence': stats.get('avg_confidence', 0),
                'Performance_Consistency': stats.get('performance_consistency', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(csv_file, index=False)
        
        # Save text report
        report_file = self.results_dir / f"comprehensive_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE LOCAL MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for model_name, stats in self.model_stats.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"  Successful images: {stats['successful_images']}\n")
                f.write(f"  Failed images: {stats['failed_images']}\n")
                f.write(f"  Total detections: {stats['total_detections']}\n")
                f.write(f"  Avg detections per image: {stats.get('avg_detections_per_image', 0):.2f}\n")
                f.write(f"  Avg confidence: {stats.get('avg_confidence', 0):.3f}\n\n")
        
        print(f"✅ Comprehensive results saved:")
        print(f"   📄 Detailed JSON: {results_file}")
        print(f"   📊 Metrics CSV: {csv_file}")
        print(f"   📄 Text Report: {report_file}")

    def create_comprehensive_visualizations(self):
        """Create comprehensive performance visualization plots"""
        print("\n=== Creating Comprehensive Visualizations ===")
        
        if not self.model_stats:
            print("❌ No results to visualize")
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Local Model Evaluation Results', fontsize=16, fontweight='bold')
        
        models = list(self.model_stats.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # Plot 1: Detection Performance
        detections = [self.model_stats[m].get('avg_detections_per_image', 0) for m in models]
        axes[0, 0].bar(models, detections, color=colors)
        axes[0, 0].set_title('Average Detections per Image')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Confidence Performance
        confidences = [self.model_stats[m].get('avg_confidence', 0) for m in models]
        axes[0, 1].bar(models, confidences, color=colors)
        axes[0, 1].set_title('Average Confidence Score')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Success Rate
        success_rates = []
        for m in models:
            stats = self.model_stats[m]
            total = stats['successful_images'] + stats['failed_images']
            rate = (stats['successful_images'] / total * 100) if total > 0 else 0
            success_rates.append(rate)
        
        axes[1, 0].bar(models, success_rates, color=colors)
        axes[1, 0].set_title('Success Rate (%)')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim([0, 100])
        
        # Plot 4: Performance Consistency
        consistency = [self.model_stats[m].get('performance_consistency', 0) for m in models]
        axes[1, 1].bar(models, consistency, color=colors)
        axes[1, 1].set_title('Performance Consistency')
        axes[1, 1].set_ylabel('Consistency Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"comprehensive_visualization_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive visualization saved: {plot_file}")

    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation of all models"""
        print("\n=== Starting Comprehensive Local Model Evaluation ===")
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("❌ No models loaded. Cannot proceed with evaluation.")
            return
        
        # Find test images
        test_images = self.find_test_images()
        if not test_images:
            print("❌ No test images found. Cannot proceed with evaluation.")
            return
        
        print(f"\nEvaluating {len(test_images)} images with {len(self.models)} models...")
        print(f"Models: {', '.join(self.models.keys())}")
        
        # Process each image with all models
        for image_path in tqdm(test_images, desc="Processing images"):
            self.process_single_image(image_path)
        
        # Analyze results
        self.analyze_comprehensive_results()
        
        # Save results
        self.save_comprehensive_results()
        
        print("✅ Comprehensive evaluation completed!")


if __name__ == "__main__":
    print("🚀 Starting Comprehensive Local Model Metrics Evaluation")
    
    # Initialize evaluator
    evaluator = LocalMetricsEvaluator()
    print("Local Metrics Evaluator initialized")
    print(f"Models directory: {evaluator.models_dir}")
    print(f"Data directory: {evaluator.data_dir}")
    print(f"Results directory: {evaluator.results_dir}")
    
    # Run comprehensive evaluation
    evaluator.run_comprehensive_evaluation()
    
    # Create visualizations only if we have results
    if hasattr(evaluator, 'model_stats') and evaluator.model_stats:
        evaluator.create_comprehensive_visualizations()
        
        # Save detection visualizations for best/worst images
        evaluator.save_detection_visualizations()
    else:
        print("⚠️ No results to visualize - skipping visualization step")
    
    print("\n🎉 Comprehensive local model evaluation completed!")
    print(f"📁 Results saved in: {evaluator.results_dir}")
    print(f"📸 Detection visualizations saved in: {evaluator.detection_viz_dir}")