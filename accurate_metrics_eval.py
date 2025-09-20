#!/usr/bin/env python3
"""
Accurate Model Metrics Evaluation Script
Evaluates models against ground truth annotations with proper accuracy metrics
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

class AccurateMetricsEvaluator:
    def __init__(self):
        """Initialize with user input for paths"""
        print("🚀 Accurate Model Metrics Evaluator")
        print("=" * 60)
        
        # Get dataset path from user
        self.data_dir = self.get_dataset_path()
        
        # Set up directories
        self.models_dir = Path("models")
        self.results_dir = Path("accurate_metrics_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection visualization directory
        self.detection_viz_dir = self.results_dir / "detection_visualizations"
        self.detection_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # VisDrone class mapping
        self.visdrone_classes = {
            0: "ignored_regions",
            1: "pedestrian", 
            2: "people",
            3: "bicycle",
            4: "car",
            5: "van",
            6: "truck",
            7: "tricycle",
            8: "awning-tricycle",
            9: "bus",
            10: "motor"
        }
        
        # COCO to VisDrone class mapping for DETR
        self.coco_to_visdrone = {
            0: 1,   # person -> pedestrian
            1: 3,   # bicycle -> bicycle  
            2: 4,   # car -> car
            3: 10,  # motorcycle -> motor
            5: 9,   # bus -> bus
            7: 6    # truck -> truck
        }
        
        # Initialize storage
        self.models = {}
        self.ground_truth = {}
        self.model_results = {}
        
        print(f"📁 Dataset directory: {self.data_dir}")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📁 Results directory: {self.results_dir}")

    def get_dataset_path(self) -> Path:
        """Get dataset path from user input"""
        print("\n📍 Dataset Location Setup")
        print("Please provide the path to your VisDrone dataset directory.")
        print("Example: /data/VisDrone2019Data or /Users/username/data/VisDrone")
        
        while True:
            path_input = input("\nEnter dataset path: ").strip()
            
            if not path_input:
                print("❌ Please enter a valid path")
                continue
                
            data_path = Path(path_input)
            
            if not data_path.exists():
                print(f"❌ Path does not exist: {data_path}")
                continue
            
            # Check for required subdirectories
            val_dir = data_path / "VisDrone2019-DET-test-dev"
            if not val_dir.exists():
                print(f"❌ VisDrone2019-DET-test-dev not found in {data_path}")
                print("Available directories:")
                for item in data_path.iterdir():
                    if item.is_dir():
                        print(f"  📁 {item.name}")
                continue
            
            print(f"✅ Valid dataset path: {data_path}")
            return data_path

    def load_models(self):
        """Load all available models"""
        print("\n=== Loading Models ===")
        
        # YOLO models
        yolo_models = {
            'YOLOv8l': self.models_dir / 'yolov8l.pt',
            'YOLOv8x': self.models_dir / 'yolov8x.pt',
        }
        
        for name, model_path in yolo_models.items():
            if model_path.exists():
                try:
                    print(f"Loading {name}...")
                    model = YOLO(str(model_path))
                    self.models[name] = {
                        'model': model,
                        'type': 'yolo',
                        'loaded': True
                    }
                    print(f"✅ {name} loaded successfully")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")

        # DETR models
        detr_models = ['DETR-ResNet50', 'DETR-ResNet101']
        
        for name in detr_models:
            try:
                print(f"Loading {name}...")
                if 'ResNet50' in name:
                    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
                else:
                    model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
                
                model.eval()
                self.models[name] = {
                    'model': model,
                    'type': 'detr',
                    'loaded': True
                }
                print(f"✅ {name} loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")

        loaded_count = sum(1 for m in self.models.values() if m.get('loaded', False))
        print(f"\n✅ Loaded {loaded_count} models successfully")

    def load_ground_truth_annotations(self):
        """Load VisDrone ground truth annotations"""
        print("\n=== Loading Ground Truth Annotations ===")
        
        # Find annotation directories
        val_annotations_dir = self.data_dir / "VisDrone2019-DET-test-dev" / "annotations"
        if not val_annotations_dir.exists():
            # Try alternative structure
            val_annotations_dir = self.data_dir / "VisDrone2019-DET-test-dev" / "VisDrone2019-DET-test-dev" / "annotations"
        
        if not val_annotations_dir.exists():
            print(f"❌ Annotations directory not found")
            return False
        
        print(f"📁 Loading annotations from: {val_annotations_dir}")
        
        annotation_files = list(val_annotations_dir.glob("*.txt"))
        print(f"Found {len(annotation_files)} annotation files")
        
        for ann_file in tqdm(annotation_files, desc="Loading annotations"):
            image_name = ann_file.stem + ".jpg"
            annotations = []
            
            try:
                with open(ann_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('<'):
                            continue
                        
                        parts = line.split(',')
                        if len(parts) >= 8:
                            x, y, w, h, score, class_id, truncation, occlusion = map(int, parts[:8])
                            
                            # Skip ignored regions and invalid annotations
                            if class_id == 0 or w <= 0 or h <= 0:
                                continue
                            
                            annotations.append({
                                'bbox': [x, y, x + w, y + h],  # Convert to x1,y1,x2,y2
                                'class': class_id,
                                'class_name': self.visdrone_classes.get(class_id, f'class_{class_id}'),
                                'score': score,
                                'truncation': truncation,
                                'occlusion': occlusion
                            })
                
                self.ground_truth[image_name] = annotations
                
            except Exception as e:
                print(f"❌ Error loading {ann_file}: {e}")
        
        print(f"✅ Loaded ground truth for {len(self.ground_truth)} images")
        return True

    def find_test_images(self):
        """Find all test images"""
        print("\n=== Finding Test Images ===")
        
        # Find images directory
        val_images_dir = self.data_dir / "VisDrone2019-DET-test-dev" / "images"
        if not val_images_dir.exists():
            val_images_dir = self.data_dir / "VisDrone2019-DET-test-dev" / "VisDrone2019-DET-test-dev" / "images"
        
        if not val_images_dir.exists():
            print(f"❌ Images directory not found")
            return []
        
        print(f"📁 Loading images from: {val_images_dir}")
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(val_images_dir.glob(ext)))
        
        # Filter to only images with ground truth
        valid_images = []
        for img_path in image_files:
            if img_path.name in self.ground_truth:
                valid_images.append(img_path)
        
        print(f"✅ Found {len(valid_images)} images with ground truth annotations")
        return valid_images

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def match_predictions_to_ground_truth(self, predictions: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Match predictions to ground truth annotations"""
        matched_predictions = []
        matched_ground_truth = []
        unmatched_predictions = []
        
        gt_matched = [False] * len(ground_truth)
        
        # Sort predictions by confidence (highest first)
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        for pred in sorted_predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                # Check class match (for DETR, map COCO to VisDrone)
                pred_class = pred['class']
                if 'detr' in pred.get('model_type', '').lower():
                    pred_class = self.coco_to_visdrone.get(pred_class, pred_class)
                
                if pred_class != gt['class']:
                    continue
                
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                matched_predictions.append({**pred, 'matched_gt': ground_truth[best_gt_idx], 'iou': best_iou})
                matched_ground_truth.append(ground_truth[best_gt_idx])
            else:
                unmatched_predictions.append(pred)
        
        unmatched_ground_truth = [gt for i, gt in enumerate(ground_truth) if not gt_matched[i]]
        
        return matched_predictions, unmatched_predictions, unmatched_ground_truth

    def calculate_accuracy_metrics(self, predictions: List[Dict], ground_truth: List[Dict], model_type: str = '') -> Dict:
        """Calculate comprehensive accuracy metrics"""
        if not ground_truth:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'ap_50': 0.0,
                'num_predictions': len(predictions),
                'num_ground_truth': 0,
                'true_positives': 0,
                'false_positives': len(predictions),
                'false_negatives': 0
            }
        
        # Add model type to predictions for class mapping
        for pred in predictions:
            pred['model_type'] = model_type
        
        matched_pred, unmatched_pred, unmatched_gt = self.match_predictions_to_ground_truth(predictions, ground_truth)
        
        tp = len(matched_pred)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate AP@0.5 (simplified)
        if matched_pred:
            sorted_matches = sorted(matched_pred, key=lambda x: x['confidence'], reverse=True)
            precisions = []
            for i in range(len(sorted_matches)):
                current_tp = i + 1
                current_fp = len([p for p in predictions if p['confidence'] >= sorted_matches[i]['confidence']]) - current_tp
                current_precision = current_tp / (current_tp + current_fp) if (current_tp + current_fp) > 0 else 0.0
                precisions.append(current_precision)
            ap_50 = np.mean(precisions) if precisions else 0.0
        else:
            ap_50 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ap_50': ap_50,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'matched_predictions': matched_pred,
            'unmatched_predictions': unmatched_pred,
            'unmatched_ground_truth': unmatched_gt
        }

    def run_yolo_inference(self, model, image_path: Path) -> List[Dict]:
        """Run YOLO inference and return detections"""
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
                        # Map YOLO classes to VisDrone classes if needed
                        visdrone_class = cls + 1  # YOLO classes are 0-indexed, VisDrone are 1-indexed
                        if visdrone_class in self.visdrone_classes:
                            detections.append({
                                'bbox': box.tolist(),
                                'confidence': float(score),
                                'class': visdrone_class,
                                'class_name': self.visdrone_classes[visdrone_class]
                            })
            
            return detections
        except Exception as e:
            print(f"❌ YOLO inference error: {e}")
            return []

    def run_detr_inference(self, model, image_path: Path) -> List[Dict]:
        """Run DETR inference and return detections"""
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
                    # Only keep classes that map to VisDrone
                    if int(cls) in self.coco_to_visdrone:
                        cx, cy, bw, bh = box
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        
                        visdrone_class = self.coco_to_visdrone[int(cls)]
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(score),
                            'class': visdrone_class,
                            'class_name': self.visdrone_classes[visdrone_class]
                        })
            
            return detections
        except Exception as e:
            print(f"❌ DETR inference error: {e}")
            return []

    def evaluate_single_image(self, image_path: Path) -> Dict:
        """Evaluate all models on a single image"""
        image_name = image_path.name
        ground_truth = self.ground_truth.get(image_name, [])
        
        image_results = {
            'image_path': str(image_path),
            'image_name': image_name,
            'ground_truth': ground_truth,
            'models': {}
        }
        
        for model_name, model_info in self.models.items():
            if not model_info.get('loaded', False):
                continue
            
            # Run inference
            if model_info['type'] == 'yolo':
                predictions = self.run_yolo_inference(model_info['model'], image_path)
            elif model_info['type'] == 'detr':
                predictions = self.run_detr_inference(model_info['model'], image_path)
            else:
                continue
            
            # Calculate accuracy metrics
            metrics = self.calculate_accuracy_metrics(predictions, ground_truth, model_info['type'])
            
            image_results['models'][model_name] = {
                'predictions': predictions,
                'metrics': metrics,
                'status': 'success'
            }
        
        return image_results

    def draw_detections_with_accuracy(self, image_path: Path, predictions: List[Dict], ground_truth: List[Dict], 
                                    metrics: Dict, model_name: str) -> Image.Image:
        """Draw detections with accuracy information"""
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw ground truth in green
        for gt in ground_truth:
            x1, y1, x2, y2 = gt['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
            draw.text((x1, y1-20), f"GT: {gt['class_name']}", fill='green', font=font)
        
        # Draw matched predictions in blue, unmatched in red
        matched_preds = metrics.get('matched_predictions', [])
        unmatched_preds = metrics.get('unmatched_predictions', [])
        
        for pred in matched_preds:
            x1, y1, x2, y2 = pred['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
            label = f"{pred['class_name']}: {pred['confidence']:.2f} (IoU: {pred['iou']:.2f})"
            draw.text((x1, y2+5), label, fill='blue', font=font)
        
        for pred in unmatched_preds:
            x1, y1, x2, y2 = pred['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            label = f"{pred['class_name']}: {pred['confidence']:.2f} (FP)"
            draw.text((x1, y2+5), label, fill='red', font=font)
        
        # Add metrics text
        metrics_text = f"{model_name} | P: {metrics['precision']:.3f} R: {metrics['recall']:.3f} F1: {metrics['f1_score']:.3f} AP: {metrics['ap_50']:.3f}"
        draw.text((10, 10), metrics_text, fill='black', font=title_font)
        
        return image

    def save_detection_visualizations(self):
        """Save detection visualizations for best and worst images by accuracy"""
        print("\n=== Saving Accurate Detection Visualizations ===")
        
        if not hasattr(self, 'model_results') or not self.model_results:
            print("⚠️ No model results available for visualization")
            return
        
        for model_name in self.models.keys():
            if not self.models[model_name].get('loaded', False):
                continue
                
            print(f"\n📸 Processing visualizations for {model_name}...")
            
            # Collect all image results for this model
            image_scores = []
            for image_result in self.model_results:
                if model_name in image_result['models']:
                    model_result = image_result['models'][model_name]
                    if model_result['status'] == 'success':
                        metrics = model_result['metrics']
                        # Use F1 score as primary ranking metric
                        score = metrics['f1_score']
                        image_scores.append({
                            'image_name': image_result['image_name'],
                            'image_path': image_result['image_path'],
                            'score': score,
                            'metrics': metrics,
                            'predictions': model_result['predictions'],
                            'ground_truth': image_result['ground_truth']
                        })
            
            if len(image_scores) < 6:
                print(f"   ⚠️ Not enough results for {model_name} ({len(image_scores)} images)")
                continue
            
            # Sort by F1 score
            image_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Get top 3 best and worst
            best_images = image_scores[:3]
            worst_images = image_scores[-3:]
            
            # Create model directory
            model_viz_dir = self.detection_viz_dir / model_name
            model_viz_dir.mkdir(exist_ok=True)
            
            # Save best images
            for i, img_data in enumerate(best_images, 1):
                try:
                    annotated_image = self.draw_detections_with_accuracy(
                        Path(img_data['image_path']),
                        img_data['predictions'],
                        img_data['ground_truth'],
                        img_data['metrics'],
                        model_name
                    )
                    
                    output_path = model_viz_dir / f"best_{i}_{img_data['image_name'].replace('.jpg', '.png')}"
                    annotated_image.save(output_path)
                    
                    print(f"   ✅ Saved best #{i}: F1={img_data['score']:.3f}, "
                          f"P={img_data['metrics']['precision']:.3f}, "
                          f"R={img_data['metrics']['recall']:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing best image {i}: {e}")
            
            # Save worst images  
            for i, img_data in enumerate(worst_images, 1):
                try:
                    annotated_image = self.draw_detections_with_accuracy(
                        Path(img_data['image_path']),
                        img_data['predictions'],
                        img_data['ground_truth'],
                        img_data['metrics'],
                        model_name
                    )
                    
                    output_path = model_viz_dir / f"worst_{i}_{img_data['image_name'].replace('.jpg', '.png')}"
                    annotated_image.save(output_path)
                    
                    print(f"   ✅ Saved worst #{i}: F1={img_data['score']:.3f}, "
                          f"P={img_data['metrics']['precision']:.3f}, "
                          f"R={img_data['metrics']['recall']:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing worst image {i}: {e}")
        
        print(f"\n✅ Accurate detection visualizations saved in: {self.detection_viz_dir}")

    def analyze_results(self):
        """Analyze comprehensive results"""
        print("\n=== Analyzing Results ===")
        
        model_stats = {}
        
        for model_name in self.models.keys():
            if not self.models[model_name].get('loaded', False):
                continue
                
            all_metrics = []
            for image_result in self.model_results:
                if model_name in image_result['models']:
                    model_result = image_result['models'][model_name]
                    if model_result['status'] == 'success':
                        all_metrics.append(model_result['metrics'])
            
            if not all_metrics:
                continue
            
            # Calculate aggregate statistics
            stats = {
                'total_images': len(all_metrics),
                'avg_precision': np.mean([m['precision'] for m in all_metrics]),
                'avg_recall': np.mean([m['recall'] for m in all_metrics]),
                'avg_f1_score': np.mean([m['f1_score'] for m in all_metrics]),
                'avg_ap_50': np.mean([m['ap_50'] for m in all_metrics]),
                'total_tp': sum(m['true_positives'] for m in all_metrics),
                'total_fp': sum(m['false_positives'] for m in all_metrics),
                'total_fn': sum(m['false_negatives'] for m in all_metrics),
                'total_predictions': sum(m['num_predictions'] for m in all_metrics),
                'total_ground_truth': sum(m['num_ground_truth'] for m in all_metrics),
                'precision_std': np.std([m['precision'] for m in all_metrics]),
                'recall_std': np.std([m['recall'] for m in all_metrics]),
                'f1_std': np.std([m['f1_score'] for m in all_metrics])
            }
            
            model_stats[model_name] = stats
        
        self.model_stats = model_stats
        self.print_results_summary()

    def print_results_summary(self):
        """Print comprehensive results summary"""
        print("\n" + "="*80)
        print("ACCURATE MODEL EVALUATION RESULTS")
        print("="*80)
        
        for model_name, stats in self.model_stats.items():
            print(f"\n📊 {model_name}")
            print(f"   📈 Images Processed: {stats['total_images']}")
            print(f"   🎯 Accuracy Metrics:")
            print(f"      Precision: {stats['avg_precision']:.3f} ± {stats['precision_std']:.3f}")
            print(f"      Recall: {stats['avg_recall']:.3f} ± {stats['recall_std']:.3f}")
            print(f"      F1-Score: {stats['avg_f1_score']:.3f} ± {stats['f1_std']:.3f}")
            print(f"      AP@0.5: {stats['avg_ap_50']:.3f}")
            print(f"   📊 Detection Statistics:")
            print(f"      True Positives: {stats['total_tp']}")
            print(f"      False Positives: {stats['total_fp']}")
            print(f"      False Negatives: {stats['total_fn']}")
            print(f"      Total Predictions: {stats['total_predictions']}")
            print(f"      Total Ground Truth: {stats['total_ground_truth']}")

    def save_results(self):
        """Save comprehensive results"""
        print("\n=== Saving Results ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        results_file = self.results_dir / f"accurate_results_{timestamp}.json"
        
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
        
        results_data = {
            'timestamp': timestamp,
            'dataset_path': str(self.data_dir),
            'model_stats': convert_numpy(self.model_stats),
            'detailed_results': convert_numpy(self.model_results)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save CSV summary
        csv_file = self.results_dir / f"model_accuracy_comparison_{timestamp}.csv"
        csv_data = []
        
        for model_name, stats in self.model_stats.items():
            csv_data.append({
                'Model': model_name,
                'Images': stats['total_images'],
                'Precision': stats['avg_precision'],
                'Recall': stats['avg_recall'],
                'F1_Score': stats['avg_f1_score'],
                'AP_50': stats['avg_ap_50'],
                'True_Positives': stats['total_tp'],
                'False_Positives': stats['total_fp'],
                'False_Negatives': stats['total_fn'],
                'Precision_Std': stats['precision_std'],
                'Recall_Std': stats['recall_std'],
                'F1_Std': stats['f1_std']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Results saved:")
        print(f"   📄 JSON: {results_file}")
        print(f"   📊 CSV: {csv_file}")

    def create_visualizations(self):
        """Create performance visualization plots"""
        print("\n=== Creating Visualizations ===")
        
        if not self.model_stats:
            print("❌ No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Accuracy Evaluation Results', fontsize=16, fontweight='bold')
        
        models = list(self.model_stats.keys())
        
        # Precision comparison
        precisions = [self.model_stats[m]['avg_precision'] for m in models]
        axes[0, 0].bar(models, precisions)
        axes[0, 0].set_title('Average Precision')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        recalls = [self.model_stats[m]['avg_recall'] for m in models]
        axes[0, 1].bar(models, recalls)
        axes[0, 1].set_title('Average Recall')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        f1_scores = [self.model_stats[m]['avg_f1_score'] for m in models]
        axes[1, 0].bar(models, f1_scores)
        axes[1, 0].set_title('Average F1-Score')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # AP@0.5 comparison
        ap_scores = [self.model_stats[m]['avg_ap_50'] for m in models]
        axes[1, 1].bar(models, ap_scores)
        axes[1, 1].set_title('Average AP@0.5')
        axes[1, 1].set_ylabel('AP@0.5')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"accuracy_visualization_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved: {plot_file}")

    def run_evaluation(self):
        """Run complete evaluation"""
        print("\n🚀 Starting Accurate Model Evaluation")
        
        # Load models
        self.load_models()
        if not any(m.get('loaded', False) for m in self.models.values()):
            print("❌ No models loaded")
            return
        
        # Load ground truth
        if not self.load_ground_truth_annotations():
            print("❌ Failed to load ground truth annotations")
            return
        
        # Find test images
        test_images = self.find_test_images()
        if not test_images:
            print("❌ No test images found")
            return
        
        print(f"\n🔄 Processing {len(test_images)} images...")
        
        # Process all images
        self.model_results = []
        for image_path in tqdm(test_images, desc="Evaluating images"):
            result = self.evaluate_single_image(image_path)
            self.model_results.append(result)
        
        # Analyze results
        self.analyze_results()
        
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save detection visualizations
        self.save_detection_visualizations()
        
        print("\n🎉 Accurate evaluation completed!")
        print(f"📁 Results saved in: {self.results_dir}")


if __name__ == "__main__":
    evaluator = AccurateMetricsEvaluator()
    evaluator.run_evaluation()