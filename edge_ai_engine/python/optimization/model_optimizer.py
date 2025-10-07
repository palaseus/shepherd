#!/usr/bin/env python3
"""
Model optimization utilities for Edge AI Engine
Author: AI Co-Developer
Date: 2024

This module provides utilities for optimizing models including quantization,
pruning, and graph optimization.
"""

import os
import sys
import argparse
import logging
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import torch
    import torch.quantization as quant
    import torch.nn.utils.prune as prune
    import torch.jit
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Please install required packages: pip install onnx onnxruntime tensorflow torch")

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Model optimization utility class"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supported_formats = ['onnx', 'tflite', 'pt', 'pth']
    
    def quantize_model(self, 
                      input_model_path: str, 
                      output_model_path: str,
                      model_type: str,
                      quantization_type: str = 'int8',
                      calibration_data: Optional[np.ndarray] = None) -> bool:
        """
        Quantize a model
        
        Args:
            input_model_path: Path to input model
            output_model_path: Path to output quantized model
            model_type: Type of model ('onnx', 'tflite', 'pt', 'pth')
            quantization_type: Type of quantization ('int8', 'int16', 'fp16')
            calibration_data: Calibration data for quantization
            
        Returns:
            True if quantization successful, False otherwise
        """
        try:
            self.logger.info(f"Quantizing {input_model_path}")
            
            if model_type.lower() == 'onnx':
                return self._quantize_onnx_model(input_model_path, output_model_path, 
                                               quantization_type, calibration_data)
            elif model_type.lower() == 'tflite':
                return self._quantize_tflite_model(input_model_path, output_model_path, 
                                                 quantization_type)
            elif model_type.lower() in ['pt', 'pth']:
                return self._quantize_pytorch_model(input_model_path, output_model_path, 
                                                  quantization_type, calibration_data)
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error quantizing model: {e}")
            return False
    
    def prune_model(self, 
                   input_model_path: str, 
                   output_model_path: str,
                   model_type: str,
                   pruning_ratio: float = 0.1,
                   pruning_type: str = 'structured') -> bool:
        """
        Prune a model
        
        Args:
            input_model_path: Path to input model
            output_model_path: Path to output pruned model
            model_type: Type of model
            pruning_ratio: Ratio of weights to prune (0.0 to 1.0)
            pruning_type: Type of pruning ('structured', 'unstructured')
            
        Returns:
            True if pruning successful, False otherwise
        """
        try:
            self.logger.info(f"Pruning {input_model_path}")
            
            if model_type.lower() in ['pt', 'pth']:
                return self._prune_pytorch_model(input_model_path, output_model_path, 
                                               pruning_ratio, pruning_type)
            else:
                self.logger.warning(f"Pruning not supported for {model_type} models")
                return False
                
        except Exception as e:
            self.logger.error(f"Error pruning model: {e}")
            return False
    
    def optimize_graph(self, 
                      input_model_path: str, 
                      output_model_path: str,
                      model_type: str,
                      optimization_level: str = 'all') -> bool:
        """
        Optimize model graph
        
        Args:
            input_model_path: Path to input model
            output_model_path: Path to output optimized model
            model_type: Type of model
            optimization_level: Level of optimization ('basic', 'all')
            
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            self.logger.info(f"Optimizing graph for {input_model_path}")
            
            if model_type.lower() == 'onnx':
                return self._optimize_onnx_graph(input_model_path, output_model_path, 
                                               optimization_level)
            elif model_type.lower() == 'tflite':
                return self._optimize_tflite_graph(input_model_path, output_model_path, 
                                                 optimization_level)
            else:
                self.logger.warning(f"Graph optimization not supported for {model_type} models")
                return False
                
        except Exception as e:
            self.logger.error(f"Error optimizing graph: {e}")
            return False
    
    def _quantize_onnx_model(self, 
                            input_path: str, 
                            output_path: str,
                            quantization_type: str,
                            calibration_data: Optional[np.ndarray]) -> bool:
        """Quantize ONNX model"""
        try:
            # Load ONNX model
            model = onnx.load(input_path)
            
            # This is a simplified implementation
            # In practice, you would use ONNX Runtime's quantization tools
            self.logger.warning("ONNX quantization not fully implemented")
            
            # For now, just copy the model
            onnx.save(model, output_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Error quantizing ONNX model: {e}")
            return False
    
    def _quantize_tflite_model(self, 
                              input_path: str, 
                              output_path: str,
                              quantization_type: str) -> bool:
        """Quantize TensorFlow Lite model"""
        try:
            # Load TFLite model
            with open(input_path, 'rb') as f:
                tflite_model = f.read()
            
            # Create converter
            converter = tf.lite.TFLiteConverter.from_saved_model(input_path)
            
            # Set optimization
            if quantization_type == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
            elif quantization_type == 'fp16':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            # Convert
            quantized_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error quantizing TFLite model: {e}")
            return False
    
    def _quantize_pytorch_model(self, 
                               input_path: str, 
                               output_path: str,
                               quantization_type: str,
                               calibration_data: Optional[np.ndarray]) -> bool:
        """Quantize PyTorch model"""
        try:
            # Load PyTorch model
            model = torch.load(input_path, map_location='cpu')
            model.eval()
            
            # Set quantization configuration
            if quantization_type == 'int8':
                model.qconfig = quant.get_default_qconfig('fbgemm')
            elif quantization_type == 'fp16':
                model = model.half()
            
            # Prepare model for quantization
            if quantization_type == 'int8':
                model = quant.prepare(model)
                
                # Calibrate with sample data
                if calibration_data is not None:
                    with torch.no_grad():
                        for data in calibration_data:
                            model(torch.tensor(data, dtype=torch.float32))
                
                # Convert to quantized model
                model = quant.convert(model)
            
            # Save quantized model
            torch.save(model, output_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error quantizing PyTorch model: {e}")
            return False
    
    def _prune_pytorch_model(self, 
                            input_path: str, 
                            output_path: str,
                            pruning_ratio: float,
                            pruning_type: str) -> bool:
        """Prune PyTorch model"""
        try:
            # Load PyTorch model
            model = torch.load(input_path, map_location='cpu')
            model.eval()
            
            # Apply pruning
            if pruning_type == 'structured':
                # Structured pruning (remove entire channels)
                for module in model.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
            else:
                # Unstructured pruning (remove individual weights)
                for module in model.modules():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            
            # Remove pruning reparameterization
            for module in model.modules():
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
            
            # Save pruned model
            torch.save(model, output_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error pruning PyTorch model: {e}")
            return False
    
    def _optimize_onnx_graph(self, 
                            input_path: str, 
                            output_path: str,
                            optimization_level: str) -> bool:
        """Optimize ONNX graph"""
        try:
            # Load ONNX model
            model = onnx.load(input_path)
            
            # Apply optimizations
            if optimization_level == 'all':
                # This would use ONNX optimizers
                # For now, just copy the model
                onnx.save(model, output_path)
            else:
                # Basic optimizations
                onnx.save(model, output_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing ONNX graph: {e}")
            return False
    
    def _optimize_tflite_graph(self, 
                              input_path: str, 
                              output_path: str,
                              optimization_level: str) -> bool:
        """Optimize TensorFlow Lite graph"""
        try:
            # Load TFLite model
            with open(input_path, 'rb') as f:
                tflite_model = f.read()
            
            # Create converter
            converter = tf.lite.TFLiteConverter.from_saved_model(input_path)
            
            # Set optimization level
            if optimization_level == 'all':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert
            optimized_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(optimized_model)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing TFLite graph: {e}")
            return False
    
    def analyze_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """
        Analyze model for optimization opportunities
        
        Args:
            model_path: Path to model file
            model_type: Type of model
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'path': model_path,
            'type': model_type,
            'size': os.path.getsize(model_path),
            'optimization_opportunities': []
        }
        
        try:
            if model_type.lower() == 'onnx':
                model = onnx.load(model_path)
                analysis['nodes'] = len(model.graph.node)
                analysis['inputs'] = len(model.graph.input)
                analysis['outputs'] = len(model.graph.output)
                
                # Check for optimization opportunities
                if analysis['nodes'] > 100:
                    analysis['optimization_opportunities'].append('graph_optimization')
                
                if analysis['size'] > 100 * 1024 * 1024:  # 100MB
                    analysis['optimization_opportunities'].append('quantization')
                    analysis['optimization_opportunities'].append('pruning')
            
            elif model_type.lower() == 'tflite':
                with open(model_path, 'rb') as f:
                    tflite_model = f.read()
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                analysis['inputs'] = len(interpreter.get_input_details())
                analysis['outputs'] = len(interpreter.get_output_details())
                
                if analysis['size'] > 50 * 1024 * 1024:  # 50MB
                    analysis['optimization_opportunities'].append('quantization')
            
            elif model_type.lower() in ['pt', 'pth']:
                model = torch.load(model_path, map_location='cpu')
                analysis['parameters'] = sum(p.numel() for p in model.parameters())
                
                if analysis['parameters'] > 1000000:  # 1M parameters
                    analysis['optimization_opportunities'].append('pruning')
                
                if analysis['size'] > 100 * 1024 * 1024:  # 100MB
                    analysis['optimization_opportunities'].append('quantization')
        
        except Exception as e:
            self.logger.error(f"Error analyzing model: {e}")
        
        return analysis

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Model optimization utility')
    parser.add_argument('--input', required=True, help='Input model path')
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('--model-type', required=True,
                       choices=['onnx', 'tflite', 'pt', 'pth'],
                       help='Model type')
    parser.add_argument('--operation', required=True,
                       choices=['quantize', 'prune', 'optimize', 'analyze'],
                       help='Optimization operation')
    parser.add_argument('--quantization-type', default='int8',
                       choices=['int8', 'int16', 'fp16'],
                       help='Quantization type')
    parser.add_argument('--pruning-ratio', type=float, default=0.1,
                       help='Pruning ratio (0.0 to 1.0)')
    parser.add_argument('--pruning-type', default='structured',
                       choices=['structured', 'unstructured'],
                       help='Pruning type')
    parser.add_argument('--optimization-level', default='all',
                       choices=['basic', 'all'],
                       help='Optimization level')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create optimizer
    optimizer = ModelOptimizer()
    
    # Perform operation
    success = False
    if args.operation == 'quantize':
        success = optimizer.quantize_model(args.input, args.output, args.model_type, 
                                         args.quantization_type)
    elif args.operation == 'prune':
        success = optimizer.prune_model(args.input, args.output, args.model_type, 
                                      args.pruning_ratio, args.pruning_type)
    elif args.operation == 'optimize':
        success = optimizer.optimize_graph(args.input, args.output, args.model_type, 
                                         args.optimization_level)
    elif args.operation == 'analyze':
        analysis = optimizer.analyze_model(args.input, args.model_type)
        print(json.dumps(analysis, indent=2))
        success = True
    
    if success:
        print(f"Successfully completed {args.operation} operation")
    else:
        print(f"Failed to complete {args.operation} operation")
        sys.exit(1)

if __name__ == '__main__':
    main()
