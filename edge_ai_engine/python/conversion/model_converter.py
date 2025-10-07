#!/usr/bin/env python3
"""
Model conversion utilities for Edge AI Engine
Author: AI Co-Developer
Date: 2024

This module provides utilities for converting models between different formats
and optimizing them for edge deployment.
"""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import torch
    import torch.jit
    import numpy as np
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Please install required packages: pip install onnx onnxruntime tensorflow torch")

logger = logging.getLogger(__name__)

class ModelConverter:
    """Model conversion utility class"""
    
    def __init__(self):
        self.supported_formats = ['onnx', 'tflite', 'pt', 'pth']
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def convert_to_onnx(self, 
                       input_model_path: str, 
                       output_model_path: str,
                       input_shape: Optional[List[int]] = None,
                       opset_version: int = 11) -> bool:
        """
        Convert a model to ONNX format
        
        Args:
            input_model_path: Path to input model
            output_model_path: Path to output ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            self.logger.info(f"Converting {input_model_path} to ONNX format")
            
            # Determine input format and convert
            input_ext = Path(input_model_path).suffix.lower()
            
            if input_ext == '.pt' or input_ext == '.pth':
                return self._pytorch_to_onnx(input_model_path, output_model_path, 
                                           input_shape, opset_version)
            elif input_ext == '.pb':
                return self._tensorflow_to_onnx(input_model_path, output_model_path, 
                                              input_shape, opset_version)
            else:
                self.logger.error(f"Unsupported input format: {input_ext}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error converting to ONNX: {e}")
            return False
    
    def convert_to_tflite(self, 
                         input_model_path: str, 
                         output_model_path: str,
                         quantization: bool = False) -> bool:
        """
        Convert a model to TensorFlow Lite format
        
        Args:
            input_model_path: Path to input model
            output_model_path: Path to output TFLite model
            quantization: Enable quantization
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            self.logger.info(f"Converting {input_model_path} to TFLite format")
            
            # Load TensorFlow model
            model = tf.keras.models.load_model(input_model_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantization:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save model
            with open(output_model_path, 'wb') as f:
                f.write(tflite_model)
            
            self.logger.info(f"Successfully converted to TFLite: {output_model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting to TFLite: {e}")
            return False
    
    def convert_to_pytorch_mobile(self, 
                                 input_model_path: str, 
                                 output_model_path: str,
                                 optimize_for_mobile: bool = True) -> bool:
        """
        Convert a model to PyTorch Mobile format
        
        Args:
            input_model_path: Path to input model
            output_model_path: Path to output PyTorch Mobile model
            optimize_for_mobile: Enable mobile optimization
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            self.logger.info(f"Converting {input_model_path} to PyTorch Mobile format")
            
            # Load PyTorch model
            model = torch.load(input_model_path, map_location='cpu')
            model.eval()
            
            # Create example input
            example_input = torch.randn(1, 3, 224, 224)  # Default input shape
            
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            
            if optimize_for_mobile:
                traced_model = torch.jit.optimize_for_mobile(traced_model)
            
            # Save model
            traced_model.save(output_model_path)
            
            self.logger.info(f"Successfully converted to PyTorch Mobile: {output_model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting to PyTorch Mobile: {e}")
            return False
    
    def _pytorch_to_onnx(self, 
                        input_path: str, 
                        output_path: str,
                        input_shape: Optional[List[int]] = None,
                        opset_version: int = 11) -> bool:
        """Convert PyTorch model to ONNX"""
        try:
            # Load PyTorch model
            model = torch.load(input_path, map_location='cpu')
            model.eval()
            
            # Create example input
            if input_shape is None:
                input_shape = [1, 3, 224, 224]  # Default shape
            
            example_input = torch.randn(*input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                example_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting PyTorch to ONNX: {e}")
            return False
    
    def _tensorflow_to_onnx(self, 
                           input_path: str, 
                           output_path: str,
                           input_shape: Optional[List[int]] = None) -> bool:
        """Convert TensorFlow model to ONNX"""
        try:
            # This would require tf2onnx library
            # For now, return False as it's not implemented
            self.logger.warning("TensorFlow to ONNX conversion not implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Error converting TensorFlow to ONNX: {e}")
            return False
    
    def validate_model(self, model_path: str, model_type: str) -> bool:
        """
        Validate a model file
        
        Args:
            model_path: Path to model file
            model_type: Type of model ('onnx', 'tflite', 'pt', 'pth')
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file does not exist: {model_path}")
                return False
            
            if model_type.lower() == 'onnx':
                model = onnx.load(model_path)
                onnx.checker.check_model(model)
                return True
                
            elif model_type.lower() == 'tflite':
                with open(model_path, 'rb') as f:
                    tflite_model = f.read()
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                interpreter.allocate_tensors()
                return True
                
            elif model_type.lower() in ['pt', 'pth']:
                model = torch.load(model_path, map_location='cpu')
                return True
                
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return False
    
    def get_model_info(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model_path: Path to model file
            model_type: Type of model
            
        Returns:
            Dictionary containing model information
        """
        info = {
            'path': model_path,
            'type': model_type,
            'size': os.path.getsize(model_path),
            'valid': False
        }
        
        try:
            if model_type.lower() == 'onnx':
                model = onnx.load(model_path)
                info['inputs'] = [input.name for input in model.graph.input]
                info['outputs'] = [output.name for output in model.graph.output]
                info['valid'] = True
                
            elif model_type.lower() == 'tflite':
                with open(model_path, 'rb') as f:
                    tflite_model = f.read()
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                info['inputs'] = interpreter.get_input_details()
                info['outputs'] = interpreter.get_output_details()
                info['valid'] = True
                
            elif model_type.lower() in ['pt', 'pth']:
                model = torch.load(model_path, map_location='cpu')
                info['valid'] = True
                
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
        
        return info

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Model conversion utility')
    parser.add_argument('--input', required=True, help='Input model path')
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('--input-format', required=True, 
                       choices=['onnx', 'tflite', 'pt', 'pth'],
                       help='Input model format')
    parser.add_argument('--output-format', required=True,
                       choices=['onnx', 'tflite', 'pt', 'pth'],
                       help='Output model format')
    parser.add_argument('--quantization', action='store_true',
                       help='Enable quantization')
    parser.add_argument('--input-shape', nargs='+', type=int,
                       help='Input tensor shape')
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output model')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create converter
    converter = ModelConverter()
    
    # Perform conversion
    success = False
    if args.output_format == 'onnx':
        success = converter.convert_to_onnx(args.input, args.output, 
                                          args.input_shape, args.opset_version)
    elif args.output_format == 'tflite':
        success = converter.convert_to_tflite(args.input, args.output, 
                                            args.quantization)
    elif args.output_format in ['pt', 'pth']:
        success = converter.convert_to_pytorch_mobile(args.input, args.output)
    
    if success:
        print(f"Successfully converted {args.input} to {args.output}")
        
        if args.validate:
            if converter.validate_model(args.output, args.output_format):
                print("Output model validation passed")
            else:
                print("Output model validation failed")
                sys.exit(1)
    else:
        print(f"Failed to convert {args.input} to {args.output}")
        sys.exit(1)

if __name__ == '__main__':
    main()
