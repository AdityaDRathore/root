# @(#)root/tmva/pymva $Id$
# Author: Aditya Rathore 2026
#
# /**********************************************************************************
#  * Project : TMVA - a Root-integrated toolkit for multivariate data analysis      *
#  * Package : TMVA                                                                 *
#  * Function: TMVA::Experimental::SOFIE::PyKeras::parse_conv2dtranspose_layer      *
#  *                                                                                *
#  * Description:                                                                   *
#  *      Parser function for extracting Keras Conv2DTranspose layer weights and    *
#  *      attributes, handling NHWC->NCHW layout tracking and contiguity guarantees.*
#  **********************************************************************************/

import numpy as np
from typing import Dict, List, Any

class LayoutTracker:
    """
    Global Layout State Tracker to prevent redundant Transpose.
    Maintains the current expected dimensional layout of the graph execution.
    """
    def __init__(self, initial_layout: str = "NHWC"):
        self.current_layout = initial_layout

def parse_conv2dtranspose_layer(layer, layout_tracker: LayoutTracker) -> List[Dict[str, Any]]:
    """
    Extracts the operator state for a Keras Conv2DTranspose layer, assuring NCHW 
    compliance for the SOFIE C++ Engine natively, generating an explicit Transpose 
    only if required.
    
    layer: Keras Conv2DTranspose layer instance
    layout_tracker: LayoutTracker object that maintains the current graph memory format
    """
    nodes = []
    
    # Resolve namespace formatting (strip dots for AST safety)
    layer_name_clean = layer.name.replace(".", "_")
    
    # 1. Topological Tensor Edge Extraction
    input_name = layer.input.name.replace(".", "_").replace("/", "_").replace(":", "_")
    output_name = layer.output.name.replace(".", "_").replace("/", "_").replace(":", "_")
    
    # 2. Layout State Tracking (NHWC vs NCHW)
    if layout_tracker.current_layout == "NHWC":
        transposed_input_name = f"{input_name}_transposed"
        # Inject an explicit Transpose node into the execution graph.
        # This transforms the input tensor `[N, H, W, C]` to `[N, C, H, W]`.
        nodes.append({
            "type": "Transpose",
            "name": f"{layer_name_clean}_layout_to_nchw",
            "perm": [0, 3, 1, 2], # NHWC -> NCHW
            "inputs": [input_name],
            "outputs": [transposed_input_name]
        })
        layout_tracker.current_layout = "NCHW"
        conv_input_name = transposed_input_name
    else:
        conv_input_name = input_name
        
    # 2. Kernel Weight Contiguity and Transposition
    weights = layer.get_weights()
    if len(weights) == 0:
        raise ValueError(f"SOFIE Inference Engine requires static weights. Node: {layer.name}")
        
    keras_kernel = weights[0]
    # Keras kernel: (H, W, OutChannels, InChannels)
    # SOFIE kernel reqs: (InChannels, OutChannels, H, W)
    # Axes mapping to achieve this: 3 -> 0, 2 -> 1, 0 -> 2, 1 -> 3
    transposed_kernel = np.transpose(keras_kernel, (3, 2, 0, 1))
    
    # Force C-buffer Contiguity for AVX/SIMD vectorization safety downstream
    contiguous_kernel = np.ascontiguousarray(transposed_kernel, dtype=np.float32)
    
    # 4. Attribute Extraction
    parsed = {
        "type": "ConvTranspose",
        "name": layer_name_clean,
        "kernel_shape": list(layer.kernel_size),
        "strides": list(layer.strides),
        "dilations": list(layer.dilation_rate),
        "group": 1,
        "weight": contiguous_kernel,
        "inputs": [conv_input_name],
        "outputs": [output_name]
    }
    
    if layer.use_bias and len(weights) > 1:
        # Bias is strictly a 1D vector of shape (OutChannels,)
        parsed["bias"] = np.ascontiguousarray(weights[1], dtype=np.float32)
        
    if layer.padding == "valid":
        parsed["autopad"] = "VALID"
        parsed["pads"] = [0, 0, 0, 0]
    elif layer.padding == "same":
        parsed["autopad"] = "SAME_UPPER"
        
    # 4. output_padding handling (Conditional Packing per Directive)
    # If the Keras architect explicitly overrides output_padding, we forward it.
    # Otherwise omit, permitting SOFIE's C++ ShapeInference to establish dimensions.
    if hasattr(layer, "output_padding") and layer.output_padding is not None:
        if isinstance(layer.output_padding, int):
            parsed["output_padding"] = [layer.output_padding] * 2
        else:
            parsed["output_padding"] = list(layer.output_padding)
        
    nodes.append(parsed)
    return nodes

if __name__ == "__main__":
    import os
    # Suppress tf warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2DTranspose
    from tensorflow.keras.models import Sequential
    
    def test_conv2dtranspose_parsing():
        print("\n--- Testing Keras Conv2DTranspose Memory Translations ---")
        
        # Build a frozen inference model
        model = Sequential([
            tf.keras.Input(shape=(4, 4, 3)), # Batch=None, H=4, W=4, C=3 (NHWC default)
            Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)
        ])
        
        # Freezing graph behavior ensuring inference mode predictability
        model.trainable = False
        
        layer = model.layers[0]
        tracker = LayoutTracker("NHWC")
        parsed_nodes = parse_conv2dtranspose_layer(layer, tracker)
        
        # Verify node sequencing
        assert len(parsed_nodes) == 2, "Failed to inject Transpose node correctly from initial NHWC state"
        trans_node = parsed_nodes[0]
        conv_node = parsed_nodes[1]
        
        assert trans_node['type'] == 'Transpose'
        assert trans_node['perm'] == [0, 3, 1, 2]
        assert tracker.current_layout == "NCHW"
        
        print(f"\n[LAYOUT TRACKER]")
        print(f"Generated Prefixed Transpose: {trans_node['name']} | perm: {trans_node['perm']}")
        print(f"New Global Layout Tracker State: {tracker.current_layout}")
        
        print(f"\n[CONV_TRANSPOSE NODE]")
        print(f"Node Type: {conv_node['type']}")
        print(f"Clean Name: {conv_node['name']}")
        print(f"Inputs Routed: {conv_node['inputs']}")
        print(f"Outputs Routed: {conv_node['outputs']}")
        print(f"Kernel Shape Extracted: {conv_node['kernel_shape']}")
        print(f"Strides Extracted: {conv_node['strides']}")
        print(f"Dilations Extracted: {conv_node['dilations']}")
        print(f"Autopad: {conv_node['autopad']}")
        
        # Contiguity and Vectorization checks
        assert conv_node['weight'].flags['C_CONTIGUOUS'], "CRITICAL: Kernel matrix is non-contiguous, AVX vectorization fault imminent!"
        if 'bias' in conv_node:
            assert conv_node['bias'].flags['C_CONTIGUOUS'], "CRITICAL: Bias vector is non-contiguous."
            
        print("\n[VECTORIZATION MEMORY CHECKS]")
        print(f"Transposed Kernel memory Contiguous (C-buffer): {conv_node['weight'].flags['C_CONTIGUOUS']}")
        
        # Check actual shapes vs expected SOFIE (InChannels, OutChannels, H, W)
        # Keras layer inputs: InChannels=3, OutChannels=8, H=3, W=3. Result = (3, 8, 3, 3)
        assert conv_node['weight'].shape == (3, 8, 3, 3), f"Shape distortion during dimensional mapping! Got {conv_node['weight'].shape}"
        print(f"Transposed Kernel Memory Layout (In, Out, H, W): {conv_node['weight'].shape} | MATCHED NCHW SOFIE SPEC.")
        
        if 'output_padding' in conv_node:
            print(f"Output Padding Packaged: {conv_node['output_padding']}")
        else:
            print("Output Padding dynamically suppressed (Deferring to C++ ShapeInference Engine).")
            
        print("\n[SUCCESS] Memory Contiguity, NHWC Tracking, and Attribute Parsing Confirmed.")
        
    test_conv2dtranspose_parsing()
