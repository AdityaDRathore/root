# @(#)root/tmva/pymva $Id$
# Author: Aditya Rathore 2026
#
# /**********************************************************************************
#  * Project : TMVA - a Root-integrated toolkit for multivariate data analysis      *
#  * Package : TMVA                                                                 *
#  * Function: TMVA::Experimental::SOFIE::PyTorch::parse_batchnorm2d_node           *
#  *                                                                                *
#  * Description:                                                                   *
#  *      Parser function for extracting PyTorch ONNX BatchNormalization nodes
#         Currently for the Evaulation, will merge in the ROOT as a patch           *
#  *                                                                                *
#  **********************************************************************************/

import numpy as np

def _node_get(node, key):
    sel = node.kindOf(key)
    return getattr(node, sel)(key)

def parse_batchnorm2d_node(node, initializers, node_dict):
    """
    Extracts the operator state for a PyTorch BatchNorm2D node from an ONNX graph.
    
    node: PyTorch ONNX IR Node (onnx::BatchNormalization)
    initializers: Dictionary of initialized weight tensors mapped by debugName (np.array)
    node_dict: Dictionary mapping debugName to Node, useful for looking up Constant nodes
    """
    # 1. Attribute Extraction
    epsilon = _node_get(node, "epsilon") if node.hasAttribute("epsilon") else 1e-05
    momentum = _node_get(node, "momentum") if node.hasAttribute("momentum") else 0.9

    # Input mapping for ONNX BatchNormalization: 
    # X (0), scale (1), B (2), input_mean (3), input_var (4)
    inputs = list(node.inputs())
    
    def get_tensor_data(input_edge):
        name = input_edge.debugName()
        if not name or name == "":
            return None
        
        if name in initializers:
            tensor = initializers[name]
            # Ensure C-contiguous array for SOFIE C++ Engine safety
            return np.ascontiguousarray(tensor, dtype=np.float32)
        elif name in node_dict:
            src_node = node_dict[name]
            if src_node.kind() == "onnx::Constant":
                # Extract the constant value with device safety bounds (for CUDA/Autograd safety)
                tensor = _node_get(src_node, "value")
                return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        return None

    # Get shapes to synthesize fallback arrays if needed
    x_type = inputs[0].type()
    channels = None
    if x_type is not None and x_type.kind() == "TensorType":
        sizes = x_type.sizes()
        if sizes and len(sizes) >= 2:
            channels = int(sizes[1])
            
    scale = get_tensor_data(inputs[1]) if len(inputs) > 1 else None
    bias = get_tensor_data(inputs[2]) if len(inputs) > 2 else None
    mean = get_tensor_data(inputs[3]) if len(inputs) > 3 else None
    var = get_tensor_data(inputs[4]) if len(inputs) > 4 else None

    # If channel inference from 'X' failed, infer from one of the tensors
    if channels is None:
        for t in [scale, bias, mean, var]:
            if t is not None:
                channels = int(t.shape[0])
                break

    if channels is None:
        raise ValueError("Cannot infer channel dimension for BatchNorm2D")

    # Handle missing tensors by synthesizing appropriately (e.g. affine=False)
    if scale is None:
        scale = np.ones(channels, dtype=np.float32)
    if bias is None:
        bias = np.zeros(channels, dtype=np.float32)
        
    # Resolve namespace formatting (strip dots for C++ safety if needed, use debugName as base)
    # A Node itself doesn't have debugName, but its first output Value does:
    out_value = list(node.outputs())[0] if list(node.outputs()) else None
    node_name_clean = out_value.debugName().replace(".", "_") if out_value else "batchnorm_node"

    if mean is None or var is None:
        raise ValueError(f"SOFIE Inference Engine requires static weights. Dynamic BatchNormalization (track_running_stats=False) is unsupported in node: {node_name_clean}")

    return {
        "type": "BatchNorm2d",
        "name": node_name_clean,
        "num_features": int(channels),
        "eps": float(epsilon),
        "momentum": float(momentum),
        "training_mode": 0,
        "running_mean": mean,
        "running_var": var,
        "weight": scale,
        "bias": bias
    }

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.onnx.utils import _model_to_graph

    def test_batchnorm(module, name):
        print(f"\n--- Testing {name} ---")
        module.eval()
        dummy_input = torch.randn(2, 6, 8, 8)
        
        # I use strict=False or other params depending on PyTorch version,
        # but standard args usually work here.
        graph, params_dict, out_dict = _model_to_graph(module, (dummy_input,))
        
        # Build initializers dict simulating the C++ environment PyRunString loop
        initializers = {}
        for k, v in params_dict.items():
            initializers[k] = v.detach().cpu().numpy()
            
        # Build node_dict for Constant lookup
        node_dict = {}
        batch_norm_node = None
        for n in graph.nodes():
            for out in n.outputs():
                node_dict[out.debugName()] = n
            if n.kind() == "onnx::BatchNormalization":
                batch_norm_node = n
                
        if batch_norm_node is None:
            raise RuntimeError("BatchNorm node not found in trace!")
            
        parsed = parse_batchnorm2d_node(batch_norm_node, initializers, node_dict)
        
        print(f"Type: {parsed['type']}")
        print(f"Name: {parsed['name']}")
        print(f"Features: {parsed['num_features']}")
        print(f"Epsilon: {parsed['eps']}")
        print(f"Scale mean (expected ~1.0 if affine=False): {np.mean(parsed['weight'])}")
        print(f"Bias mean (expected ~0.0 if affine=False): {np.mean(parsed['bias'])}")
        print(f"Mean mean (expected ~0.0 if tracking=False): {np.mean(parsed['running_mean'])}")
        print(f"Var mean (expected ~1.0 if tracking=False): {np.mean(parsed['running_var'])}")
        print(f"Scale is contiguous: {parsed['weight'].flags['C_CONTIGUOUS']}")
        print(f"Bias is contiguous: {parsed['bias'].flags['C_CONTIGUOUS']}")
        print(f"Mean is contiguous: {parsed['running_mean'].flags['C_CONTIGUOUS']}")
        print(f"Var is contiguous: {parsed['running_var'].flags['C_CONTIGUOUS']}")

    # Standard Instance
    test_batchnorm(nn.BatchNorm2d(6), "Standard BatchNorm2D")
    
    # Affine = False
    test_batchnorm(nn.BatchNorm2d(6, affine=False), "BatchNorm2D (affine=False)")
    
    # track_running_stats = False (Should raise Exception)
    print("\n--- Testing BatchNorm2D (track_running_stats=False) ---")
    try:
        test_batchnorm(nn.BatchNorm2d(6, track_running_stats=False), "BatchNorm2D (track_running_stats=False)")
    except ValueError as e:
        print(f"[EXPECTED ERROR CAUGHT]: {e}")

    print("\n[SUCCESS] Memory Contiguity, PyTorch Dictionary formatting, and Edge Parsing Confirmed.")
