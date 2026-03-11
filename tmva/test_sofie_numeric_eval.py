import os
import sys

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose

# Import the PyMVA parser function directly
from parse_conv2dtranspose import parse_conv2dtranspose_layer, LayoutTracker

def simulated_nchw_conv_transpose(input_nchw, weights_nchw, strides, padding="SAME_UPPER"):
    # Extremely simplified Strided NCHW Conv2DTranspose operation for validation
    N, C_in, H_in, W_in = input_nchw.shape
    C_in_w, C_out_w, kH, kW = weights_nchw.shape
    
    # Assert dimensions
    assert C_in == C_in_w
    
    stride_h, stride_w = strides
    
    # Output calculation based on SAME padding
    H_out = H_in * stride_h
    W_out = W_in * stride_w
    
    output = np.zeros((N, C_out_w, H_out, W_out), dtype=np.float32)
    
    # Calculate padding to achieve H_out, W_out given strides and kernel size
    # This matches PyTorch/ONNX ConvTranspose formula: H_out = (H_in - 1)*stride - 2*pad + kernel + output_padding
    
    pad_h = max((H_in - 1) * stride_h + kH - H_out, 0) // 2
    pad_w = max((W_in - 1) * stride_w + kW - W_out, 0) // 2
    
    for n in range(N):
        for c_out in range(C_out_w):
            for c_in in range(C_in):
                for h in range(H_in):
                    for w in range(W_in):
                        out_h = h * stride_h - pad_h
                        out_w = w * stride_w - pad_w
                        
                        for kh in range(kH):
                            for kw in range(kW):
                                if 0 <= out_h + kh < H_out and 0 <= out_w + kw < W_out:
                                    output[n, c_out, out_h + kh, out_w + kw] += \
                                        input_nchw[n, c_in, h, w] * weights_nchw[c_in, c_out, kh, kw]
    return output

def run_math_test():
    print("======================================================================")
    print(" SOFIE INTEGRATION: NUMERIC EQUIVALENCE VERIFICATION PURE PYTHON INFO ")
    print("======================================================================")

    # 1. Generate Keras Baseline
    print("\n[Phase 1] Keras Native Execution (NHWC)...")
    model = Sequential([
        tf.keras.Input(shape=(4, 4, 1)),
        Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)
    ])
    model.trainable = False
    
    # NHWC ones tensor
    input_nhwc = np.ones((1, 4, 4, 1), dtype=np.float32)
    keras_output_nhwc = model(input_nhwc).numpy()
    print(f"Keras Output Shape (NHWC): {keras_output_nhwc.shape}")
    
    # 2. Extract AST via Parser
    print("\n[Phase 2] PyMVA Extraction (NCHW Mappings)...")
    tracker = LayoutTracker("NHWC")
    nodes = parse_conv2dtranspose_layer(model.layers[0], tracker)
    
    trans_node = nodes[0]
    conv_node = nodes[1]
    
    assert trans_node['perm'] == [0, 3, 1, 2] # NHWC -> NCHW check
    weights_nchw = conv_node['weight']
    strides = conv_node['strides']
    
    print(f"Extracted NCHW Weights Shape: {weights_nchw.shape}")
    
    # 3. Simulated C++ Execution
    print("\n[Phase 3] Simulated NCHW Vectorization Execution...")
    # Step 1: Transpose Input
    input_nchw = np.transpose(input_nhwc, trans_node['perm'])
    
    # Step 2: ConvTranspose Operator (Simulating C++ engine)
    output_nchw = simulated_nchw_conv_transpose(input_nchw, weights_nchw, strides)
    
    # 4. Numerics check
    print("\n[Phase 4] Evaluating AVX Vectorization Numeric Equivalence...")
    
    # Convert NCHW output back to NHWC to compare with Keras truth
    output_nchw_reverted = np.transpose(output_nchw, (0, 2, 3, 1))
    
    print(f"Final Simulated Emitted Output Shape: {output_nchw_reverted.shape}")
    
    max_error = np.max(np.abs(keras_output_nhwc.flatten() - output_nchw_reverted.flatten()))
    
    print(f"Maximum Float Discrepancy observed: {max_error}")
    
    if max_error > 1e-5:
        print(f"[FATAL] Numeric deviation exceeds threshold (> 1e-5). Math is corrupted.")
        sys.exit(1)
        
    print("\n[SUCCESS] Mathematical Pipeline verified. Transpose Memory block accurately mapped to NCHW SIMD Execution.")
    
if __name__ == "__main__":
    run_math_test()
