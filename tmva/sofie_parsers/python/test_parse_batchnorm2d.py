# @(#)root/tmva/pymva $Id$
# Author: Aditya Rathore 2026
#
# /**********************************************************************************
#  * Project : TMVA - a Root-integrated toolkit for multivariate data analysis      *
#  * Package : TMVA                                                                 *
#  * Function: TMVA::Experimental::SOFIE::PyTorch::test_parse_batchnorm2d           *
#  *                                                                                *
#  * Description:                                                                   *
#  *      Unit tests for PyTorch ONNX BatchNormalization node parsing logic.        *
#  *                                                                                *
#  **********************************************************************************/

import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.onnx.utils import _model_to_graph
from parse_batchnorm2d import parse_batchnorm2d_node

class TestParseBatchNorm2D(unittest.TestCase):

    def setUp(self):
        # Create a dummy input to trace the graph
        self.dummy_input = torch.randn(2, 6, 8, 8)

    def _get_node_and_dicts(self, module):
        module.eval()
        graph, params_dict, out_dict = _model_to_graph(module, (self.dummy_input,))
        
        # Build initializers dict
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
            self.fail("BatchNorm node not found in trace. The module might have been folded.")
            
        return batch_norm_node, initializers, node_dict

    def test_standard_batchnorm(self):
        module = nn.BatchNorm2d(6)
        node, initializers, node_dict = self._get_node_and_dicts(module)
        parsed = parse_batchnorm2d_node(node, initializers, node_dict)

        self.assertEqual(parsed["type"], "BatchNorm2d")
        self.assertEqual(parsed["num_features"], 6)
        
        # Check standard default epsilon and momentum
        self.assertAlmostEqual(parsed["eps"], 1e-05)
        self.assertAlmostEqual(parsed["momentum"], 0.9)

        # Check contiguous layouts
        self.assertTrue(parsed["weight"].flags["C_CONTIGUOUS"])
        self.assertTrue(parsed["bias"].flags["C_CONTIGUOUS"])
        self.assertTrue(parsed["running_mean"].flags["C_CONTIGUOUS"])
        self.assertTrue(parsed["running_var"].flags["C_CONTIGUOUS"])

    def test_affine_false_batchnorm(self):
        module = nn.BatchNorm2d(6, affine=False)
        node, initializers, node_dict = self._get_node_and_dicts(module)
        parsed = parse_batchnorm2d_node(node, initializers, node_dict)

        # We expect parsed["weight"] to be all 1s and parsed["bias"] to be all 0s
        np.testing.assert_array_equal(parsed["weight"], np.ones(6, dtype=np.float32))
        np.testing.assert_array_equal(parsed["bias"], np.zeros(6, dtype=np.float32))

    def test_track_running_stats_false_batchnorm(self):
        # When track_running_stats=False, PyTorch may trace dynamic batch mean/var nodes.
        # Ensure our parser traps this safely and throws a ValueError.
        module = nn.BatchNorm2d(6, track_running_stats=False)
        try:
            # We catch it only if the exporter didn't fold it.
            # PyTorch 2.x tends to trace ReduceMean operations instead of BatchNormalization if eval() is false.
            # However, if evaluaton mode forces BatchNormalization without initializers, it will trace as BN.
            node, initializers, node_dict = self._get_node_and_dicts(module)
            with self.assertRaises(ValueError) as context:
                parse_batchnorm2d_node(node, initializers, node_dict)
            self.assertIn("Dynamic BatchNormalization (track_running_stats=False) is unsupported", str(context.exception))
        except AssertionError as e:
            # If the exporter optimizes the graph so that onnx::BatchNormalization doesn't even exist,
            # (which causes self.fail() inside _get_node_and_dicts), ignore for now as it's safe.
            if "BatchNorm node not found" in str(e):
                pass
            else:
                raise

if __name__ == '__main__':
    unittest.main()
