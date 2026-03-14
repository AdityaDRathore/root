# GSoC 2026 · CERN-HSF / ROOT — SOFIE Parser Evaluation

> **Branch:** `gsoc26-evaluation` · **Candidate:** Aditya Rathore · **GitHub:** [@AdityaDRathore](https://github.com/AdityaDRathore)
> **Project:** Improvements of ML Inference on PyTorch and Keras Models
> **Mentors:** Lorenzo Moneta · Sanjiban Sengupta (CERN / ROOT Core Team)
> **Submission Deadline:** 14 March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Was Implemented](#2-what-was-implemented)
3. [Repository Structure](#3-repository-structure)
4. [Build Instructions](#4-build-instructions)
5. [Execution Matrix](#5-execution-matrix)
6. [Architectural Justifications](#6-architectural-justifications)
7. [Upstream Contributions](#7-upstream-contributions)
8. [Testing Overview](#8-testing-overview)
9. [Engineering Reports](#9-engineering-reports)

---

## 1. Project Overview

This branch contains my evaluation submissions for the GSoC 2026 CERN-HSF project targeting `TMVA::Experimental::SOFIE` — the System for Optimized Fast Inference code Emission inside ROOT. SOFIE translates trained ML models (PyTorch, Keras, ONNX) into self-contained, dependency-free C++ inference code optimised for HEP analysis workflows.

The exercises required extending the SOFIE **PyTorch** and **Keras parsers** with new operator types, each demanding careful handling of memory layout, language-boundary safety, and C++ shape-inference responsibility delegation.

**Prior contributions on upstream ROOT (not on this branch):**
- **PR #21528** — Extended `RModelParser_PyTorch.cxx` with 10 new operators (Tanh, Softmax, LeakyRelu, Add, Sub, Mul, MatMul, Flatten, Reshape, BatchNormalization)
- **PR #21582** — CMake silent exclusion hotfix (`BLAS_FOUND OR use_gsl_cblas`)
- **Issue #21527** — Filed `UTILITY::Clean_name()` dot-erasure tensor collision bug
- **PR #20933, #20944, #21092, #20876** — ONNX operator additions (HardSigmoid, HardSwish, Softplus, GELU)
- **Issue #21071** — Filed `Softplus` numerical stability bug (PR #21092)

---

## 2. What Was Implemented

### Exercise 4 — PyTorch BatchNorm2D Parser

**Commit:** [`782b912`](https://github.com/AdityaDRathore/root/commit/782b9123f8f73711e4c2716ae5719590c6fcd521)
**Files:** `tmva/sofie_parsers/python/parse_batchnorm2d.py` · `test_parse_batchnorm2d.py`

Implements `parse_batchnorm2d_node(node, initializers, node_dict)` — a Python parser that extracts `onnx::BatchNormalization` nodes from PyTorch JIT IR traces into a SOFIE-compatible dictionary payload. Key design decisions: raw `σ²` is passed unchanged (epsilon pre-fusion prevention), `training_mode` is hardcoded to `0` (inference-only contract), `track_running_stats=False` is hard-rejected with a descriptive `ValueError` (silent physics corruption prevention), and all tensors are forced to C-contiguous `float32` before crossing the pybind11 boundary.

### Exercise 5 — Keras Conv2DTranspose Parser + NHWC→NCHW Memory Tracking

**Commit:** [`e366013`](https://github.com/AdityaDRathore/root/commit/e3660134568ef4e4b7957346f4fd988a2062e012)
**Files:** `tmva/sofie_parsers/python/parse_conv2dtranspose.py` · `sofie/test/TestRModelParserKeras.C` · `sofie/test/generateKerasModels.py` · `tmva/test_sofie_numeric_eval.py`

Implements `parse_conv2dtranspose_layer(layer, layout_tracker)` — a Keras parser that bridges channels-last (NHWC) weight storage to SOFIE's ONNX-compliant channels-first (NCHW) architecture. Introduces a `LayoutTracker` class for global layout state management, enforces kernel axes permutation `(3, 2, 0, 1)`, enforces C-buffer contiguity, defers spatial padding math to the C++ ShapeInference engine, and expands `TestRModelParserKeras.C` with true AST topology contract tests and NCHW numeric equivalence checks at `1e-5` tolerance.

---

## 3. Repository Structure

```
root/
├── tmva/
│   ├── sofie_parsers/
│   │   └── python/
│   │       ├── parse_batchnorm2d.py          # Exercise 4: PyTorch BatchNorm2D parser
│   │       ├── test_parse_batchnorm2d.py     # Exercise 4: unittest suite (3 tests)
│   │       └── parse_conv2dtranspose.py      # Exercise 5: Keras Conv2DTranspose parser
│   ├── sofie/
│   │    └── test/
│   │        ├── TestRModelParserKeras.C        # Exercise 5: GTest contract + numeric test
│   │        ├── generateKerasModels.py         # Exercise 5: model generation (Conv2DTranspose)
│   │        └── generatePyTorchModels.py       # Exercise 4: model generation (BatchNorm)
│   ├── test_sofie_numeric_eval.py              # Exercise 5: 4-phase Python numeric pipeline
├── README.md                                   # This file
├── GSoC2026_BatchNorm2D_Parser_AdityaRathore.pdf
└── GSoC2026_Conv2DTranspose_Parser_AdityaRathore.pdf
```

> **Note:** Parser files in `tmva/sofie_parsers/python/` are the canonical submission artifacts. The `sofie/test/` directory contains test harnesses and model generators that validate the parsers end-to-end through the full SOFIE C++ pipeline.

---

## 4. Build Instructions

### Prerequisites

```bash
# Required system packages
sudo apt-get install cmake python3-dev libgsl-dev

# Python dependencies
pip install numpy tensorflow torch torchvision
```

### CMake Configuration

```bash
# Clone and switch to evaluation branch
git clone https://github.com/AdityaDRathore/root.git
cd root
git checkout gsoc26-evaluation

# Create build directory
mkdir build && cd build

# Configure — SOFIE + Python bindings + GSL CBLAS fallback
# The (BLAS_FOUND OR use_gsl_cblas) fix (PR #21582) is required
# for the GTest targets to be included in the build manifest.
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dtmva=ON \
  -Dtmva-sofie=ON \
  -Dpyroot=ON \
  -Dpymva=ON \
  -Dbuiltin_gsl=ON \
  -DPYTHON_EXECUTABLE=$(which python3)

# Build — target the SOFIE test binary specifically
cmake --build . --target tmva-sofie-tests -- -j$(nproc)
```

> **Why `-Dbuiltin_gsl=ON`?** Without an external BLAS installation, ROOT falls back to GSL CBLAS. Before PR #21582, this caused `BLAS_FOUND=OFF` which silently excluded all SOFIE GTest targets from the build manifest. The upstream fix ensures `(BLAS_FOUND OR use_gsl_cblas)` correctly gates the test targets. If you have external BLAS (`libopenblas-dev`), you can omit `-Dbuiltin_gsl=ON`.

### Verify the Build

```bash
# Confirm GTest target was built (should not be empty)
ls bin/ | grep tmva-sofie-tests

# Confirm parser files are present
ls ../tmva/sofie_parsers/python/
# Expected: parse_batchnorm2d.py  parse_conv2dtranspose.py  test_parse_batchnorm2d.py
```

---

## 5. Execution Matrix

### 5.1 — Exercise 4: PyTorch BatchNorm2D Parser

#### Python Unit Tests (standalone, no ROOT build required)

```bash
cd tmva/sofie_parsers/python/

# Run the unittest suite (3 tests, ~0.066s)
python -m pytest test_parse_batchnorm2d.py -v
# or
python test_parse_batchnorm2d.py

# Expected output:
# test_standard_batchnorm ........................... PASS
# test_affine_false_batchnorm ....................... PASS
# test_track_running_stats_false_batchnorm ......... PASS
# Ran 3 tests in 0.066s — OK
```

#### Parser Self-Test (inline __main__ block)

```bash
cd tmva/sofie_parsers/python/
python parse_batchnorm2d.py

# Expected output:
# --- Testing Standard BatchNorm2D ---
# Type: BatchNorm2d | Features: 6 | Epsilon: 1e-05
# Scale is contiguous: True | Bias is contiguous: True
# --- Testing BatchNorm2D (track_running_stats=False) ---
# [EXPECTED ERROR CAUGHT]: SOFIE Inference Engine requires static weights...
# [SUCCESS] Memory Contiguity, PyTorch Dictionary formatting, and Edge Parsing Confirmed.
```

#### Generate PyTorch Test Model

```bash
cd tmva/sofie/test/
python generatePyTorchModels.py
# Produces: PyTorchModelBatchNorm.pt
```

---

### 5.2 — Exercise 5: Keras Conv2DTranspose Parser

#### Parser Self-Test (inline __main__ block)

```bash
cd tmva/sofie_parsers/python/
python parse_conv2dtranspose.py

# Expected output:
# --- Testing Keras Conv2DTranspose Memory Translations ---
# [LAYOUT TRACKER]
# Generated Prefixed Transpose: conv2d_transpose_layout_to_nchw | perm: [0, 3, 1, 2]
# New Global Layout Tracker State: NCHW
# [CONV_TRANSPOSE NODE]
# Node Type: ConvTranspose | Kernel Shape: [3, 3] | Strides: [2, 2]
# Transposed Kernel Memory Layout (In, Out, H, W): (3, 8, 3, 3) | MATCHED NCHW SOFIE SPEC.
# [SUCCESS] Memory Contiguity, NHWC Tracking, and Attribute Parsing Confirmed.
```

#### Four-Phase Numeric Equivalence Test (Python)

```bash
# Run from the tmva/ directory (parse_conv2dtranspose.py must be importable)
cd tmva/
PYTHONPATH=sofie_parsers/python python test_sofie_numeric_eval.py

# Expected output:
# [Phase 1] Keras Native Execution (NHWC)...
#   Keras Output Shape (NHWC): (1, 8, 8, 2)
# [Phase 2] PyMVA Extraction (NCHW Mappings)...
#   Extracted NCHW Weights Shape: (1, 2, 3, 3)
# [Phase 3] Simulated NCHW Vectorization Execution...
# [Phase 4] Evaluating AVX Vectorization Numeric Equivalence...
#   Maximum Float Discrepancy observed: 0.0
# [SUCCESS] Mathematical Pipeline verified.
```

#### Generate Keras Test Model

```bash
cd tmva/sofie/test/
python generateKerasModels.py
# Produces: KerasModelConv2DTranspose.keras (among others)
```

#### C++ GTest Contract + Numeric Test (requires ROOT build)

```bash
cd build/

# Run only the Conv2DTranspose GTest
./bin/tmva-sofie-tests --gtest_filter="RModelParser_Keras.CONV2D_TRANSPOSE"

# Expected output:
# [ RUN      ] RModelParser_Keras.CONV2D_TRANSPOSE
# [       OK ] RModelParser_Keras.CONV2D_TRANSPOSE (Xs)
#
# The test verifies:
#   1. operators.size() == 2
#   2. operators[0]->Type() == "Transpose"
#   3. operators[1]->Type() == "ConvTranspose"
#   4. |SOFIE_output[i] - Keras_NCHW_output[i]| <= 1e-5 for all i

# Run the full SOFIE Keras test suite
./bin/tmva-sofie-tests --gtest_filter="RModelParser_Keras.*"
```

---

## 6. Architectural Justifications

### 6.1 — NHWC → NCHW Memory Layout Alignment

Keras defaults to channels-last (NHWC) storage. SOFIE's C++ backend, following ONNX convention, requires channels-first (NCHW). The kernel weight tensor has a different axis order between the two frameworks:

| Framework | Kernel Layout | Axis Order |
|-----------|--------------|------------|
| Keras | `W_Keras ∈ R^(H×W×C_out×C_in)` | `(H, W, C_out, C_in)` |
| SOFIE/ONNX | `W_SOFIE ∈ R^(C_in×C_out×H×W)` | `(C_in, C_out, H, W)` |

The required axes permutation is `(3, 2, 0, 1)`. Applying `np.transpose` alone is insufficient — it returns a strided **view**, not a new contiguous buffer. Passing a strided pointer through the pybind11 bridge to a C++ `float*` causes a segmentation fault because the C++ side assumes `stride == sizeof(float)` per element.

**Resolution:** `np.ascontiguousarray(transposed_kernel, dtype=np.float32)` is mandatory after every transpose. This guarantees C-contiguous layout, `float32` element type, and is a no-op if already contiguous.

### 6.2 — Global Layout State Tracker

A naïve parser injects `Transpose(NHWC→NCHW)` before every Conv2DTranspose and `Transpose(NCHW→NHWC)` after it. In a 16-layer UNet decoder this doubles memory-copy overhead — an O(N) transpose tax.

The `LayoutTracker` maintains a single `current_layout` string. A Transpose node is injected **only when** the state is `"NHWC"`, after which it is permanently flipped to `"NCHW"`. A model with N Conv2DTranspose layers pays the transposition cost exactly once.

```python
if layout_tracker.current_layout == "NHWC":
    nodes.append({"type": "Transpose", "perm": [0, 3, 1, 2], ...})
    layout_tracker.current_layout = "NCHW"   # permanent — no revert
```

### 6.3 — Deferred Spatial Mathematics (ShapeInference Boundary)

Static padding computation in Python (`pad_h = max((H_in - 1) * S_h + kH - H_out, 0) // 2`) is incorrect. The padding required for `padding='same'` depends on runtime input tensor dimensions — not known statically at parse time. Hardcoding static padding overwrites the C++ ShapeInference calculation, breaking shape resolution for non-standard input sizes.

**Resolution:** For `padding='same'`, set `autopad='SAME_UPPER'` and omit static pad values entirely. The C++ engine resolves them dynamically. Similarly, `output_padding` is omitted (not defaulted to `[0, 0]`) when the Keras attribute is `None`.

### 6.4 — Dynamic State Trapping (BatchNorm2D)

When `track_running_stats=False`, PyTorch computes `μ` and `σ²` from the current mini-batch at runtime. At `batch_size=1` (common in ROOT HEP analysis): `μ = x`, `σ² = 0`, and the normalised output degenerates to `y = β` — the input signal is completely erased. A synthesised `mean=0, var=1` fallback would prevent a C++ crash but produce silently wrong physics results.

**Resolution:** Raise a descriptive `ValueError`. An inference engine must fail loudly rather than fail silently. Silent data corruption in physics analysis is infinitely worse than a hard abort.

### 6.5 — Epsilon Pre-Fusion Prevention (BatchNorm2D)

`ROperator_BatchNormalization::Initialize()` computes the fused scale as:

```cpp
fused_scale_data[i] = original_scale[i] / std::sqrt(original_var[i] + fepsilon);
```

Pre-injecting `ε` in Python before shipping `σ²` to the C++ side would result in `σ² + 2ε` under the square root — a measurable bias at HEP inference precision levels. The parser ships raw `σ²` unchanged and respects the arithmetic responsibility boundary.

---

## 7. Upstream Contributions

| Ref | Type | Description | Status |
|-----|------|-------------|--------|
| [PR #21528](https://github.com/root-project/root/pull/21528) | Feature | Extend `RModelParser_PyTorch.cxx` with 10 new operators | Open |
| [PR #21582](https://github.com/root-project/root/pull/21582) | Bugfix | CMake: `BLAS_FOUND OR use_gsl_cblas` — fix silent GTest exclusion on builtin-GSL nodes | Open |
| [Issue #21527](https://github.com/root-project/root/issues/21527) | Bug Report | `UTILITY::Clean_name()` erases dots causing tensor name collisions | Open |
| [PR #20933](https://github.com/root-project/root/pull/20933) | Feature | ONNX HardSigmoid operator | Open |
| [PR #20944](https://github.com/root-project/root/pull/20944) | Feature | ONNX HardSwish operator | Open |
| [PR #21092](https://github.com/root-project/root/pull/21092) | Bugfix | Softplus numerical stability | Open |
| [PR #20876](https://github.com/root-project/root/pull/20876) | Feature | Exact-CDF GELU operator | Open |
| [Issue #20171](https://github.com/root-project/root/issues/21071) | Improvement | Softplus numerical stability | Open |

### CMake Hotfix Detail (PR #21582)

The TMVA test configurations gated GTest targets on `if(BLAS_FOUND)`. When ROOT uses the builtin GSL CBLAS fallback (`-Dbuiltin_gsl=ON`), `use_gsl_cblas` is set but `BLAS_FOUND` remains `OFF` — causing the entire SOFIE execution test suite to be silently excluded from the build manifest while CI reported green.

```cmake
# Before (broken):
if (tpython AND ROOT_KERAS_FOUND AND BLAS_FOUND)

# After (PR #21582):
if (tpython AND ROOT_KERAS_FOUND AND (BLAS_FOUND OR use_gsl_cblas))
```

### Clean_name() Collision Detail (Issue #21527)

`UTILITY::Clean_name()` in `SOFIE_common.cxx:513` replaces `'-'→'_'` but **erases** dots. TorchScript generates sequential dotted names (`input.1`, `input1`). After `Clean_name()`, both map to `input1` — causing duplicate C++ variable declarations or silent tensor data overwrites.

```cpp
// Proposed fix (one line):
std::replace(s.begin(), s.end(), '.', '_');  // before the erase call
// input.1 → input_1  (distinct from input1)
```

Parser-level workaround is implemented in PR #21528: `.replace('.','_')` on the Python side before names reach `Clean_name()`.

---

## 8. Testing Overview

### Exercise 4 — BatchNorm2D

| Test | Harness | What It Validates | Result |
|------|---------|------------------|--------|
| `test_standard_batchnorm` | Python `unittest` | `type`, `num_features=6`, `eps=1e-5`, `momentum=0.9`, all 4 arrays C-contiguous `float32` | ✅ PASS |
| `test_affine_false_batchnorm` | Python `unittest` | Synthesised `weight==ones(6)`, `bias==zeros(6)` when `affine=False` | ✅ PASS |
| `test_track_running_stats_false` | Python `unittest` | `ValueError` raised with correct message when `track_running_stats=False` | ✅ PASS |

### Exercise 5 — Conv2DTranspose

| Test | Harness | What It Validates | Result |
|------|---------|------------------|--------|
| Layout state transition | Python `__main__` | Transpose node injected once; tracker flips to NCHW permanently | ✅ PASS |
| Kernel weight permutation | Python `__main__` | Output shape `(C_in, C_out, H, W)` correctly matches SOFIE spec | ✅ PASS |
| Memory contiguity | Python `__main__` | `weight.flags['C_CONTIGUOUS']` and `bias.flags['C_CONTIGUOUS']` both `True` | ✅ PASS |
| Padding attribute dispatch | Python `__main__` | `autopad='SAME_UPPER'` / `'VALID'` + `pads=[0,0,0,0]` correctly set | ✅ PASS |
| `output_padding` omission | Python `__main__` | Key absent from payload when Keras attribute is `None` | ✅ PASS |
| Four-phase numeric pipeline | `test_sofie_numeric_eval.py` | Max float discrepancy between NCHW simulation and Keras ground truth < `1e-5` | ✅ PASS |
| AST topology contract | `TestRModelParserKeras.C` (GTest) | `operators[0]->Type()=="Transpose"`, `operators[1]->Type()=="ConvTranspose"` | ✅ PASS |
| NCHW numeric equivalence | `TestRModelParserKeras.C` (GTest) | `|SOFIE[i] - Keras_NCHW[i]| <= 1e-5` for all output elements | ✅ PASS |

### Numeric Tolerance Rationale

The `1e-5` threshold is deliberate. Machine epsilon for `float32` is `≈ 1.19 × 10⁻⁷`. Due to summation tree reordering between NumPy (C backend) and C++ AVX/FMA instructions during fractionally-strided convolutions, `1e-5` accounts for FMA accumulation drift while remaining tight enough to immediately flag structural stride or padding miscalculations. A tolerance of `1e-3` was explicitly rejected — it masks single-pixel padding misalignments.

---

## 9. Engineering Reports

Detailed per-exercise technical write-ups are available in the repository:

| Document | Exercise | Contents |
|----------|----------|----------|
| [`GSoC2026_BatchNorm2D_Parser_AdityaRathore.pdf`](./GSoC2026_BatchNorm2D_Parser_AdityaRathore.pdf) | Exercise 4 | Mathematical model, design decisions (ε pre-fusion, `training_mode` hardcoding, `track_running_stats=False` hard reject, memory contiguity), implementation walkthrough, 4 bugs encountered and fixed, test coverage, lessons learned |
| [`GSoC2026_Conv2DTranspose_Parser_AdityaRathore.pdf`](./GSoC2026_Conv2DTranspose_Parser_AdityaRathore.pdf) | Exercise 5 | Operator mathematics, NHWC/NCHW layout analysis, `LayoutTracker` design, spatial deferral rationale, `List[Dict]` return contract, 5 bugs encountered and fixed (including `Clean_name()` Issue #21527 and CMake PR #21582), 4-phase numeric verification, true AST contract testing |

---

## Contact

**Aditya Rathore**
GSoC 2026 Candidate — SOFIE PyTorch/Keras Parser
GitHub: [@AdityaDRathore](https://github.com/AdityaDRathore)
Email: adityarathore7067@gmail.com

---

<div align="center">

*Submitted March 2026 · For review by Lorenzo Moneta & Sanjiban Sengupta · CERN-HSF / ROOT-TMVA-SOFIE*

</div>