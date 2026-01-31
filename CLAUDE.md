# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cuTile Python is a GPU programming language for NVIDIA GPUs that provides a high-level Python API for writing tile-based GPU kernels. It compiles Python code through a multi-stage pipeline (Python → HIR → IR → Bytecode → MLIR → GPU cubin) using the Tile IR compiler (`tileiras`).

**Requirements:** NVIDIA Blackwell GPU, Driver r580+, CUDA Toolkit 13.1+, Python 3.10-3.13

## Build Commands

```bash
# Development install (builds C++ extension via CMake)
pip install -e .

# Rebuild C++ extension after changes
make -C build

# Install test dependencies
pip install -r test/requirements.txt

# Install experimental features
pip install ./experimental
```

## Testing

```bash
# Run a specific test file
pytest test/test_copy.py

# Run all tests
pytest test/

# Run benchmarks
pytest test/bench_*.py
```

Test markers:
- `use_mlir` - Tests requiring internal MLIR extension

## Linting

```bash
# Python linting (max line length: 100)
flake8

# C++ linting
python scripts/cpplint.py

# Check inline samples are synchronized
python test/tools/inline_samples.py --check

# License header check (REUSE compliance)
scripts/check_license.sh
```

## Architecture

### Compilation Pipeline
```
@ct.kernel decorated Python function
    ↓
AST → HIR (ast2hir.py) - High-level IR close to Python
    ↓
HIR → IR (hir2ir.py) - Lowered intermediate representation
    ↓
Optimization passes (alias analysis, DCE, code motion, pattern rewriting)
    ↓
IR → Bytecode (ir2bytecode.py)
    ↓
Bytecode → MLIR (tileiras compiler)
    ↓
GPU cubin execution (C++ extension)
```

### Key Source Directories

- `src/cuda/tile/` - Main public API
  - `_execution.py` - `@kernel` and `@function` decorators
  - `_compile.py` - Compilation orchestrator
  - `_stub.py` - Type stubs with API documentation (53KB)
  - `_ir/` - IR data structures (ir.py, hir.py, ops.py, type.py)
  - `_passes/` - Optimization passes (~80 passes)
  - `_bytecode/` - Bytecode serialization
- `cext/` - C++ extension using CUDA Driver API
- `experimental/` - Unstable experimental features (autotuner)
- `samples/` - Example kernels (MatMul, FMHA, FFT, LayerNorm, etc.)

### Key Patterns

- **Thread-safe compilation:** Global `_compiler_lock` protects multi-stage compilation
- **Kernel specialization:** Kernels are cached and specialized based on runtime argument values
- **Immutable IR:** Passes create new IR structures rather than modifying in-place
- **Protocol-based typing:** Extensive use of `typing.Protocol` in `_stub.py` for IDE support

### Debug Environment Variables
- `CUDA_TILE_DUMP_BYTECODE` - Save bytecode to disk
- `CUDA_TILE_DUMP_TILEIR` - Save MLIR module
- `CUDA_TILE_COMPILER_TIMEOUT_SEC` - Compiler timeout (default 60s)

## Contributing

- Branch naming: `<username>/<type>-<name>` where type is `fea`/`enh`/`bug`
- Commits must be signed off (DCO): `git commit -s -m "message"`
- All files require SPDX license headers (Apache 2.0)
- PRs need labels: "breaking"/"non-breaking" + "feature"/"improvement"/"bugfix"/"documentation"
