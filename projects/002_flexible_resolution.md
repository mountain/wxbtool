# Project 002: Flexible Resolution Support - Eliminating Hard-coded Spatial Dimensions

**Status**: Draft  
**Created**: 2025-01-18  
**Author**: AI Assistant  
**Type**: Architecture Decision Record & Technical Specification

## Executive Summary

This project addresses a critical code quality issue in wxbtool: hard-coded spatial dimensions (32×64) are scattered throughout the codebase, making it difficult to support multiple resolutions and violating DRY principles. This document proposes a systematic refactoring to centralize these dimensions in the Settings system.

## Problem Statement

### Current Issues

Through comprehensive analysis of the wxbtool codebase, we identified **122 instances** in source code and **39 instances** in test files where the dimensions `32` and `64` are hard-coded. These represent spatial grid dimensions for 5.625° resolution data (32 latitude points × 64 longitude points).

### Affected Areas

#### 1. Model Specifications (`wxbtool/specs/res5_625/`)
- **Files**: `t850weyn.py`, `t850rasp.py`, `t850recur.py`, `z500weyn.py`
- **Issues**: 
  - Hard-coded `.view(-1, span, height, 32, 64)` calls
  - Fixed reshape operations: `rst.view(-1, 1, 32, 64)`
  - Tensor slicing with magic numbers: `input[:, :, :, 63:64]`

**Example from `t850weyn.py`:**
```python
z500 = norm_z500(
    kwargs["geopotential"].view(
        -1, self.setting.input_span, self.setting.height, 32, 64  # ← Hard-coded
    )[:, :, self.setting.levels.index("500")]
)
```

#### 2. Neural Network Models (`wxbtool/zoo/res5_625/unet/`)
- **Files**: `t850d3sm_weyn.py`, `t850d3bg_weyn.py`, `t850d3hg_weyn.py`, `t850d3sm_rasp.py`
- **Issues**:
  - Hard-coded UNet spatial configuration: `spatial=(32, 64 + 2)`
  - Boundary padding logic: `input[:, :, :, 63:64]` and `input[:, :, :, 0:1]`

#### 3. Core Infrastructure (`wxbtool/nn/`, `wxbtool/util/`)
- **Files**: `model.py`, `lightning.py`, `infer.py`, `eval.py`, `plotter.py`
- **Issues**:
  - Grid generation: `np.meshgrid(np.linspace(0, 1, num=32), np.linspace(0, 1, num=64))`
  - Coordinate arrays: `"lat": np.linspace(87.1875, -87.1875, 32)`
  - Data reshaping: `results[var].reshape(-1, 32, 64)`

#### 4. Data Generation (`wxbtool/datagen/`)
- **Files**: `gen_climate.py`, `util.py`
- **Issues**:
  - Latitude/longitude grid generation with magic numbers
  - Image generation with fixed dimensions: `image_size=(32, 64)`

#### 5. Test Suite (`tests/`)
- **Files**: All test model files and `spec.py`
- **Issues**:
  - Hard-coded tensor shapes: `np.zeros((1, span, 32, 64), dtype=np.float32)`
  - Fixed output dimensions in model tests

#### 6. Command Line Interface (`wxbtool/wxb.py`)
- **Issue**: Default batch size hard-coded as 64 (related but separate concern)

### Architecture Gap

The existing `Setting` class in `wxbtool/nn/setting.py` defines:
- ✅ `resolution = "5.625deg"`
- ✅ Temporal parameters (`input_span`, `pred_span`, etc.)
- ✅ Variable configurations
- ❌ **MISSING**: Spatial grid dimensions corresponding to the resolution

This gap forces developers to scatter magic numbers throughout the code, creating:
- **Maintenance burden**: Changes require updating multiple files
- **Error-prone development**: Easy to miss instances during modifications  
- **Limited extensibility**: Cannot easily support multiple resolutions
- **Code duplication**: Same dimensions repeated across files
- **Testing complexity**: Hard to create resolution-agnostic tests

## Proposed Solution

### Architecture Overview

Implement a **3-phase refactoring approach** to centralize spatial dimensions in the Settings system while maintaining backward compatibility and following the project's engineering protocols.

### Phase 1: Extend Settings Infrastructure

#### 1.1 Base Setting Class Enhancement
**File**: `wxbtool/nn/setting.py`

Add spatial dimension properties:
```python
class Setting:
    def __init__(self):
        # ... existing code ...
        
        # Spatial grid dimensions
        self.lat_size = 32      # Number of latitude grid points
        self.lon_size = 64      # Number of longitude grid points
        
        # Derived properties
        self.spatial_shape = (self.lat_size, self.lon_size)
        self.total_spatial_size = self.lat_size * self.lon_size
        
        # Grid boundaries (for 5.625deg resolution)
        self.lat_range = (87.1875, -87.1875)
        self.lon_range = (0, 354.375)
```

#### 1.2 Resolution Registry System
**New File**: `wxbtool/nn/resolution.py`

```python
class ResolutionConfig:
    """Configuration for different spatial resolutions"""
    
    RESOLUTIONS = {
        "5.625deg": {
            "lat_size": 32,
            "lon_size": 64, 
            "lat_range": (87.1875, -87.1875),
            "lon_range": (0, 354.375),
            "lat_step": 5.625,
            "lon_step": 5.625,
        },
        # Future resolutions can be added here
        "2.8125deg": {
            "lat_size": 64,
            "lon_size": 128,
            "lat_range": (87.1875, -87.1875),
            "lon_range": (0, 357.1875),
            "lat_step": 2.8125,
            "lon_step": 2.8125,
        },
    }
    
    @classmethod
    def get_config(cls, resolution: str) -> dict:
        if resolution not in cls.RESOLUTIONS:
            raise ValueError(f"Unsupported resolution: {resolution}")
        return cls.RESOLUTIONS[resolution]
```

#### 1.3 Setting Class Integration
Update existing setting classes to use the registry:

```python
from wxbtool.nn.resolution import ResolutionConfig

class Setting:
    def __init__(self):
        self.resolution = "5.625deg"
        # ... existing code ...
        
        # Load spatial configuration
        spatial_config = ResolutionConfig.get_config(self.resolution)
        self.lat_size = spatial_config["lat_size"]
        self.lon_size = spatial_config["lon_size"]
        self.lat_range = spatial_config["lat_range"]
        self.lon_range = spatial_config["lon_range"]
        self.spatial_shape = (self.lat_size, self.lon_size)
```

### Phase 2: Systematic Code Refactoring

#### 2.1 Model Specifications Refactoring
**Target Files**: `wxbtool/specs/res5_625/*.py`

**Before**:
```python
z500 = norm_z500(
    kwargs["geopotential"].view(
        -1, self.setting.input_span, self.setting.height, 32, 64
    )[:, :, self.setting.levels.index("500")]
)
```

**After**:
```python
z500 = norm_z500(
    kwargs["geopotential"].view(
        -1, self.setting.input_span, self.setting.height, 
        self.setting.lat_size, self.setting.lon_size
    )[:, :, self.setting.levels.index("500")]
)
```

#### 2.2 Neural Network Models Refactoring
**Target Files**: `wxbtool/zoo/res5_625/unet/*.py`

**Before**:
```python
spatial=(32, 64 + 2)
input = th.cat((input[:, :, :, 63:64], input, input[:, :, :, 0:1]), dim=3)
```

**After**:
```python
spatial=(self.setting.lat_size, self.setting.lon_size + 2)
lon_last_idx = self.setting.lon_size - 1
input = th.cat((input[:, :, :, lon_last_idx:], input, input[:, :, :, :1]), dim=3)
```

#### 2.3 Infrastructure Code Refactoring
**Target Files**: `wxbtool/nn/*.py`, `wxbtool/util/*.py`

**Before**:
```python
x, y = np.meshgrid(np.linspace(0, 1, num=32), np.linspace(0, 1, num=64))
"lat": np.linspace(87.1875, -87.1875, 32)
```

**After**:
```python
lat_size, lon_size = self.setting.spatial_shape
x, y = np.meshgrid(
    np.linspace(0, 1, num=lat_size), 
    np.linspace(0, 1, num=lon_size)
)
lat_start, lat_end = self.setting.lat_range
"lat": np.linspace(lat_start, lat_end, self.setting.lat_size)
```

### Phase 3: Testing and Validation

#### 3.1 Test Suite Updates
**Target Files**: `tests/*.py`

Create parameterized tests that work with different resolutions:
```python
@pytest.mark.parametrize("resolution", ["5.625deg", "2.8125deg"])
def test_model_with_resolution(resolution):
    setting = TestSetting()
    setting.resolution = resolution
    # Test will automatically use correct dimensions
```

#### 3.2 Backward Compatibility
- Maintain existing APIs during transition
- Add deprecation warnings for direct dimension access
- Provide migration guide for external users

#### 3.3 Validation Framework
```python
class ResolutionValidator:
    @staticmethod
    def validate_setting(setting):
        """Ensure spatial dimensions match resolution configuration"""
        expected = ResolutionConfig.get_config(setting.resolution)
        assert setting.lat_size == expected["lat_size"]
        assert setting.lon_size == expected["lon_size"]
```

## Implementation Plan

### Milestones

#### Milestone 1: Foundation (Week 1)
- [ ] Create `wxbtool/nn/resolution.py`
- [ ] Extend base `Setting` class with spatial properties
- [ ] Add validation framework
- [ ] Write comprehensive unit tests

#### Milestone 2: Core Refactoring (Week 2-3)
- [ ] Refactor model specifications (`wxbtool/specs/`)
- [ ] Update neural network models (`wxbtool/zoo/`)
- [ ] Refactor infrastructure code (`wxbtool/nn/`, `wxbtool/util/`)
- [ ] Update data generation code (`wxbtool/datagen/`)

#### Milestone 3: Testing and Polish (Week 4)
- [ ] Update all test files
- [ ] Add integration tests for multiple resolutions
- [ ] Create migration documentation
- [ ] Performance validation

#### Milestone 4: Documentation and Release (Week 5)
- [ ] Update user documentation
- [ ] Create developer migration guide
- [ ] Add examples for new resolution support
- [ ] Release with deprecation notices

### Risk Mitigation

1. **Regression Risk**: Comprehensive test suite ensuring no behavior changes
2. **Performance Risk**: Benchmark critical paths to ensure no performance degradation
3. **API Compatibility**: Gradual migration with deprecation warnings
4. **Team Adoption**: Clear documentation and examples

## Benefits

### Immediate Benefits
- **Eliminate Code Duplication**: Single source of truth for spatial dimensions
- **Reduce Error Risk**: No more manual synchronization of magic numbers
- **Improve Maintainability**: Changes require updates in one place only
- **Better Code Readability**: Self-documenting dimension usage

### Long-term Benefits
- **Multi-Resolution Support**: Easy to add support for different grid resolutions
- **Extensibility**: Framework for future spatial configuration needs
- **Testing**: Resolution-agnostic test suite
- **Performance**: Potential for resolution-specific optimizations

### Compliance with Engineering Protocols
This refactoring aligns with the project's core principles:
- ✅ **Clarity over Cleverness**: Explicit dimension configuration
- ✅ **Type Safety**: Proper type hints for all new code
- ✅ **Test Rigorously**: Comprehensive test coverage
- ✅ **Focused Changes**: Systematic, atomic refactoring approach

## Future Considerations

### Extension Points
1. **Dynamic Resolution**: Runtime resolution switching
2. **Custom Grids**: Support for irregular or custom spatial grids
3. **Resolution Interpolation**: Automatic data conversion between resolutions
4. **Multi-Scale Models**: Models that operate on multiple resolutions simultaneously

### Related Improvements
1. **Batch Size Configuration**: Similarly centralize batch size defaults
2. **Device Configuration**: Centralized device management
3. **Path Configuration**: Centralized data path management

## Conclusion

This refactoring addresses a fundamental architectural debt in wxbtool by eliminating hard-coded spatial dimensions and creating a flexible, maintainable system for resolution management. The proposed 3-phase approach ensures minimal disruption while providing significant long-term benefits for code quality and extensibility.

The implementation follows the project's engineering protocols and positions wxbtool for future enhancements including multi-resolution support and improved testing frameworks.

---

**Next Steps**: 
1. Review and approve this technical specification
2. Begin Phase 1 implementation with foundation components
3. Establish automated testing for all changes
4. Create detailed implementation tracking issues

**Dependencies**:
- No external dependencies
- Requires coordination with ongoing development to minimize conflicts
- Testing infrastructure should be established before major refactoring begins
