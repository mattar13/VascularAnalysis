#!/usr/bin/env python3
"""
GPU Benchmark and Test Script for VesselTracer

Standalone test script that benchmarks GPU vs CPU performance
for VesselTracer operations.

Simply run: python test/gpu_benchmark.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from VesselTracer import VesselTracer
    from VesselTracer.tracer import GPU_AVAILABLE
except ImportError as e:
    print(f"Error importing VesselTracer: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_gpu_availability():
    """Test if GPU acceleration is available."""
    print("="*60)
    print("GPU AVAILABILITY TEST")
    print("="*60)
    
    print(f"GPU Available: {GPU_AVAILABLE}")
    
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            device = cp.cuda.Device(0)
            print(f"GPU Device: {device}")
            
            total_mem = device.mem_info[1] / 1024**3  # Convert to GB
            free_mem = device.mem_info[0] / 1024**3
            used_mem = total_mem - free_mem
            
            print(f"GPU Memory: {used_mem:.1f} / {total_mem:.1f} GB used")
            print(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
            print(f"CuPy Version: {cp.__version__}")
            
            return True
        except Exception as e:
            print(f"GPU test failed: {e}")
            return False
    else:
        print("CuPy not installed or GPU not available")
        return False

def create_test_volume(size: Tuple[int, int, int]) -> np.ndarray:
    """Create a synthetic test volume with vessel-like structures."""
    print(f"Creating test volume of size {size}...")
    volume = np.random.random(size).astype(np.float32)
    
    # Add some structure to make it more realistic
    z, y, x = size
    center_z, center_y, center_x = z//2, y//2, x//2
    
    # Create some vessel-like structures
    for i in range(3):
        start_y = np.random.randint(y//4, 3*y//4)
        start_x = np.random.randint(x//4, 3*x//4)
        
        for z_idx in range(z):
            y_offset = int(10 * np.sin(z_idx * 0.1))
            x_offset = int(8 * np.cos(z_idx * 0.08))
            
            cy, cx = start_y + y_offset, start_x + x_offset
            
            # Add circular structure
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < y and 0 <= nx < x:
                        dist = np.sqrt(dy**2 + dx**2)
                        if dist <= 5:
                            volume[z_idx, ny, nx] = max(volume[z_idx, ny, nx], 
                                                      0.8 * np.exp(-dist**2 / 10))
    
    return volume

def benchmark_gaussian_filter(test_volume: np.ndarray, sigma: tuple, use_gpu: bool) -> Dict:
    """Benchmark Gaussian filtering."""
    from scipy.ndimage import gaussian_filter
    
    print(f"\nBenchmarking Gaussian Filter (sigma={sigma})...")
    
    # Create tracer instance for GPU methods
    tracer = VesselTracer.__new__(VesselTracer)
    tracer.use_gpu = use_gpu
    tracer.gpu_enabled = use_gpu and GPU_AVAILABLE
    tracer.verbose = 1
    
    results = {}
    
    # CPU test
    print("  Running CPU Gaussian filter...")
    start_time = time.time()
    cpu_result = gaussian_filter(test_volume, sigma=sigma)
    cpu_time = time.time() - start_time
    results['cpu_time'] = cpu_time
    print(f"  CPU time: {cpu_time:.3f}s")
    
    # GPU test
    if tracer.gpu_enabled:
        print("  Running GPU Gaussian filter...")
        start_time = time.time()
        gpu_result = tracer._gpu_gaussian_filter(test_volume, sigma)
        gpu_time = time.time() - start_time
        results['gpu_time'] = gpu_time
        results['speedup'] = cpu_time / gpu_time if gpu_time > 0 else 0
        
        # Check accuracy
        diff = np.mean(np.abs(cpu_result - gpu_result))
        results['accuracy_diff'] = diff
        
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Speedup: {results['speedup']:.1f}x")
        print(f"  Accuracy diff: {diff:.2e}")
    else:
        print("  GPU not available")
        results['gpu_time'] = None
        results['speedup'] = None
        results['accuracy_diff'] = None
    
    return results

def benchmark_median_filter(test_volume: np.ndarray, size: int, use_gpu: bool) -> Dict:
    """Benchmark median filtering."""
    from scipy.ndimage import median_filter
    
    print(f"\nBenchmarking Median Filter (size={size})...")
    
    # Create tracer instance for GPU methods
    tracer = VesselTracer.__new__(VesselTracer)
    tracer.use_gpu = use_gpu
    tracer.gpu_enabled = use_gpu and GPU_AVAILABLE
    tracer.verbose = 1
    
    results = {}
    
    # CPU test
    print("  Running CPU median filter...")
    start_time = time.time()
    cpu_result = median_filter(test_volume, size=size)
    cpu_time = time.time() - start_time
    results['cpu_time'] = cpu_time
    print(f"  CPU time: {cpu_time:.3f}s")
    
    # GPU test
    if tracer.gpu_enabled:
        print("  Running GPU median filter...")
        start_time = time.time()
        gpu_result = tracer._gpu_median_filter(test_volume, size)
        gpu_time = time.time() - start_time
        results['gpu_time'] = gpu_time
        results['speedup'] = cpu_time / gpu_time if gpu_time > 0 else 0
        
        # Check accuracy
        diff = np.mean(np.abs(cpu_result - gpu_result))
        results['accuracy_diff'] = diff
        
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Speedup: {results['speedup']:.1f}x")
        print(f"  Accuracy diff: {diff:.2e}")
    else:
        print("  GPU not available")
        results['gpu_time'] = None
        results['speedup'] = None
        results['accuracy_diff'] = None
    
    return results

def run_benchmark_tests():
    """Run comprehensive GPU benchmarks."""
    print("\n" + "="*60)
    print("COMPREHENSIVE GPU BENCHMARK")
    print("="*60)
    
    # Test different volume sizes
    test_sizes = [
        (50, 256, 256),    # Small
        (100, 512, 512),   # Medium  
        (200, 512, 512),   # Large
    ]
    
    all_results = []
    
    for size in test_sizes:
        print(f"\n{'='*40}")
        print(f"Testing volume size: {size}")
        print(f"{'='*40}")
        
        # Create test volume
        test_volume = create_test_volume(size)
        
        # Test operations
        operations = [
            ('Gaussian σ=2.0', lambda: benchmark_gaussian_filter(test_volume, (2.0, 2.0, 2.0), True)),
            ('Gaussian σ=5.0', lambda: benchmark_gaussian_filter(test_volume, (5.0, 5.0, 5.0), True)),
            ('Median size=3', lambda: benchmark_median_filter(test_volume, 3, True)),
            ('Median size=5', lambda: benchmark_median_filter(test_volume, 5, True)),
        ]
        
        for op_name, op_func in operations:
            try:
                result = op_func()
                result['operation'] = op_name
                result['volume_size'] = size
                all_results.append(result)
            except Exception as e:
                print(f"Error in {op_name}: {e}")
    
    return all_results

def generate_summary_report(results: List[Dict]):
    """Generate a summary report of benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY REPORT")
    print("="*60)
    
    gpu_results = [r for r in results if r.get('gpu_time') is not None]
    
    if not gpu_results:
        print("No GPU results available")
        return
    
    print(f"Total operations tested: {len(results)}")
    print(f"GPU-accelerated operations: {len(gpu_results)}")
    
    # Calculate statistics
    speedups = [r['speedup'] for r in gpu_results if r['speedup'] is not None]
    if speedups:
        avg_speedup = np.mean(speedups)
        max_speedup = max(speedups)
        min_speedup = min(speedups)
        
        print(f"\nSpeedup Statistics:")
        print(f"  Average speedup: {avg_speedup:.1f}x")
        print(f"  Maximum speedup: {max_speedup:.1f}x")
        print(f"  Minimum speedup: {min_speedup:.1f}x")
    
    # Accuracy statistics
    accuracies = [r['accuracy_diff'] for r in gpu_results if r['accuracy_diff'] is not None]
    if accuracies:
        avg_accuracy = np.mean(accuracies)
        max_accuracy = max(accuracies)
        
        print(f"\nAccuracy Statistics:")
        print(f"  Average difference: {avg_accuracy:.2e}")
        print(f"  Maximum difference: {max_accuracy:.2e}")
    
    print(f"\nDetailed Results:")
    print("-" * 50)
    for result in results:
        print(f"{result['operation']} (size {result['volume_size']}):")
        print(f"  CPU: {result['cpu_time']:.3f}s")
        if result.get('gpu_time'):
            print(f"  GPU: {result['gpu_time']:.3f}s")
            print(f"  Speedup: {result['speedup']:.1f}x")
            print(f"  Accuracy: {result['accuracy_diff']:.2e}")
        else:
            print("  GPU: Not available")
        print()

def test_vesseltracer_integration():
    """Test GPU acceleration with actual VesselTracer workflow."""
    print("\n" + "="*60)
    print("VESSELTRACER INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test the built-in benchmark method
        tracer = VesselTracer.__new__(VesselTracer)
        tracer.use_gpu = True
        tracer.gpu_enabled = GPU_AVAILABLE
        tracer.verbose = 2
        tracer._log = lambda msg, level=1, timing=None: print(f"  {msg}")
        
        if hasattr(tracer, 'benchmark_gpu'):
            print("Running built-in GPU benchmark...")
            benchmark_results = tracer.benchmark_gpu((50, 256, 256))
            
            print("Built-in benchmark results:")
            for key, value in benchmark_results.items():
                if value is not None:
                    if 'time' in key:
                        print(f"  {key}: {value:.3f}s")
                    elif 'speedup' in key:
                        print(f"  {key}: {value:.1f}x")
                    elif 'diff' in key:
                        print(f"  {key}: {value:.2e}")
                    else:
                        print(f"  {key}: {value}")
        else:
            print("Built-in benchmark method not available")
            
    except Exception as e:
        print(f"Integration test failed: {e}")

def test_cpu_only_mode():
    """Test CPU-only mode for comparison."""
    print("\n" + "="*60)
    print("CPU-ONLY MODE TEST")
    print("="*60)
    
    # Create small test volume for quick comparison
    test_volume = create_test_volume((50, 256, 256))
    
    print("Testing with GPU disabled...")
    
    # Test Gaussian filter
    result = benchmark_gaussian_filter(test_volume, (2.0, 2.0, 2.0), use_gpu=False)
    print("CPU-only mode working correctly")

def main():
    """Main test function."""
    print("VesselTracer GPU Benchmark Test")
    print("===============================")
    print("Standalone test - no arguments needed")
    print()
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    
    # Always test CPU-only mode
    test_cpu_only_mode()
    
    if gpu_available:
        print(f"\nRunning benchmarks with GPU enabled...")
        
        # Run comprehensive benchmarks
        results = run_benchmark_tests()
        
        # Generate summary
        generate_summary_report(results)
        
        # Test integration
        test_vesseltracer_integration()
    else:
        print("\nGPU not available - skipping GPU benchmarks")
        print("Install CuPy and ensure CUDA is available for GPU acceleration")
    
    print(f"\nBenchmark complete!")
    print("="*60)

if __name__ == "__main__":
    main() 