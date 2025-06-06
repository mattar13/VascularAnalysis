#!/usr/bin/env python3
"""
Comprehensive Vessel Tracing Benchmark

This script benchmarks all operations in the vessel tracing pipeline
to identify performance bottlenecks and measure processing times.

Simply run: python test/vessel_tracing_benchmark.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import os

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from VesselTracer import VesselTracer
    from VesselTracer.tracer import GPU_AVAILABLE
except ImportError as e:
    print(f"Error importing VesselTracer: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class VesselTracingBenchmark:
    """Comprehensive benchmarking class for vessel tracing operations."""
    
    def __init__(self):
        self.results = {}
        self.test_volume = None
        self.tracer = None
        
    def create_synthetic_czi(self, size: Tuple[int, int, int], filename: str) -> str:
        """Create a synthetic CZI-like volume for testing."""
        import tifffile
        
        print(f"Creating synthetic volume of size {size}...")
        z, y, x = size
        
        # Create base volume with noise
        volume = np.random.normal(0.1, 0.05, size).astype(np.float32)
        volume = np.clip(volume, 0, 1)
        
        # Add vessel-like structures
        num_vessels = 8
        for i in range(num_vessels):
            # Random vessel parameters
            start_z = np.random.randint(0, z//4)
            end_z = np.random.randint(3*z//4, z)
            center_y = np.random.randint(y//4, 3*y//4)
            center_x = np.random.randint(x//4, 3*x//4)
            
            radius = np.random.uniform(3, 8)
            intensity = np.random.uniform(0.6, 0.9)
            
            # Create vessel path with some curvature
            zz = np.linspace(start_z, end_z, end_z - start_z + 1)
            yy = center_y + 20 * np.sin(np.linspace(0, 2*np.pi, len(zz)))
            xx = center_x + 15 * np.cos(np.linspace(0, 3*np.pi, len(zz)))
            
            # Add vessel to volume
            for z_idx, y_idx, x_idx in zip(zz.astype(int), yy.astype(int), xx.astype(int)):
                if 0 <= z_idx < z and 0 <= y_idx < y and 0 <= x_idx < x:
                    # Create circular cross-section
                    for dy in range(-int(radius*2), int(radius*2)+1):
                        for dx in range(-int(radius*2), int(radius*2)+1):
                            nz, ny, nx = z_idx, y_idx + dy, x_idx + dx
                            if 0 <= ny < y and 0 <= nx < x:
                                dist = np.sqrt(dy**2 + dx**2)
                                if dist <= radius:
                                    vessel_intensity = intensity * np.exp(-dist**2 / (2*radius**2))
                                    volume[nz, ny, nx] = max(volume[nz, ny, nx], vessel_intensity)
        
        # Add some larger background structures
        for i in range(3):
            center_z = np.random.randint(z//4, 3*z//4)
            center_y = np.random.randint(y//4, 3*y//4)
            center_x = np.random.randint(x//4, 3*x//4)
            
            bg_radius = np.random.uniform(30, 60)
            bg_intensity = np.random.uniform(0.3, 0.5)
            
            for dz in range(-int(bg_radius), int(bg_radius)+1):
                for dy in range(-int(bg_radius), int(bg_radius)+1):
                    for dx in range(-int(bg_radius), int(bg_radius)+1):
                        nz, ny, nx = center_z + dz, center_y + dy, center_x + dx
                        if 0 <= nz < z and 0 <= ny < y and 0 <= nx < x:
                            dist = np.sqrt(dz**2 + dy**2 + dx**2)
                            if dist <= bg_radius:
                                bg_val = bg_intensity * np.exp(-dist**2 / (2*bg_radius**2))
                                volume[nz, ny, nx] += bg_val
        
        # Normalize and add final noise
        volume = np.clip(volume, 0, 1)
        volume += np.random.normal(0, 0.02, size).astype(np.float32)
        volume = np.clip(volume, 0, 1)
        
        # Convert to uint16 (typical for microscopy)
        volume_uint16 = (volume * 65535).astype(np.uint16)
        
        # Save as TIFF (simpler than CZI for testing)
        tifffile.imwrite(filename, volume_uint16)
        print(f"Synthetic volume saved to {filename}")
        
        return filename
    
    def benchmark_operation(self, operation_name: str, operation_func, *args, **kwargs) -> Dict:
        """Benchmark a single operation."""
        print(f"  Benchmarking {operation_name}...")
        
        # Warm-up run
        try:
            operation_func(*args, **kwargs)
        except Exception as e:
            print(f"    Warm-up failed: {e}")
        
        # Timed run
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        benchmark_result = {
            'operation': operation_name,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'success': success,
            'error': error_msg,
            'result_shape': getattr(result, 'shape', None) if hasattr(result, 'shape') else None,
            'result_type': type(result).__name__
        }
        
        if success:
            print(f"    ✓ {operation_name}: {benchmark_result['execution_time']:.3f}s")
        else:
            print(f"    ✗ {operation_name}: FAILED - {error_msg}")
        
        return benchmark_result
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available
    
    def benchmark_full_pipeline(self, volume_size: Tuple[int, int, int]) -> Dict:
        """Benchmark the complete vessel tracing pipeline."""
        print(f"\n{'='*60}")
        print(f"BENCHMARKING FULL PIPELINE - Volume Size: {volume_size}")
        print(f"{'='*60}")
        
        pipeline_results = {}
        
        # Create synthetic data
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            # Step 1: Create synthetic volume
            pipeline_results['data_creation'] = self.benchmark_operation(
                "Data Creation",
                self.create_synthetic_czi,
                volume_size,
                tmp_filename
            )
            
            # Step 2: Initialize VesselTracer
            pipeline_results['initialization'] = self.benchmark_operation(
                "VesselTracer Initialization",
                self._init_tracer,
                tmp_filename
            )
            
            if not pipeline_results['initialization']['success']:
                return pipeline_results
            
            # Step 3: Segment ROI
            pipeline_results['segment_roi'] = self.benchmark_operation(
                "ROI Segmentation",
                self.tracer.segment_roi,
                remove_dead_frames=True,
                dead_frame_threshold=1.5
            )
            
            # Step 4: Median Filter
            pipeline_results['median_filter'] = self.benchmark_operation(
                "Median Filtering",
                self.tracer.median_filter,
                median_filter_size=3
            )
            
            # Step 5: Detrend
            pipeline_results['detrend'] = self.benchmark_operation(
                "Detrending",
                self.tracer.detrend
            )
            
            # Step 6: Background Smoothing
            pipeline_results['background_smoothing'] = self.benchmark_operation(
                "Background Smoothing",
                self.tracer.background_smoothing,
                epsilon=1e-6,
                mode='gaussian'
            )
            
            # Step 7: Regular Smoothing
            pipeline_results['smoothing'] = self.benchmark_operation(
                "Gaussian Smoothing",
                self.tracer.smooth
            )
            
            # Step 8: Binarization
            pipeline_results['binarization'] = self.benchmark_operation(
                "Binarization",
                self.tracer.binarize
            )
            
            # Step 9: Skeletonization
            pipeline_results['skeletonization'] = self.benchmark_operation(
                "Skeletonization",
                self.tracer.skeletonize
            )
            
            # Step 10: Region Analysis
            pipeline_results['region_analysis'] = self.benchmark_operation(
                "Region Analysis",
                self.tracer.determine_regions
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
        
        return pipeline_results
    
    def _init_tracer(self, filename: str) -> VesselTracer:
        """Initialize VesselTracer with test file."""
        # Create a simple config for testing
        config_path = Path(__file__).parent.parent / 'config' / 'default_vessel_config.yaml'
        self.tracer = VesselTracer(filename, str(config_path))
        return self.tracer
    
    def benchmark_individual_operations(self, volume_size: Tuple[int, int, int]) -> Dict:
        """Benchmark individual operations in isolation."""
        print(f"\n{'='*60}")
        print(f"BENCHMARKING INDIVIDUAL OPERATIONS - Volume Size: {volume_size}")
        print(f"{'='*60}")
        
        # Create test data
        test_volume = np.random.random(volume_size).astype(np.float32)
        
        individual_results = {}
        
        # Import required functions
        from scipy.ndimage import gaussian_filter, median_filter
        from skimage.filters import threshold_triangle, threshold_otsu
        from skimage.morphology import remove_small_objects, binary_closing, binary_opening, ball
        
        # Test Gaussian filtering
        individual_results['gaussian_small'] = self.benchmark_operation(
            "Gaussian Filter (σ=1.0)",
            gaussian_filter,
            test_volume,
            sigma=(1.0, 1.0, 1.0)
        )
        
        individual_results['gaussian_large'] = self.benchmark_operation(
            "Gaussian Filter (σ=5.0)",
            gaussian_filter,
            test_volume,
            sigma=(5.0, 5.0, 5.0)
        )
        
        # Test median filtering
        individual_results['median_small'] = self.benchmark_operation(
            "Median Filter (size=3)",
            median_filter,
            test_volume,
            size=3
        )
        
        individual_results['median_large'] = self.benchmark_operation(
            "Median Filter (size=5)",
            median_filter,
            test_volume,
            size=5
        )
        
        # Test thresholding
        individual_results['threshold_triangle'] = self.benchmark_operation(
            "Triangle Threshold",
            threshold_triangle,
            test_volume.ravel()
        )
        
        individual_results['threshold_otsu'] = self.benchmark_operation(
            "Otsu Threshold",
            threshold_otsu,
            test_volume.ravel()
        )
        
        # Create binary volume for morphological operations
        binary_volume = test_volume > 0.5
        
        # Test morphological operations
        individual_results['remove_small_objects'] = self.benchmark_operation(
            "Remove Small Objects",
            remove_small_objects,
            binary_volume,
            min_size=64
        )
        
        individual_results['binary_opening'] = self.benchmark_operation(
            "Binary Opening",
            binary_opening,
            binary_volume,
            ball(2)
        )
        
        individual_results['binary_closing'] = self.benchmark_operation(
            "Binary Closing",
            binary_closing,
            binary_volume,
            ball(2)
        )
        
        return individual_results
    
    def benchmark_memory_usage(self, volume_sizes: List[Tuple[int, int, int]]) -> Dict:
        """Benchmark memory usage scaling with volume size."""
        print(f"\n{'='*60}")
        print("MEMORY USAGE SCALING BENCHMARK")
        print(f"{'='*60}")
        
        memory_results = {}
        
        for size in volume_sizes:
            print(f"\nTesting volume size: {size}")
            voxel_count = np.prod(size)
            
            # Test basic array operations
            start_memory = self.get_memory_usage()
            
            # Create volume
            volume = np.random.random(size).astype(np.float32)
            after_creation = self.get_memory_usage()
            
            # Apply some operations
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(volume, sigma=(1.0, 1.0, 1.0))
            after_smooth = self.get_memory_usage()
            
            # Binary operations
            binary = volume > 0.5
            after_binary = self.get_memory_usage()
            
            # Clean up
            del volume, smoothed, binary
            final_memory = self.get_memory_usage()
            
            memory_results[size] = {
                'voxel_count': voxel_count,
                'start_memory': start_memory,
                'after_creation': after_creation,
                'after_smooth': after_smooth,
                'after_binary': after_binary,
                'final_memory': final_memory,
                'peak_usage': max(after_creation, after_smooth, after_binary) - start_memory,
                'memory_per_voxel': (max(after_creation, after_smooth, after_binary) - start_memory) / voxel_count * 1024 * 1024  # bytes per voxel
            }
            
            print(f"  Peak memory usage: {memory_results[size]['peak_usage']:.1f} MB")
            print(f"  Memory per voxel: {memory_results[size]['memory_per_voxel']:.2f} bytes")
        
        return memory_results
    
    def generate_comprehensive_report(self, all_results: Dict):
        """Generate a comprehensive benchmark report."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE VESSEL TRACING BENCHMARK REPORT")
        print(f"{'='*80}")
        
        # System information
        print("\nSYSTEM INFORMATION:")
        print(f"  GPU Available: {GPU_AVAILABLE}")
        if GPU_AVAILABLE:
            try:
                import cupy as cp
                device = cp.cuda.Device(0)
                total_mem = device.mem_info[1] / 1024**3
                print(f"  GPU Memory: {total_mem:.1f} GB")
            except:
                print("  GPU Memory: Unable to query")
        
        try:
            import psutil
            print(f"  CPU Count: {psutil.cpu_count()}")
            print(f"  Total RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        except ImportError:
            print("  System info: psutil not available")
        
        # Pipeline performance summary
        if 'pipeline' in all_results:
            print(f"\n{'='*50}")
            print("PIPELINE PERFORMANCE SUMMARY")
            print(f"{'='*50}")
            
            for size, results in all_results['pipeline'].items():
                print(f"\nVolume Size: {size}")
                print("-" * 30)
                
                total_time = 0
                successful_ops = 0
                
                for op_name, op_result in results.items():
                    if op_result['success']:
                        print(f"  {op_name:20s}: {op_result['execution_time']:6.3f}s")
                        total_time += op_result['execution_time']
                        successful_ops += 1
                    else:
                        print(f"  {op_name:20s}: FAILED")
                
                print(f"  {'Total Time':20s}: {total_time:6.3f}s")
                print(f"  {'Success Rate':20s}: {successful_ops}/{len(results)} operations")
                
                # Calculate throughput
                voxel_count = np.prod(size)
                if total_time > 0:
                    throughput = voxel_count / total_time / 1e6  # MVoxels/sec
                    print(f"  {'Throughput':20s}: {throughput:6.2f} MVoxels/sec")
        
        # Individual operations summary
        if 'individual' in all_results:
            print(f"\n{'='*50}")
            print("INDIVIDUAL OPERATIONS PERFORMANCE")
            print(f"{'='*50}")
            
            for size, results in all_results['individual'].items():
                print(f"\nVolume Size: {size}")
                print("-" * 30)
                
                for op_name, op_result in results.items():
                    if op_result['success']:
                        voxel_count = np.prod(size)
                        throughput = voxel_count / op_result['execution_time'] / 1e6
                        print(f"  {op_name:25s}: {op_result['execution_time']:6.3f}s ({throughput:5.1f} MVox/s)")
        
        # Memory usage summary
        if 'memory' in all_results:
            print(f"\n{'='*50}")
            print("MEMORY USAGE ANALYSIS")
            print(f"{'='*50}")
            
            print(f"{'Volume Size':15s} {'Voxels':10s} {'Peak MB':10s} {'Bytes/Voxel':12s}")
            print("-" * 50)
            
            for size, mem_info in all_results['memory'].items():
                print(f"{str(size):15s} {mem_info['voxel_count']:10d} "
                      f"{mem_info['peak_usage']:10.1f} {mem_info['memory_per_voxel']:12.2f}")
        
        # Performance recommendations
        print(f"\n{'='*50}")
        print("PERFORMANCE RECOMMENDATIONS")
        print(f"{'='*50}")
        
        recommendations = []
        
        if GPU_AVAILABLE:
            recommendations.append("✓ GPU acceleration is available for filtering operations")
        else:
            recommendations.append("• Consider installing CuPy for GPU acceleration")
        
        # Analyze pipeline timing
        if 'pipeline' in all_results:
            for size, results in all_results['pipeline'].items():
                slow_ops = [(name, res['execution_time']) for name, res in results.items() 
                           if res['success'] and res['execution_time'] > 1.0]
                
                if slow_ops:
                    slow_ops.sort(key=lambda x: x[1], reverse=True)
                    recommendations.append(f"• For {size} volumes, consider optimizing: {slow_ops[0][0]} ({slow_ops[0][1]:.1f}s)")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        print(f"\n{'='*80}")

def main():
    """Main benchmarking function."""
    print("VesselTracer Comprehensive Benchmark")
    print("====================================")
    print("Testing all vessel tracing operations")
    print()
    
    benchmark = VesselTracingBenchmark()
    
    # Define test volumes (start small to avoid memory issues)
    test_volumes = [
        (50, 256, 256),    # Small
        (100, 512, 512),   # Medium
        (150, 512, 512),   # Large
    ]
    
    all_results = {}
    
    # Benchmark full pipeline
    print("Phase 1: Full Pipeline Benchmarks")
    all_results['pipeline'] = {}
    for volume_size in test_volumes:
        try:
            all_results['pipeline'][volume_size] = benchmark.benchmark_full_pipeline(volume_size)
        except Exception as e:
            print(f"Pipeline benchmark failed for {volume_size}: {e}")
            all_results['pipeline'][volume_size] = {'error': str(e)}
    
    # Benchmark individual operations
    print("\nPhase 2: Individual Operations Benchmarks")
    all_results['individual'] = {}
    for volume_size in test_volumes:
        try:
            all_results['individual'][volume_size] = benchmark.benchmark_individual_operations(volume_size)
        except Exception as e:
            print(f"Individual operations benchmark failed for {volume_size}: {e}")
            all_results['individual'][volume_size] = {'error': str(e)}
    
    # Benchmark memory usage
    print("\nPhase 3: Memory Usage Analysis")
    try:
        all_results['memory'] = benchmark.benchmark_memory_usage(test_volumes)
    except Exception as e:
        print(f"Memory benchmark failed: {e}")
        all_results['memory'] = {'error': str(e)}
    
    # Generate comprehensive report
    benchmark.generate_comprehensive_report(all_results)
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main() 