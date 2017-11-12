import numpy as np
import logging
import time
import sys

from shining_software.qc_opencl.raytracing_opencl_kernels.gpu_utils import get_opencl_kernel_path
from test_opencl._wrapper_test_opencl_module import TestOpenCLVoxelRaytraceWrapperGpu
from test_opencl._wrapper_test_opencl_module import has_gpu_support



def aligned_malloc(shape, dtype, fill_values=0):
    '''
    Make a numpy array ensuring that the data memory is
    aligned, for gpu use
    :param shape, dtype: for the numpy array
    :param fill_values: values will be initialized to this
    '''
    INTEL_GPU_MEM_ALIGNMENT = 4096
    INTEL_GPU_CACHE_ALIGNMENT = 64

    alignment = INTEL_GPU_MEM_ALIGNMENT  # starting pointer alignment
    size_requirement = INTEL_GPU_CACHE_ALIGNMENT * 8  # 64-byte cache aligned
    itemsize = np.dtype(dtype).itemsize
    assert alignment % itemsize == 0
    assert size_requirement % itemsize == 0
    nbytes = np.prod(shape) * itemsize
    nbytes_padded = nbytes + alignment + size_requirement
    buf = np.empty(nbytes_padded / itemsize, dtype=dtype)
    buf.fill(fill_values)
    start_index = -buf.ctypes.data % alignment
    start_index = start_index / itemsize
    data = buf[start_index:start_index + nbytes / itemsize]
    data = np.reshape(data, shape)
    assert data.ctypes.data % alignment == 0
    return data


def make_full_volume(shape, n_voxels, marking_threshold):
    bit_threshold = np.unpackbits(np.array([marking_threshold], dtype=np.uint8))[-marking_threshold.bit_length():]
    voxels_per_int32 = 32 / marking_threshold.bit_length()
    padding = 32 % marking_threshold.bit_length()
    bit_seq = np.hstack((np.zeros(padding, dtype=np.uint8), np.tile(bit_threshold, voxels_per_int32)))
    assert bit_seq.shape == (32, )
    byte_seq = np.packbits(bit_seq)  # this is a big-endian representation of the uint32
    if sys.byteorder == 'little':
        byte_seq = byte_seq[::-1]
    fill_value = np.ascontiguousarray(byte_seq).view(np.uint32)[0]
    v_shape = tuple(shape) + (1 + (n_voxels - 1) / (32 / marking_threshold.bit_length()), )
    v = aligned_malloc(v_shape, np.uint32, fill_values=fill_value)
    if n_voxels % voxels_per_int32 != 0:
        remaining_voxels = n_voxels % voxels_per_int32
        padding = 32 - remaining_voxels * marking_threshold.bit_length()
        bit_seq = np.hstack((np.zeros(padding, dtype=np.uint8), np.tile(bit_threshold, remaining_voxels)))
        assert bit_seq.shape == (32, )
        byte_seq = np.packbits(bit_seq)  # this is a big-endian representation of the uint32
        if sys.byteorder == 'little':
            byte_seq = byte_seq[::-1]
        fill_value = np.ascontiguousarray(byte_seq).view(np.uint32)[0]
        v[..., -1] = fill_value
    counts = aligned_malloc(shape, np.uint8, n_voxels)
    known = aligned_malloc(shape, np.uint8)
    return v, counts, known



def demo_raytracing_gpu():
    print "Checking for GPU Support"
    if has_gpu_support(False)==0:
        print "No opencl support, Segmentation Fault (Core dump) is expected"

    marking_threshold = 1
    voxel_gpu_tracer = TestOpenCLVoxelRaytraceWrapperGpu(0.03, 0.03, marking_threshold, get_opencl_kernel_path())

    print "Loaded gpu wrapper"
    n_voxels = 10
    v, counts, known = make_full_volume((10, 10), n_voxels, marking_threshold)
    points = aligned_malloc((2, 3), dtype=np.int16)
    points[0, 0] = 9
    points[0, 1] = 9
    points[0, 2] = 4
    points[1, 0] = 9
    points[1, 1] = 9
    points[1, 2] = 4
    voxel_origin = np.array([4, 4, 4], dtype=np.int16)
    t = time.time()
    valid = voxel_gpu_tracer.raytrace_pointcloud(v, n_voxels, counts, known, points, voxel_origin, 200., 200., True, True)
    print "known:\n", known
    print "counts:\n", counts
    print "valids:\n", valid
    print "volume:\n", v

if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    demo_raytracing_gpu()
