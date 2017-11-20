#include <boost/python.hpp>
#include "numpy_boost_python.hpp"
#include <boost/format.hpp>
#include <raytrace_utils.hpp>
#include <iostream>
#include <cmath>
#include <map>
#include <limits>
#include <chrono>
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include "gpu_utils.h"

#define PRINT_DEBUG false
#define VOXEL_DEBUG false
#define USE_LOCAL_MEM false
#define RAWSIZE(m) (m.num_elements() * sizeof(*(m.data())))
#define ZEROCOPYSIZE(m) (zeroCopySizeAlignment(RAWSIZE(m)))


namespace py = boost::python;

// https://wiki.python.org/moin/boost.python/HowTo#Multithreading_Support_for_my_function
class ScopedGILRelease
{
public:
    ScopedGILRelease() {m_thread_state = PyEval_SaveThread();}
    ~ScopedGILRelease() {
        PyEval_RestoreThread(m_thread_state);
        m_thread_state = NULL;
    }
private:
    PyThreadState * m_thread_state;
};



void translate_exception(std::runtime_error const& e)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}


int has_gpu_support(bool debug) {
    int err;
    cl_uint num_of_platforms;
    clGetPlatformIDs(0, NULL, &num_of_platforms);
    if (debug) {
        cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_of_platforms);
        clGetPlatformIDs(num_of_platforms, platforms, NULL);
        std::cout << "Platform count: " << num_of_platforms << std::endl;
        if (num_of_platforms > 0) std::cout << "Platform names:\n";
        for(cl_uint i = 0; i < num_of_platforms; ++i) {
            size_t platform_name_length = 0;
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, 0, &platform_name_length);
            char* platform_name = new char[200];
            err = clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, platform_name_length, platform_name, 0);
            std::cout << "    [" << i << "] " << platform_name << std::endl;
        }
        free(platforms);
        if (num_of_platforms == 0) std::cout << "No GPU support found" << std::endl;
        if (num_of_platforms > 1) std::cout << "GPU module will not work because more than 1 GPU was found" << std::endl;
    }
    return (num_of_platforms == 1);
}


// Helper structure to initialize and hold basic OpenCL objects.
// Contains platform, device, context and queue.
struct OpenCLBasic
{
    cl_platform_id* platforms;
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program pointcloud_program;      // compute program
    cl_kernel pointcloud_kernel;        // compute kernel
    cl_uint max_work_item_dims;         // work group dimensions
    size_t max_work_group_size;         // max number of work items per work group
    size_t max_work_item_sizes[255];// max number of work items per work group
    size_t local_mem_size;              // size of device local memory
    bool using_nvidia_gpu;

    OpenCLBasic (string const & kernel_full_path) {
        char* value;
        size_t valueSize;
        cl_uint num_of_platforms;
        int err;
        int gpu = 1;
        using_nvidia_gpu = false;

        // get all platforms
        clGetPlatformIDs(0, NULL, &num_of_platforms);
        platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_of_platforms);
        if (VOXEL_DEBUG) printf("Platform count: %d\n", num_of_platforms);
        clGetPlatformIDs(num_of_platforms, platforms, NULL);

        // support either beignet (on intel gpu) or nvidia opencl (on test servers)
        if (VOXEL_DEBUG and num_of_platforms > 1) std::cout << "GPU module will not work because more than 1 GPU was found" << std::endl;
        assert(num_of_platforms == 1);

        // Get the name itself for the i-th platform
        if (VOXEL_DEBUG && num_of_platforms) std::cout << "Platform names:" << std::endl;
        for(cl_uint i = 0; i < num_of_platforms; ++i) {
            size_t platform_name_length = 0;
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, 0, &platform_name_length);
            char* platform_name = new char[200];
            err = clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, platform_name_length, platform_name, 0);
            if (VOXEL_DEBUG) std::cout << "    [" << i << "] " << platform_name << std::endl;
            if (strstr(platform_name, "NVIDIA") != NULL) {
                using_nvidia_gpu = true;
                if (VOXEL_DEBUG) std::cout << "Found Nvidia platform. Using memory copy to gpu-memory" << std::endl;
            }
        }

        err = clGetDeviceIDs(platforms[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
        if (err != CL_SUCCESS) {
            err = clGetDeviceIDs(0, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
            if (err != CL_SUCCESS)
                throw std::runtime_error("Error: Failed to create a device group: " + string(opencl_error_to_str(err)));
        }

        // Create a compute context
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        if (!context) {
            throw std::runtime_error("Error: Failed to create a compute context!: " + string(opencl_error_to_str(err)));
        }

        // Create a command queue
        commands = clCreateCommandQueue(context, device_id, 0, &err);
        if (!commands) {
            throw std::runtime_error("Error: Failed to create a command queue!: " + string(opencl_error_to_str(err)));
        }

        load_build(kernel_full_path, pointcloud_program, "kernel_pointcloud.cl");

        pointcloud_kernel = clCreateKernel(pointcloud_program, "raytrace_pointcloud", &err);
        if (!pointcloud_kernel || err != CL_SUCCESS) {
            throw std::runtime_error("Error: Failed to create raytrace_pointcloud kernel! " + string(opencl_error_to_str(err)));
        }

        // get the max number of work items per work group
        err = clGetDeviceInfo(
            device_id,
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(max_work_group_size),
            &max_work_group_size,
            0
            );
        SAMPLE_CHECK_ERRORS(err);
        
        err = clGetDeviceInfo(
            device_id,
            CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
            sizeof(max_work_item_dims),
            &max_work_item_dims,
            0
            );
        SAMPLE_CHECK_ERRORS(err);

        // get the max number of work items per work group again
        err = clGetDeviceInfo(
            device_id,
            CL_DEVICE_MAX_WORK_ITEM_SIZES,
            max_work_item_dims * sizeof(max_work_item_sizes[0]),
            &max_work_item_sizes,
            0
            );
        SAMPLE_CHECK_ERRORS(err);

        // get the local memory in the device
        err = clGetDeviceInfo(
            device_id,
            CL_DEVICE_LOCAL_MEM_SIZE,
            sizeof(local_mem_size),
            &local_mem_size,
            0
            );
        SAMPLE_CHECK_ERRORS(err);
     }

     void load_build(string const& kernel_full_path, cl_program& pg, string const& kernel_name)
     {
        int err;

        // Create the compute program from the source buffer
        if (VOXEL_DEBUG) printf("Full kernel path is: %s\n", (kernel_full_path + "/" + kernel_name).c_str());
        char* kernel_source = ReadSources((kernel_full_path + "/" + kernel_name).c_str());
        pg = clCreateProgramWithSource(context, 1, (const char **) & kernel_source, NULL, &err);
        if (!pg) {
            throw std::runtime_error("Error: Failed to create compute program!\n" + string(opencl_error_to_str(err)));
        }
        free(kernel_source);

        // Build the program ocl
        err = clBuildProgram(pg, 0, NULL, NULL, NULL, NULL);
        size_t len;
        char buffer[5000];
        clGetProgramBuildInfo(pg, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        if (err != CL_SUCCESS)  {
            printf("Program output:\n");
            printf("--------------\n");
            printf("%s\n", buffer);
            printf("--------------\n");
            throw std::runtime_error("Error: Failed to build the program" + kernel_name + "!\n" + string(opencl_error_to_str(err)));
        }
     }

    ~OpenCLBasic () {
        // Release objects in the opposite order of creation
        if (pointcloud_kernel) {
            if (VOXEL_DEBUG) std::cout << "Releasing pointcloud kernel " << pointcloud_kernel << std::endl;
            cl_int err = clReleaseKernel(pointcloud_kernel);
            SAMPLE_CHECK_ERRORS(err);
        }

        if (pointcloud_program) {
            if (VOXEL_DEBUG) std::cout << "Releasing pointcloud program " << pointcloud_program << std::endl;
            cl_int err = clReleaseProgram(pointcloud_program);
            SAMPLE_CHECK_ERRORS(err);
        }

        if(commands) {
            if (VOXEL_DEBUG) std::cout << "Releasing commands" << commands << std::endl;
            cl_int err = clReleaseCommandQueue(commands);
            SAMPLE_CHECK_ERRORS(err);
        }

        if(context) {
            if (VOXEL_DEBUG) std::cout << "Releasing context" << context << std::endl;
            cl_int err = clReleaseContext(context);
            SAMPLE_CHECK_ERRORS(err);
        }

        if (platforms) {
            if (VOXEL_DEBUG) std::cout << "Freeing platforms pointer " << (unsigned long) platforms << std::endl;
            free(platforms);
        }
    }
};


class ClBufferMaker
{
protected:
    cl_mem g;
    size_t size;
    cl_int err;
    bool need_copy;
    void* ptr;
public:
    ClBufferMaker():g((cl_mem)0), ptr(NULL), size(0), err(CL_SUCCESS), need_copy(false) {};
    virtual ~ClBufferMaker() {
        if (g != (cl_mem)0) {
            err = clReleaseMemObject(g);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    cl_mem get_gpu_mem() {
        return g;
    }

    void _make_buffer(void* cpu_mem, size_t expected_size, bool is_aligned, OpenCLBasic& ocl, bool debug,
                      const char* name, bool read_only) {
        /*
        Release if necessary previously allocated GPU buffer, create new one for given cpu memory block,
        check that it is aligned (if is_aligned is True)
        */
        if (g != (cl_mem)0) {  // buffer had already been allocated for different memory, release it
            if (debug) std::cout << "Releasing previously allocated " << string(name) << " clbuffer" << std::endl;
            err = clReleaseMemObject(g);
            SAMPLE_CHECK_ERRORS(err);
        }
        ptr = (void *) cpu_mem;
        size = expected_size;
        need_copy = ocl.using_nvidia_gpu || !is_aligned;
        cl_mem_flags flag = need_copy ? 0:CL_MEM_USE_HOST_PTR;  // nvidia doesn't allow use host pointer
        flag |= (read_only ? CL_MEM_READ_ONLY:CL_MEM_READ_WRITE);
        if (debug) std::cout << "Creating " << string(name) << " buffer" << std::endl;
        g = clCreateBuffer (ocl.context, flag, size, (need_copy ? NULL:ptr), &err);
        SAMPLE_CHECK_ERRORS(err);
        if (g == (cl_mem)0)
            throw Error("Failed to create " + string(name) + " buffer!");
        if (is_aligned && !verifyZeroCopyPtr(ptr, size))
            throw Error(string(name) + " ptr is not zero-copy aligned!");
    }

    void ensure_copy_from_gpu_to_cpu(OpenCLBasic& ocl) {
        if (need_copy) {
            err = clEnqueueReadBuffer(ocl.commands, g, CL_TRUE, 0, size, ptr, 0, NULL, NULL);
            SAMPLE_CHECK_ERRORS(err);
        }
    }
};


template<class T, int NDims> class ClBufferMakerFromNumpy: public ClBufferMaker
 {
public:
    cl_mem allocate_from_numpy(const numpy_boost<T, NDims>& m, const char* name, bool debug,
        OpenCLBasic& ocl, bool is_aligned=true, bool read_only=false) {
    /*  Takes care of allocating a gpu buffer for the numpy matrix m, and
        copying from m to g if necessary (if memory is not aligned or if an NVIDIA gpu
        is being used). If the internal pointer ptr and size size coincide with
        m.data() and its size, it is assumed that the buffer had already been allocated
        and no work is done (except copying memory if needed). Notice that no CPU memory
        allocation is done in any case, the data in the numpy array is used as is.
    */
        // Assert that elements are contiguous, so strides exactly equal the number of elements
        size_t expected_size = is_aligned ? ZEROCOPYSIZE(m):RAWSIZE(m);

        if (ptr != (void *) m.data() || size != expected_size) {  // need to allocate buffer
            _make_buffer((void *) m.data(), expected_size, is_aligned, ocl, debug, name, read_only);
            if (debug) {
                std::cout << string(name) << " array:" << std::endl;
                std::cout << "\tshape:";
                for (int i = 0; i < m.num_dimensions(); i++) {
                    std::cout << m.shape()[i];
                     if (i < m.num_dimensions() - 1) std::cout << ", ";
                     else std::cout << std::endl;
                }
                std::cout << "\tstrides:";
                for (int i = 0; i < m.num_dimensions(); i++) {
                    std::cout << m.strides()[i];
                     if (i < m.num_dimensions() - 1) std::cout << ", ";
                     else std::cout << std::endl;
                }
                std::cout << "\tsize: " << size << std::endl;
                std::cout << "\tIs aligned: " << (is_aligned ? "true":"false") << std::endl;
                std::cout << "\tPointer: " << (unsigned long) ptr << std::endl;
            }
        }
        else {
            if (debug) std::cout << string(name) << " already allocated, doing nothing" << std::endl;
        }

        // make sure the array is contiguous, we assume it when we compute the size
        for (int i = 0; i < m.num_dimensions() - 1; i++) {
            int tot_elem = 1;
            for (int j = i + 1; j < m.num_dimensions(); j++) tot_elem *= m.shape()[j];
            if (m.strides()[i] != tot_elem)
                throw std::runtime_error((boost::format("Elements in array %s are not contiguous") % name).str());
        }

        // in the case of nvidia or non-aligned data we need to copy the buffer
        if (need_copy) {
            err = clEnqueueWriteBuffer(ocl.commands, g, CL_TRUE, 0, size, ptr, 0, NULL, NULL);
            SAMPLE_CHECK_ERRORS(err);
        }
        return g;
    }
};


template<class T> class ClBufferMakerForArray: public ClBufferMaker
{
public:
    ClBufferMakerForArray():m_length(0) {};
    cl_mem allocate_for_flat_array(int length, const char* name, const bool debug, OpenCLBasic& ocl, bool read_only=false) {
        /*  Takes care of allocating an aligned cpu array of given length (in elements, not bytes)
            and the corresponding gpu buffer g. If the internal size size corresponds with the
            desired length, no work is done. NDIMS has to be 1 for this to work.
        */
        size_t expected_size = zeroCopySizeAlignment(length * sizeof(T), ocl.device_id);

        if (size != expected_size) {  // need to allocate memory and buffer
            aligned_free(ptr);
            ptr = aligned_malloc(expected_size, zeroCopyPtrAlignment(ocl.device_id));
            _make_buffer(ptr, expected_size, true, ocl, debug, name, read_only);
            m_length = length;
            if (debug) {
                std::cout << string(name) << " array:" << std::endl;
                std::cout << "\tnum elems:" << m_length << std::endl;
                std::cout << "\tsize: " << size << std::endl;
                std::cout << "\tIs aligned: true" << std::endl;
                std::cout << "\tPointer: " << (unsigned long) ptr << std::endl;
            }
        }
        else {
            if (debug) std::cout << string(name) << " already allocated, doing nothing" << std::endl;
        }

        // notice we don't write the memory from cpu to gpu even if it is NVIDIA: the memory
        // has just been allocated and contains garbage.

        return g;
    }

    void ensure_copy_to_gpu(OpenCLBasic& ocl) {
        if (need_copy) {
            err = clEnqueueWriteBuffer(ocl.commands, g, CL_TRUE, 0, size, ptr, 0, NULL, NULL);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    T* get_ptr() {
        return (T*) ptr;
    }

    int get_length() {
        return m_length;
    }

    virtual ~ClBufferMakerForArray() {
        if (VOXEL_DEBUG) std::cout << "Aligned-freeing pointer " << (unsigned long) ptr << std::endl;
        aligned_free(ptr);
    }
private:
    int m_length;
};


class VoxelRaytraceBoostWrapperGpu
{
private:
    OpenCLBasic ocl;
    ClBufferMakerFromNumpy<boost::uint32_t, 3> volume_buf;
    ClBufferMakerFromNumpy<boost::uint8_t, 2> column_counts_buf;
    ClBufferMakerFromNumpy<boost::uint8_t, 2> known_buf;
    ClBufferMakerFromNumpy<boost::int16_t, 2> endpoints_buf;
    ClBufferMakerFromNumpy<boost::int16_t, 2> line_defs_buf;
    ClBufferMakerFromNumpy<boost::int32_t, 1> idx_buf;
    ClBufferMakerFromNumpy<float, 1> ranges_buf;
    ClBufferMakerForArray<boost::uint64_t> packed_markings_buf;
    float xy_resolution;
    float z_resolution;
    ClBufferMakerForArray<boost::uint32_t> bit_check_masks_buf;
    ClBufferMakerForArray<boost::uint32_t> bit_mark_single_weight_buf;
    ClBufferMakerForArray<boost::uint32_t> bit_clear_masks_buf;
    short n_voxels_per_uint32;

    // cumulative profiling times for loading, executing and reading on gpu
    int _memory_in_time = 0;
    int _kernel_time = 0;
    int _memory_out_time = 0;

public:
    VoxelRaytraceBoostWrapperGpu(float _xy_resolution, float _z_resolution,
        unsigned int marking_threshold, string const &kernel_full_name):
        ocl(kernel_full_name),
        xy_resolution(_xy_resolution),
        z_resolution(_z_resolution)
    {
        // pre-calculate bit masks for setting, clearing and checking voxels in bit format
        if (marking_threshold == 0 || marking_threshold >= 256) throw std::runtime_error("marking_threshold has go be between 1 and 255");
        short n_bits_per_voxel = 1;
        unsigned int marking_threshold_copy = marking_threshold;
        while(marking_threshold_copy >>= 1) ++n_bits_per_voxel;
        n_voxels_per_uint32 = 32 / n_bits_per_voxel;  // notice 32 need not be divisible by n_bits_per_voxel

        cl_mem bit_check_masks_g = bit_check_masks_buf.allocate_for_flat_array(MAX_VOXELS + 1, "bit_check_masks", VOXEL_DEBUG, ocl, true);
        cl_mem bit_mark_single_weight_g = bit_mark_single_weight_buf.allocate_for_flat_array(MAX_VOXELS + 1, "mark_weights", VOXEL_DEBUG, ocl, true);
        cl_mem bit_clear_masks_g = bit_clear_masks_buf.allocate_for_flat_array(MAX_VOXELS + 1, "bit_clear_masks", VOXEL_DEBUG, ocl, true);
        cl_int err = CL_SUCCESS;
        err = clFinish(ocl.commands);
        SAMPLE_CHECK_ERRORS(err);

        for (int i=0; i<=MAX_VOXELS; i++) {
            short bit_idx = n_bits_per_voxel * (i % n_voxels_per_uint32);
            bit_check_masks_buf.get_ptr()[i] = ((boost::uint32_t) marking_threshold) << bit_idx;  // for checking if above threshold, with bitwise and
            bit_mark_single_weight_buf.get_ptr()[i] = ((boost::uint32_t) 1) << bit_idx;  // for marking, by adding
            bit_clear_masks_buf.get_ptr()[i] = ~(((((boost::uint32_t) 1) << n_bits_per_voxel) - 1) << bit_idx);  // for clearing, with bitwise and
        }
        bit_check_masks_buf.ensure_copy_to_gpu(ocl);
        bit_mark_single_weight_buf.ensure_copy_to_gpu(ocl);
        bit_clear_masks_buf.ensure_copy_to_gpu(ocl);
        err = clFinish(ocl.commands);
        SAMPLE_CHECK_ERRORS(err);
    }

    ~VoxelRaytraceBoostWrapperGpu() {}

    numpy_boost<boost::int16_t, 2> raytrace_pointcloud(
                              const numpy_boost<boost::uint32_t, 3>& volume,
                              int n_voxels,
                              const numpy_boost<boost::uint8_t, 2>& column_counts,
                              const numpy_boost<boost::uint8_t, 2>& known,
                              const numpy_boost<boost::int16_t, 2>& endpoints,
                              const numpy_boost<boost::int16_t, 1>& origin,
                              float raytrace_range, float obstacle_range,
                              //short max_z_clearing_for_known,
                              bool do_clearing, bool do_marking)
    {
        cl_int err = CL_SUCCESS;
        const bool debug = VOXEL_DEBUG;

        const short x0 = origin[0];
        const short y0 = origin[1];
        const short z0 = origin[2];
        short max_x = volume.shape()[1];
        short max_y = volume.shape()[0];
        short max_z = n_voxels;
        if (x0 < 0 || x0 >= max_x || y0 < 0 || y0 >= max_y ||
                z0 < 0 || z0 >= max_z) {
            int empty_dims[] = {0};
            numpy_boost<boost::int16_t, 2> numpy_empty_markings(empty_dims);
            return numpy_empty_markings;
        }

        cl_mem volume_g = volume_buf.allocate_from_numpy(volume, "volume", debug, ocl);
        cl_mem known_g = known_buf.allocate_from_numpy(known, "known", debug, ocl);
        cl_mem column_counts_g = column_counts_buf.allocate_from_numpy(column_counts, "column_counts", debug, ocl);
        cl_mem endpoints_g = endpoints_buf.allocate_from_numpy(endpoints, "endpoints", debug, ocl, true, true);
        cl_mem packed_markings_g = packed_markings_buf.allocate_for_flat_array(endpoints.shape()[0], "packed_markings", debug, ocl);
        cl_mem bit_check_masks_g = bit_check_masks_buf.get_gpu_mem();
        cl_mem bit_clear_masks_g = bit_clear_masks_buf.get_gpu_mem();
        err = clFinish(ocl.commands);
        SAMPLE_CHECK_ERRORS(err);

        const unsigned int dim_0 = volume.shape()[0];
        const unsigned int dim_1 = volume.shape()[1];
        const unsigned int dim_2 = volume.shape()[2];
        const unsigned int stride_0 = volume.strides()[0];
        const unsigned int stride_1 = volume.strides()[1];
        const uint8_t do_clearing_g = do_clearing;
        const uint8_t do_marking_g = do_marking;
        const float sq_xy_resolution = xy_resolution * xy_resolution;
        const float sq_z_resolution = z_resolution * z_resolution;
        const short max_z_clearing_for_known_g = 0; // max_z_clearing_for_known;
        const short n_voxels_per_uint32_g = n_voxels_per_uint32;

        cl_kernel* kernel = &ocl.pointcloud_kernel;
        err = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &volume_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 1, sizeof(cl_mem), &endpoints_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 2, sizeof(cl_mem), &known_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 3, sizeof(cl_mem), &column_counts_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 4, sizeof(short), &x0);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 5, sizeof(short), &y0);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 6, sizeof(short), &z0);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 7, sizeof(unsigned int), &dim_0);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 8, sizeof(unsigned int), &dim_1);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 9, sizeof(unsigned int), &n_voxels);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 10, sizeof(unsigned int), &stride_0);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 11, sizeof(unsigned int), &stride_1);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 12, sizeof(float), &raytrace_range);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 13, sizeof(float), &obstacle_range);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 14, sizeof(uint8_t), &do_clearing_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 15, sizeof(uint8_t), &do_marking_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 16, sizeof(cl_mem), &packed_markings_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 17, sizeof(float), &sq_xy_resolution);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 18, sizeof(float), &sq_z_resolution);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 19, sizeof(cl_mem), &bit_check_masks_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 20, sizeof(cl_mem), &bit_clear_masks_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 21, sizeof(short), &n_voxels_per_uint32_g);
        SAMPLE_CHECK_ERRORS(err);
        err = clSetKernelArg(*kernel, 22, sizeof(short), &max_z_clearing_for_known_g);
        SAMPLE_CHECK_ERRORS(err);

        size_t global_work_size[1] = {endpoints.shape()[0]};
        err = clEnqueueNDRangeKernel(ocl.commands, ocl.pointcloud_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
        SAMPLE_CHECK_ERRORS(err);
        err = clFinish(ocl.commands);
        SAMPLE_CHECK_ERRORS(err);

        volume_buf.ensure_copy_from_gpu_to_cpu(ocl);
        known_buf.ensure_copy_from_gpu_to_cpu(ocl);
        column_counts_buf.ensure_copy_from_gpu_to_cpu(ocl);
        endpoints_buf.ensure_copy_from_gpu_to_cpu(ocl);
        packed_markings_buf.ensure_copy_from_gpu_to_cpu(ocl);
        err = clFinish(ocl.commands);
        SAMPLE_CHECK_ERRORS(err);

        int w_dims[] = { 1 };
        numpy_boost<boost::int16_t, 1> weights(w_dims);
        weights[0] = 1;
        return filter_and_process_markings(packed_markings_buf, do_marking, volume, n_voxels, column_counts, known, weights);
    }

    numpy_boost<boost::int16_t, 2> filter_and_process_markings(ClBufferMakerForArray<boost::uint64_t> &packed_markings_buf,
                                                               bool do_marking,
                                                               const numpy_boost<boost::uint32_t, 3>& volume,
                                                               int n_voxels,
                                                               const numpy_boost<boost::uint8_t, 2>& column_counts,
                                                               const numpy_boost<boost::uint8_t, 2>& known,
                                                               const numpy_boost<boost::int16_t, 1>& weights) {
    /*  Takes packed markings (where each mark, defined by int16 validity flag, x, y, and z, is packed into one uint64),
        For valid marks, it replaces the validity flag with the marking weight (in-place), and does marking on the volume if necessary.
        It returns a numpy array with the valid marks.
    */
        // collect valid marks and replace validity with weight for the valid ones
        std::vector<boost::uint64_t> valid_packed_markings;
        std::vector<boost::uint64_t> bad_markings;
        std::vector<size_t> bad_idx;
        int w_step = (weights.size() == 1) ? 0 : 1;  // if weights only has one element, use it repeatedly
        for (boost::uint64_t* p = packed_markings_buf.get_ptr(), w_idx = 0;
             p < packed_markings_buf.get_ptr() + packed_markings_buf.get_length(); ++p, w_idx+=w_step) {
            short valid = *(short*)p;
            short y = *((short*)p + 1);
            short x = *((short*)p + 2);
            short z = *((short*)p + 3);
            if ((valid != 0 && valid != 1) || (valid != 0 && (x < 0 || y < 0 || z < 0 || x >= volume.shape()[1] ||
                y >= volume.shape()[0] || z >= n_voxels))) {
                bad_markings.push_back(*p);
                bad_idx.push_back(p - packed_markings_buf.get_ptr());
                if (PRINT_DEBUG){
                    std::cerr << "Error in filter_and_process_markings!" << std::endl;
                    std::string error_msg = (boost::format("Bad voxel (idx %i of %i) has validity %i, x=%i, y=%i, z=%i") %
                        (p - packed_markings_buf.get_ptr()) %
                        packed_markings_buf.get_length() % valid % x % y % z).str();
                    std::cerr << error_msg.c_str() << std::endl;
                }
            }

            if (valid) {  // look at the first short in the int64, which is the valid/not valid flag
                *(short*)p = weights[w_idx];  // replace validity with weight, in-place
                valid_packed_markings.push_back(*p);
            }
        }

        if (bad_markings.size() > 0) {
            std::string error_msg = (boost::format("%i bad voxels in filter_and_process_markings:)") % bad_markings.size()).str();
            for (int j=0; j < bad_markings.size(); ++j) {
                boost::uint64_t* pb = bad_markings.data() + j;
                short bad_valid = *(short*)pb;
                short bad_y = *((short*)pb + 1);
                short bad_x = *((short*)pb + 2);
                short bad_z = *((short*)pb + 3);
                error_msg += (boost::format(" at %i of %i, valid=%i, (x,y,z)= (%i, %i, %i);") % bad_idx[j] % packed_markings_buf.get_length() %
                    bad_valid % bad_x % bad_y % bad_z).str();
                }

            std::cerr << "[voxel_raytrace_gpu][ERROR] " ;
            std::cerr << error_msg.c_str() << std::endl;
            throw std::runtime_error(error_msg.c_str());
        }

        numpy_boost<boost::uint32_t, 3>& writeable_volume = *const_cast<numpy_boost<boost::uint32_t, 3>* > (&volume);
        numpy_boost<boost::uint8_t, 2>& writeable_column_counts = *const_cast<numpy_boost<boost::uint8_t, 2>* > (&column_counts);
        numpy_boost<boost::uint8_t, 2>& writeable_known = *const_cast<numpy_boost<boost::uint8_t, 2>* > (&known);
        return process_markings(valid_packed_markings, do_marking, writeable_volume,
                                writeable_column_counts, writeable_known,
                                n_voxels_per_uint32, bit_check_masks_buf.get_ptr(), bit_mark_single_weight_buf.get_ptr());
    }

    numpy_boost<boost::int16_t, 2> get_voxels(const numpy_boost<boost::uint32_t, 3>& volume,
                                              int n_voxels,
                                              const numpy_boost<boost::uint8_t, 2>& column_counts) {
    /*
    Calls get_voxels_impl in raytrace_utils.hpp, se docstring there
    */
        return get_voxels_implementation(volume, n_voxels, column_counts, n_voxels_per_uint32, bit_check_masks_buf.get_ptr());
    }

    void print_debug_statistics() {
        std::cout <<
            " Write to gpu="  << double(this->_memory_in_time)/1e6 <<
            "s Kernel=" << double(this->_kernel_time)/1e6 <<
            "s Read and postprocess=" << double(this->_memory_out_time)/1e6 <<
            "s Total=" << double(this->_memory_in_time + this->_kernel_time + this->_memory_out_time)/1e6 <<
            "s\n";
    }

};
