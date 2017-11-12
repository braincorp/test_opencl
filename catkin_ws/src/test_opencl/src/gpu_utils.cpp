#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include <assert.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <exception>
#include <vector>
#include <cerrno>
#include <CL/cl_gl.h>

#pragma warning( push )

#include <sys/time.h>
#include <unistd.h>
#include <libgen.h>
#include <boost/format.hpp>
#include "gpu_utils.h"

using std::string;

#define GPU_UTILS_DEBUG false

const char* opencl_error_to_str (cl_int error)
{
#define CASE_CL_CONSTANT(NAME) case NAME: return #NAME;

    // Suppose that no combinations are possible.
    // TODO: Test whether all error codes are listed here
    switch(error)
    {
        CASE_CL_CONSTANT(CL_SUCCESS)
        CASE_CL_CONSTANT(CL_DEVICE_NOT_FOUND)
        CASE_CL_CONSTANT(CL_DEVICE_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_COMPILER_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        CASE_CL_CONSTANT(CL_OUT_OF_RESOURCES)
        CASE_CL_CONSTANT(CL_OUT_OF_HOST_MEMORY)
        CASE_CL_CONSTANT(CL_PROFILING_INFO_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_MEM_COPY_OVERLAP)
        CASE_CL_CONSTANT(CL_IMAGE_FORMAT_MISMATCH)
        CASE_CL_CONSTANT(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        CASE_CL_CONSTANT(CL_BUILD_PROGRAM_FAILURE)
        CASE_CL_CONSTANT(CL_MAP_FAILURE)
        CASE_CL_CONSTANT(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        CASE_CL_CONSTANT(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        CASE_CL_CONSTANT(CL_INVALID_VALUE)
        CASE_CL_CONSTANT(CL_INVALID_DEVICE_TYPE)
        CASE_CL_CONSTANT(CL_INVALID_PLATFORM)
        CASE_CL_CONSTANT(CL_INVALID_DEVICE)
        CASE_CL_CONSTANT(CL_INVALID_CONTEXT)
        CASE_CL_CONSTANT(CL_INVALID_QUEUE_PROPERTIES)
        CASE_CL_CONSTANT(CL_INVALID_COMMAND_QUEUE)
        CASE_CL_CONSTANT(CL_INVALID_HOST_PTR)
        CASE_CL_CONSTANT(CL_INVALID_MEM_OBJECT)
        CASE_CL_CONSTANT(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CASE_CL_CONSTANT(CL_INVALID_IMAGE_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_SAMPLER)
        CASE_CL_CONSTANT(CL_INVALID_BINARY)
        CASE_CL_CONSTANT(CL_INVALID_BUILD_OPTIONS)
        CASE_CL_CONSTANT(CL_INVALID_PROGRAM)
        CASE_CL_CONSTANT(CL_INVALID_PROGRAM_EXECUTABLE)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_NAME)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_DEFINITION)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL)
        CASE_CL_CONSTANT(CL_INVALID_ARG_INDEX)
        CASE_CL_CONSTANT(CL_INVALID_ARG_VALUE)
        CASE_CL_CONSTANT(CL_INVALID_ARG_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_ARGS)
        CASE_CL_CONSTANT(CL_INVALID_WORK_DIMENSION)
        CASE_CL_CONSTANT(CL_INVALID_WORK_GROUP_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_WORK_ITEM_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_GLOBAL_OFFSET)
        CASE_CL_CONSTANT(CL_INVALID_EVENT_WAIT_LIST)
        CASE_CL_CONSTANT(CL_INVALID_EVENT)
        CASE_CL_CONSTANT(CL_INVALID_OPERATION)
        CASE_CL_CONSTANT(CL_INVALID_GL_OBJECT)
        CASE_CL_CONSTANT(CL_INVALID_BUFFER_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_MIP_LEVEL)
        CASE_CL_CONSTANT(CL_INVALID_GLOBAL_WORK_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_PROPERTY)
		CASE_CL_CONSTANT(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR)
    default:
        return boost::str(boost::format("UNKNOWN ERROR CODE %i") % error).c_str();
}

#undef CASE_CL_CONSTANT
}

void* aligned_malloc (size_t size, size_t alignment)
{
    // a number of requirements should be met
    assert(alignment > 0);
    assert((alignment & (alignment - 1)) == 0); // test for power of 2
    assert(alignment >= sizeof(void*));
    assert(size >= sizeof(void*));
    assert(size/sizeof(void*)*sizeof(void*) == size);

    // allocate extra memory and convert to size_t to perform calculations
    char* orig = new char[size + alignment + sizeof(void*)];
    // calculate an aligned position in the allocated region
    // assumption: (size_t)orig does not lose lower bits
    char* aligned =
        orig + (
        (((size_t)orig + alignment + sizeof(void*)) & ~(alignment - 1)) -
        (size_t)orig
        );
    // save the original pointer to use it in aligned_free
    *((char**)aligned - 1) = orig;
    return aligned;
}


void aligned_free (void *aligned)
{
    if(!aligned)return; // behaves as delete: calling with 0 is NOP
    delete [] *((char**)aligned - 1);
}


cl_uint zeroCopyPtrAlignment (cl_device_id device)
{
    // Please refer to Intel Zero Copy Tutorial and OpenCL Performance Guide
    return 4096;
}


size_t zeroCopySizeAlignment (size_t requiredSize, cl_device_id device)
{
    // Please refer to Intel Zero Copy Tutorial and OpenCL Performance Guide
    // The following statement rounds requiredSize up to the next 64-byte boundary
    return requiredSize + (~requiredSize + 1) % 64;   // or even shorter: requiredSize + (-requiredSize) % 64
}


bool verifyZeroCopyPtr (void* ptr, size_t sizeOfContentsOfPtr)
{
    return                                  // To enable zero-copy for buffer objects
        (unsigned long)ptr % 4096  ==  0   // pointer should be aligned to 4096 bytes boundary
        &&                                  // and
        sizeOfContentsOfPtr % 64  ==  0;    // size of memory should be aligned to 64 bytes boundary.
}


cl_uint requiredOpenCLAlignment (cl_device_id device)
{
    cl_uint result = 0;
    cl_int err = clGetDeviceInfo(
        device,
        CL_DEVICE_MEM_BASE_ADDR_ALIGN,
        sizeof(result),
        &result,
        0
        );
    SAMPLE_CHECK_ERRORS(err);
    assert(result%8 == 0);
    return result/8;    // clGetDeviceInfo returns value in bits, convert it to bytes
}

cl_platform_id GetIntelOCLPlatform()
{
    cl_platform_id pPlatforms[10] = { 0 };
    char pPlatformName[128] = { 0 };

    cl_uint uiPlatformsCount = 0;
    cl_int err = clGetPlatformIDs(10, pPlatforms, &uiPlatformsCount);
    for (cl_uint ui = 0; ui < uiPlatformsCount; ++ui)
    {
        err = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
        if ( err != CL_SUCCESS )
        {
            printf("ERROR: Failed to retreive platform vendor name.\n");
            return NULL;
        }

        if (!strcmp(pPlatformName, "Intel(R) OpenCL"))
            return pPlatforms[ui];
    }

    return NULL;
}


void BuildFailLog( cl_program program,
                  cl_device_id device_id )
{
    size_t paramValueSizeRet = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &paramValueSizeRet);

    char* buildLogMsgBuf = (char *)malloc(sizeof(char) * paramValueSizeRet + 1);
    if( buildLogMsgBuf )
    {
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, paramValueSizeRet, buildLogMsgBuf, &paramValueSizeRet);
        buildLogMsgBuf[paramValueSizeRet] = '\0';    // mark end of message string

        printf("Build Log:\n");
        puts(buildLogMsgBuf);
        fflush(stdout);

        free(buildLogMsgBuf);
    }
}


char *ReadSources(const char *fileName)
{
    FILE *file = fopen(fileName, "rb");
    if (!file)
    {
        printf("ERROR: Failed to open file '%s'\n", fileName);
        return NULL;
    }

    if (fseek(file, 0, SEEK_END))
    {
        printf("ERROR: Failed to seek file '%s'\n", fileName);
        fclose(file);
        return NULL;
    }

    long size = ftell(file);
    if (size == 0)
    {
        printf("ERROR: Failed to check position on file '%s'\n", fileName);
        fclose(file);
        return NULL;
    }

    rewind(file);

    char *src = (char *)malloc(sizeof(char) * size + 1);
    if (!src)
    {
        printf("ERROR: Failed to allocate memory for file '%s'\n", fileName);
        fclose(file);
        return NULL;
    }

    if (GPU_UTILS_DEBUG) printf("Reading file '%s' (size %ld bytes)\n", fileName, size);
    size_t res = fread(src, 1, sizeof(char) * size, file);
    if (res != sizeof(char) * size)
    {
        printf("ERROR: Failed to read file '%s'\n", fileName);
        fclose(file);
        free(src);
        return NULL;
    }

    src[size] = '\0'; /* NULL terminated */
    fclose(file);

    return src;
}
