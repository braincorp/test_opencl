#ifndef _INTEL_OPENCL_SAMPLE_BASIC_HPP_
#define _INTEL_OPENCL_SAMPLE_BASIC_HPP_


#include <cstdlib>
#include <cassert>
#include <string>
#include <stdexcept>
#include <sstream>
#include <typeinfo>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <exception>
#include <iostream>
#include <CL/cl.h>

using std::cerr;
using std::string;

// Returns textual representation of the OpenCL error code.
const char* opencl_error_to_str (cl_int error);

// Base class for all exception in samples
class Error : public std::runtime_error
{
public:
    Error (const string& msg) :
        std::runtime_error(msg)
    {
    }
};

// Allocates piece of aligned memory
// alignment should be a power of 2
// Out of memory situation is reported by throwing std::bad_alloc exception
void* aligned_malloc (size_t size, size_t alignment);

// Deallocates memory allocated by aligned_malloc
void aligned_free (void *aligned);

// Report about an OpenCL problem.
// Macro is used instead of a function here
// to report source file name and line number.
#define SAMPLE_CHECK_ERRORS(ERR)                        \
    if(ERR != CL_SUCCESS)                               \
    {                                                   \
        throw Error(                                    \
            "OpenCL error " +                           \
            string(opencl_error_to_str(ERR)) +                  \
            " happened in file " + to_str(__FILE__) +   \
            " at line " + to_str(__LINE__) + "."        \
        );                                              \
    }


// Query for several frequently used device/kernel capabilities
// Recomended alignment in bytes for memory used in clCreateBuffer with CL_MEM_USE_HOST_PTR.
// Returned value is sufficiently large to enable zero-copy behaviour on Intel Processor Graphics.
cl_uint zeroCopyPtrAlignment (cl_device_id device = 0);

// Extends required buffer size to a value which is sufficient to enable
// zero-copy behaviour for buffers created with CL_MEM_USE_HOST_PTR on Intel Processor Graphics.
size_t zeroCopySizeAlignment (size_t requiredSize, cl_device_id device = 0);

// Verifies if ptr and sizeOfContentOfPtr satisfy alignment rules which
// should be held to enable zero-copy behaviour on Intel Processor Graphics in case if an OpenCL buffer
// is created using CL_MEM_USE_HOST_PTR flag and provided memory area.
bool verifyZeroCopyPtr (void* ptr, size_t sizeOfContentsOfPtr);

char *ReadSources(const char *fileName);

cl_platform_id GetIntelOCLPlatform();

void BuildFailLog( cl_program program, cl_device_id device_id );



// Convert from a value of a given type to string with optional formatting.
// T should have operator<< defined to be written to stream.
template <typename T>
string to_str (const T x, std::streamsize width = 0, char fill = ' ')
{
    using namespace std;
    ostringstream os;
    os << setw(width) << setfill(fill) << x;
    if(!os)
    {
        throw Error("Cannot represent object as a string");
    }
    return os.str();
}

#endif // _INTEL_OPENCL_SAMPLE_BASIC_HPP_

