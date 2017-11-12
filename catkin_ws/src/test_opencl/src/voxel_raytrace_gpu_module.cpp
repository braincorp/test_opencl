#include "voxel_raytrace_gpu.hpp"

BOOST_PYTHON_MODULE(_wrapper_test_opencl_module)
{
    import_array();
    py::register_exception_translator<std::runtime_error>(&translate_exception);
    py::def("has_gpu_support", has_gpu_support, py::arg("debug"));
    py::class_<VoxelRaytraceBoostWrapperGpu, boost::noncopyable>(
        "TestOpenCLVoxelRaytraceWrapperGpu", boost::python::init<float, float, unsigned int, string const &>())
            .def("raytrace_pointcloud", &VoxelRaytraceBoostWrapperGpu::raytrace_pointcloud)
            .def("raytrace", &VoxelRaytraceBoostWrapperGpu::raytrace)
            .def("raytrace", &VoxelRaytraceBoostWrapperGpu::raytrace_defaults)
            .def("get_voxels", &VoxelRaytraceBoostWrapperGpu::get_voxels)
            .def("print_debug_statistics", &VoxelRaytraceBoostWrapperGpu::print_debug_statistics)
    ;
}
