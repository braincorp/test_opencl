#include "voxel_raytrace_gpu.hpp"

BOOST_PYTHON_MODULE(_wrapper_test_opencl_module)
{
    import_array();
    numpy_boost_python_register_type<float, 1>();
    numpy_boost_python_register_type<boost::uint8_t, 2>();
    numpy_boost_python_register_type<boost::uint8_t, 1>();
    numpy_boost_python_register_type<boost::int32_t, 1>();
    numpy_boost_python_register_type<boost::int16_t, 1>();
    numpy_boost_python_register_type<boost::int16_t, 2>();
    numpy_boost_python_register_type<boost::uint32_t, 3>();
    numpy_boost_python_register_type<boost::uint8_t, 3>();
    numpy_boost_python_register_type<float, 3>();


    py::register_exception_translator<std::runtime_error>(&translate_exception);
    py::def("has_gpu_support", has_gpu_support, py::arg("debug"));
    py::class_<VoxelRaytraceBoostWrapperGpu>("VoxelRaytraceBoostWrapperGpu", boost::python::init<float, float, unsigned int, string const &>())
            .def("raytrace_pointcloud", &VoxelRaytraceBoostWrapperGpu::raytrace_pointcloud)
            .def("print_debug_statistics", &VoxelRaytraceBoostWrapperGpu::print_debug_statistics)
    ;
}
