#define swap(a, b)  { a ^= b; b^= a; a ^= b; }

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void raytrace_pointcloud_marking(__global ulong* volume,
                                          const __global ushort* endpoint,
                                          __global uchar* known,
                                          __global uchar* column_counts,
                                          const short x0,
                                          const short y0,
                                          const short z0,
                                          const unsigned int dim_0,
                                          const unsigned int dim_1,
                                          const unsigned int n_voxels,
                                          const unsigned int stride0,
                                          const unsigned int stride1,
                                          const float raytrace_range,
                                          const float obstacle_range,
                                          const uchar do_clearing,
                                          const uchar do_marking,
                                          __global uchar* valid_markings,
                                          const float sq_xy_resolution,
                                          const float sq_z_resolution)
{
    int gid = get_global_id(0);

    short x1 = endpoint[3 * gid];
    short y1 = endpoint[3 * gid + 1];
    short z1 = endpoint[3 * gid + 2];
    float sq_obstacle_range = obstacle_range * obstacle_range;

    float r2 = ((float) sq_xy_resolution*(x1-x0)*(x1-x0)) +
        ((float) sq_xy_resolution*(y1-y0)*(y1-y0)) +
        ((float) sq_z_resolution*(z1-z0)*(z1-z0));

    if (x1 >= dim_1 || x1 < 0 || y1 >= dim_0 || y1 < 0 || z1 >= n_voxels || z1 < 0)
        valid_markings[gid] = 0;
    else if (r2 > sq_obstacle_range)
        valid_markings[gid] = 0;
    else {
        valid_markings[gid] = 1;
        if (do_marking) {
            __global uint* volume_ptr = (__global uint*) (&volume[y1*stride0 + x1*stride1]);
            int coarse_idx = z1 / 32;
            ulong val = 1 << (z1 % 32); // mark
            uint old_val = atomic_or(&volume_ptr[coarse_idx], val);
            int idx = y1*dim_1 + x1;
            known[idx] = 1;
            // count columns only when the bit at that location is empty
            if ((old_val & val) == 0) { 
                __global uint* column_ptr = (__global uint*)(&column_counts[idx-(idx%4)]);
                atomic_add(column_ptr, (1 << (8 * (idx%4))));
            }
        }
    }
}
