#define swap(a, b)  { a ^= b; b^= a; a ^= b; }
#define swap_float(a, b) { float temp = a; a = b; b = temp;}  // opencl doesn't like xoring floats

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define clear_point_delayed() {			\
    uint old_val, col_old_val;  \
    int coarse_idx, flat_idx;   \
    bool was_above_threshold;   \
    short x_tmp = x;			\
    short y_tmp = y;			\
    short z_tmp = z;			\
    if (swap_xz) {              \
        swap(x_tmp, z_tmp);	    \
    }                           \
    if (swap_xy) {              \
        swap(x_tmp, y_tmp);	    \
    }                           \
    swap(x_tmp, last_x_clear);      \
    swap(y_tmp, last_y_clear);      \
    swap(z_tmp, last_z_clear);      \
    if (x_tmp != SHRT_MAX) {        \
    if (z_tmp < n_voxels) {	\
    coarse_idx = z_tmp / n_voxels_per_uint32;  \
    __global uint* volume_data = (__global uint*) (&volume[y_tmp*stride0 + x_tmp*stride1]); \
    old_val = atomic_and(&volume_data[coarse_idx], bit_clear_masks[z_tmp]); \
    was_above_threshold = (old_val & bit_check_masks[z_tmp]) == bit_check_masks[z_tmp]; \
    flat_idx = y_tmp * dim_1 + x_tmp; \
    if (z_tmp <= max_z_clearing_for_known) known[flat_idx] = 1; \
    if (was_above_threshold) { \
            __global uint* column_ptr = (__global uint*)(&column_counts[flat_idx-(flat_idx%4)]); \
            col_old_val = atomic_sub(column_ptr, (1 << (8 * (flat_idx%4))));	\
    } \
    } \
    } \
}

__kernel void raytrace_pointcloud(__global uint* volume,
                                  const __global ushort* endpoint,
                                  __global uchar* known,
                                  __global uchar* column_counts,
                                  short x0,
                                  short y0,
                                  short z0,
                                  unsigned int dim_0,
                                  unsigned int dim_1,
                                  unsigned int n_voxels,
                                  const unsigned int stride0,
                                  const unsigned int stride1,
                                  const float raytrace_range,
                                  const float obstacle_range,
                                  const uchar do_clearing,
                                  const uchar is_marking,
                                  __global short4* packed_markings,
                                  const float sq_xy_resolution,
                                  const float sq_z_resolution,
                                 __constant uint* bit_check_masks,
                                 __constant uint* bit_clear_masks,
                                 const short n_voxels_per_uint32,
                                 const short max_z_clearing_for_known
                                 )
{
    int gid = get_global_id(0);
    short x1 = endpoint[3 * gid];
    short y1 = endpoint[3 * gid + 1];
    short z1 = endpoint[3 * gid + 2];
    float sq_obstacle_range = obstacle_range * obstacle_range;
    float sq_raytrace_range = raytrace_range * raytrace_range;
    float x2_resolution = sq_xy_resolution;
    float y2_resolution = sq_xy_resolution;
    float z2_resolution = sq_z_resolution;

    short dx = abs_diff(x1, x0);
    short dy = abs_diff(y1, y0);
    short dz = abs_diff(z1, z0);

    unsigned int max_x = dim_1;
    unsigned int max_y = dim_0;
    unsigned int max_z = n_voxels;

    // we make x always the longest dimension
    bool swap_xy = (dy > dx);
    if (swap_xy) {
        swap(x0, y0);
        swap(x1, y1);
        swap(dx, dy);
        swap(max_x, max_y);
        // not swapping x and y resolution since they are the same
    }
    bool swap_xz = (dz > dx);
    if (swap_xz) {
        swap(x0, z0);
        swap(x1, z1);
        swap(dx, dz);
        swap(max_x, max_z);
        swap_float(x2_resolution, z2_resolution);
    }

    // direction of line
    short step_x = (x0 > x1) ? -1:1;
    short step_y = (y0 > y1) ? -1:1;
    short step_z = (z0 > z1) ? -1:1;

    short x_bound = step_x > 0 ? max_x:-1;
    short y_bound = step_y > 0 ? max_y:-1;
    short z_bound = step_z > 0 ? max_z:-1;

    //drift controls when to step in the y and z planes
    //starting value is centered
    short drift_xy  = (dx / 2) + dy;  // + dy to ensure the fist step is always in the longest direction
    short drift_xz  = (dx / 2) + dz;  // + dz to ensure the fist step is always in the longest direction

    short y = y0;
    short z = z0;
    short end_x = x1 + step_x;
    float r2;
    // step through longest delta (which we have swapped to x)
    short x;
    short last_x_clear = SHRT_MAX;
    short last_y_clear = SHRT_MAX;
    short last_z_clear = SHRT_MAX;
    if (do_clearing) {
        for (x = x0; x != end_x; x += step_x) {
            if (x == x_bound) break;
            r2 = ((float)(x2_resolution*(x-x0)*(x-x0))) +
                 ((float)(y2_resolution*(y-y0)*(y-y0))) +
                 ((float)(z2_resolution*(z-z0)*(z-z0)));
            if (r2 > sq_raytrace_range) break;
	        clear_point_delayed();

            //update progress in other planes
            drift_xy = drift_xy - dy;
            drift_xz = drift_xz - dz;

            //step in y plane
            if (drift_xy < 0) {
                y += step_y;
                if (y == y_bound) break;
                drift_xy += dx;
	    	    clear_point_delayed();
            }

            //same in z
            if (drift_xz < 0) {
                z += step_z;
                if (z == z_bound) break;
                drift_xz += dx;
        		clear_point_delayed();
            }
        }
    }
    r2 = ((float)(x2_resolution*(x1-x0)*(x1-x0))) +
         ((float)(y2_resolution*(y1-y0)*(y1-y0))) +
         ((float)(z2_resolution*(z1-z0)*(z1-z0)));

    if (swap_xz) {
        swap(x1, z1);
    }
    if (swap_xy) {
        swap(x1, y1);
    }
    bool valid_mark = !(x1 >= dim_1 || x1 < 0 || y1 >= dim_0 || y1 < 0 || z1 >= n_voxels || z1 < 0 || r2 > sq_obstacle_range);
    if (valid_mark) {
        if (do_clearing && !is_marking) clear_point_delayed();  // flush the last point to clear only if we are not going to mark it!
        packed_markings[gid] = (short4)(1, y1, x1, z1);  // valid mark
    }
    else {
        if (do_clearing) clear_point_delayed();  // flush the last point to clear
        packed_markings[gid] = (short4)(0, y1, x1, z1);  // invalid mark
    }
}
