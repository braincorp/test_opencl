#define swap(a, b)  { a ^= b; b^= a; a ^= b; }
#define swap_float(a, b)  { float temp = b; b = a; a = temp; }

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define clear_point_delayed() {     \
    uint old_val, col_old_val;      \
    int coarse_idx, flat_idx;       \
    bool was_above_threshold;       \
    short x_tmp = x;                \
    short y_tmp = y;                \
    short z_tmp = z;                \
    if (swap_xz) {                  \
        swap(x_tmp, z_tmp);	        \
    }                               \
    if (swap_xy) {                  \
        swap(x_tmp, y_tmp);	        \
    }                               \
    swap(x_tmp, last_x_clear);      \
    swap(y_tmp, last_y_clear);      \
    swap(z_tmp, last_z_clear);      \
    if (x_tmp != SHRT_MAX) {        \
        if (z_tmp < n_voxels) {        \
            coarse_idx = z_tmp / n_voxels_per_uint32; \
            for (int x_beam = x_tmp - beam_width; x_beam <= x_tmp + beam_width; ++x_beam) { \
                if (x_beam >= 0 && x_beam < dim_1) { \
                    for (int y_beam = y_tmp - beam_width; y_beam <= y_tmp + beam_width; ++y_beam) { \
                        if (y_beam >= 0 && y_beam < dim_0) { \
                            __global uint* volume_data = (__global uint*) (&volume[y_beam * stride0 + x_beam * stride1 + coarse_idx]); \
                            old_val = atomic_and(volume_data, bit_clear_masks[z_tmp]); \
                            was_above_threshold = (old_val & bit_check_masks[z_tmp]) == bit_check_masks[z_tmp]; \
                            flat_idx = y_beam * dim_1 + x_beam; \
                            if (z_tmp <= max_z_clearing_for_known) known[flat_idx] = 1; \
                            if (was_above_threshold) { \
                                __global uint* column_ptr = (__global uint*)(&column_counts[flat_idx - (flat_idx % 4)]); \
                                col_old_val = atomic_sub(column_ptr, (1 << (8 * (flat_idx % 4)))); \
                            } \
                        } \
                    } \
                } \
            } \
        } \
    } \
}

__kernel void raytrace_depth(__global uint* volume,
                             __global uchar* known,
                             __global uchar* column_counts,
                             __global const short* line_defs,
                             __global const int* idx,
                             __global const float* ranges,
                             const short x0,
                             const short y0,
                             const unsigned int dim_0,
                             const unsigned int dim_1,
                             const unsigned int n_voxels,
                             const unsigned int stride0,
                             const unsigned int stride1,
                             const float raytrace_range,
                             const float obstacle_range,
                             const uchar do_clearing,
                             const uchar is_marking,
                             __global short4* packed_markings,
                             const float sq_xy_resolution,
                             const float sq_z_resolution,
                             const short min_z_mark,
                             const short max_z_mark,
                             const short max_z_clearing_for_known,
                             const int beam_width,
                             __constant uint* bit_check_masks,
                             __constant uint* bit_clear_masks,
                             const short n_voxels_per_uint32,
                             const int tot_rays,
                             const unsigned int rows_per_workgroup,
                             const unsigned int max_work_group_size
                             )
{
    int gid = get_global_id(0);
    packed_markings[gid] = (short4)(0, 0, 0, 0);  // invalid mark to begin with
    int id = idx[gid];
    float clearing_range = min(ranges[gid], raytrace_range);
    if (clearing_range <= 0.) return;

    short x_max = dim_1;
    short y_max = dim_0;
    short z_max = n_voxels;

    short start_x = x0 + line_defs[id * 6];
    short start_y = y0 + line_defs[id * 6 + 1];
    short start_z = line_defs[id * 6 + 2];
    if (start_x < 0 || start_x >= x_max || start_y < 0 || start_y >= y_max ||
        start_z < 0 || start_z >= z_max) return;

    float r2_x = sq_xy_resolution;
    float r2_y = sq_xy_resolution;
    float r2_z = sq_z_resolution;

    //length along each axis
    short dx = abs(line_defs[id * 6 + 3]);
    short dy = abs(line_defs[id * 6 + 4]);
    short dz = abs(line_defs[id * 6 + 5]);

    //direction along each axis
    short step_x = clamp(line_defs[id * 6 + 3], (short)-1, (short)1);
    short step_y = clamp(line_defs[id * 6 + 4], (short)-1, (short)1);
    short step_z = clamp(line_defs[id * 6 + 5], (short)-1, (short)1);

    short x_bound = step_x > 0 ? x_max:-1;
    short y_bound = step_y > 0 ? y_max:-1;
    short z_bound = step_z > 0 ? z_max:max(min_z_mark - 1, -1);

    bool swap_xy = (dy > dx);
    if (swap_xy) {
        swap(start_x, start_y);
        swap(dx, dy);
        swap(step_x, step_y);
        swap(x_max, y_max);
        swap_float(r2_x, r2_y);
        swap_float(x_bound, y_bound);
    }
    bool swap_xz = (dz > dx);
    if (swap_xz) {
        swap(start_x, start_z);
        swap(dx, dz);
        swap(step_x, step_z);
        swap(x_max, z_max);
        swap_float(r2_x, r2_z);
        swap_float(x_bound, z_bound);
    }

    // The following does the main trick of this function: calculate
    // which voxel along the line corresponds to the given distance.
    // It's a bit messy because of the integer arithmetic, which
    // only allows us to find 4 possible candidates for the line end;
    // the correct among the 4 candidates has to be found by brute force
    float dx2 = (float) dx * (float) dx;
    float y_sq_term = r2_y * (float) dy * (float) dy / dx2;  // increase square in distance in the y direction for each unit in the x direction
    float z_sq_term = r2_z * (float) dz * (float) dz / dx2;  // increase square in distance in the z direction for each unit in the x direction
    float total_r_scale = sqrt(r2_x + y_sq_term + z_sq_term);  // total increase in distance when x increases by one
    float true_dx_short = floor(clearing_range / total_r_scale);  // the shortest dx where distance would be close to the desired range
    float true_dx_long = true_dx_short + 1.0;  // the desired dx for the given range must be between true_dx_short and true_dx_long
    // Here we compute the 4 candidate voxels for the mark
    float r_short = true_dx_short * total_r_scale;
    float r2_medium_low = r_short * r_short + r2_x * (2 * true_dx_short + 1);
    float r_medium_low = sqrt(r2_medium_low);
    float r_medium_high = sqrt(r2_medium_low + y_sq_term * (2 * true_dx_short + 1));
    float r_long = true_dx_long * total_r_scale;
    float err_short = fabs(r_short - clearing_range);
    float err_medium_low = fabs(r_medium_low - clearing_range);
    float err_medium_high = fabs(r_medium_high - clearing_range);
    float err_long = fabs(r_long - clearing_range);
    // And now the logic to select the closest candidate to the given distance
    bool remove_point = false;
    bool add_point = false;
    short true_dx = (short) (1 + true_dx_short + 0.5);
    if (err_short < err_medium_low) {
        remove_point = true;
    }
    else if (err_medium_low < err_medium_high) {
    }
    else if (err_medium_high < err_long) {
        add_point = true;
    }
    else {
        true_dx += 1;
        remove_point = true;
    }

    // Finally, the raytracing itself
    short x_end = start_x + step_x * true_dx;
    //drift controls when to step in the medium and short directions
    //starting value is centered
    short drift_xy  = (dx / 2) + dy;  // + dy to ensure the fist step is always in the longest direction
    short drift_xz  = (dx / 2) + dz;  // + dz to ensure the fist step is always in the longest direction

    short x = start_x;
    short y = start_y;
    short z = start_z;
    short last_x_clear = SHRT_MAX;
    short last_y_clear = SHRT_MAX;
    short last_z_clear = SHRT_MAX;

    if (!do_clearing) {  // no need to trace, skip directly to the mark
        x = x_end;
        int end_drift_xy = (int) (dx / 2) - (int) (true_dx - 1) * (int) dy;
        int end_drift_xz = (int) (dx / 2) - (int) (true_dx - 1) * (int) dz;
        short n_y_steps = - (end_drift_xy - dx + 1) / dx;
        short n_z_steps = - (end_drift_xz - dx + 1) / dx;
        y += step_y * n_y_steps;
        z += step_z * n_z_steps;
        drift_xy = (short) (end_drift_xy + n_y_steps * dx);
        drift_xz = (short) (end_drift_xz + n_z_steps * dx);
        // Check if the loop would result in out-of-bounds
        // As a special case, if x_end == x_bound, we may have to continue,
        //    even though it is at the very bound.
        if ((x != x_bound && (x < 0 || x >= x_max)) || y < 0 || y >= y_max || z < 0 || z >= z_max) return;
        }
    else {  // trace with clearing
        for (; x != x_end; x += step_x) {
            if (x == x_bound) break;
            clear_point_delayed();

            //update progress in other planes
            drift_xy -= dy;
            drift_xz -= dz;

            //step in medium axis
            if (drift_xy < 0) {
                y += step_y;
                if (y == y_bound) break;
                drift_xy += dx;
                clear_point_delayed();
            }

            //step in short axis
            if (drift_xz < 0) {
                z += step_z;
                if (z == z_bound) break;
                drift_xz += dx;
                clear_point_delayed();
            }
        }
    }

    // Final adjustments depending on the exact ending voxel candidate
    if (x == x_end) {  // otherwise we are out of bounds
        if (remove_point) {
            if (x != start_x) x -= step_x;
        }
        if (x != x_bound) {  // got to the desired endpoint, check termination and marking
            if (add_point) {  // need to add one more x point to get as close as possible to desired length
                if (do_clearing) {
                    clear_point_delayed();
                }
                drift_xy -= dy;
                if (drift_xy < 0) {
                    y += step_y;
                }
            }
            if (y != y_bound) {
                if (swap_xz) swap(x, z);
                if (swap_xy) swap(x, y);
                bool valid_mark = ranges[gid] <= obstacle_range && z >= min_z_mark && z <= max_z_mark;
                if (valid_mark) {
                    packed_markings[gid] = (short4)(1, y, x, z);  // a valid mark
                    if (do_clearing && !is_marking) clear_point_delayed();  // flush the last point to clear only if we are not going to mark it!
                    return;
                }
            }
        }
    }
    if (do_clearing) clear_point_delayed();  // flush the last point to clear, we know we will not mark it
}
