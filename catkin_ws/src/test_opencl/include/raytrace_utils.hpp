#ifndef _RAYTRACE_UTILS_
#define _RAYTRACE_UTILS_

#include <boost/python.hpp>
#include <shining_utils/numpy_boost_python.hpp>
#include <iostream>
#define MAX_VOXELS 255


bool check_is_big_endian(void)
{
/* From http://stackoverflow.com/questions/1001307/detecting-endianness-programmatically-in-a-c-program */
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}


bool big_endian_packing_sort(boost::uint64_t i, boost::uint64_t j) {
    // We want to order by xyz first and w second, but with big endians xyz are least
    // significant, so we rearrange bits to make them most significant
    i = (i << 16) || (i >> 48);  // put xyz first, then w
    j = (j << 16) || (j >> 48);  // put xyz first, then w
    return (i < j);
}


numpy_boost<boost::int16_t, 2> process_markings(std::vector<boost::uint64_t>& packed_markings,
                                                bool do_marking,
                                                numpy_boost<boost::uint32_t, 3>& volume,
                                                numpy_boost<boost::uint8_t, 2>& column_counts,
                                                numpy_boost<boost::uint8_t, 2>& known,
                                                unsigned short n_voxels_per_uint32,
                                                boost::uint32_t *bit_check_masks,
                                                boost::uint32_t *mark_weights) {
    /*
    packed_markings: a vector of uint64 where each number packs four short ints (w, y, x, z),
        in that order in memory, y, x and z being the coordinates of a mark and w being the
        weight of the mark. This is done to quickly sort it and eliminate duplicates, which
        is necessary when marking_threshold > 1
    This unpacks the packed markings into a numpy array, to be returned (mainly for debug
        purposes). If do_marking is true, it also does the actual marking in the volume (and
        updates column_counts and known).
    Notice that, if marking_threshold > 1, marks in the returned numpy array need not reflect
        actual marks (since they can be below threshold); also the numpy array may have
        duplicate marks, but they only count as one for the actual marking.
    */
    // First, unpack marks into numpy array, to be returned, mainly for debug purposes
    size_t dims[] = { packed_markings.size(), 3 };
    numpy_boost<boost::int16_t, 2> numpy_markings(dims);
    for (int i = 0; i < dims[0]; ++i) {
        short* p = (short*) &(packed_markings[i]);
        short y = *++p;  // pre-increment to skip the first value (the weight)
        short x = *++p;
        short z = *++p;
        numpy_markings[i][0] = y;
        numpy_markings[i][1] = x;
        numpy_markings[i][2] = z;
    }

    if (do_marking) {
        // For the actual marking, first remove duplicates
        bool is_big_endian = check_is_big_endian();
        if (is_big_endian) {  // with big endian interpretation of the uint64, standard ordering will not work with our packing scheme
            std::sort(packed_markings.begin(), packed_markings.end(), big_endian_packing_sort);  // little endian order is right for our packing scheme
        }
        else std::sort(packed_markings.begin(), packed_markings.end());  // little endian order is right for our packing scheme
        std::vector<boost::uint64_t>::iterator last = std::unique(packed_markings.begin(), packed_markings.end());
        // And proceed to mark
        // If a mark is repeated with different weights we only use the max weight (the last packed marking in order)
        int len = last - packed_markings.begin();
        for (int i = 0; i < len; ++i) {
            if (!is_big_endian && i < len - 1 && (packed_markings[i] >> 16) == (packed_markings[i + 1] >> 16)) continue;  // same mark with different weight, proceed to bigger weight
            if (is_big_endian && i < len - 1 && (packed_markings[i] << 16) == (packed_markings[i + 1] << 16)) continue;  // same mark with different weight, proceed to bigger weight
            short* p = (short*) &(packed_markings[i]);
            short w = *p++;
            short y = *p++;
            short x = *p++;
            short z = *p;
            if (x < 0 || y < 0 || z < 0 || x >= known.shape()[1] || y >= known.shape()[0] || z >= volume.shape()[2] * n_voxels_per_uint32) {
                std::cerr << "Error in process_markings!" << std::endl;
                std::string error_msg = (boost::format("Bad voxel, idx %i of %i! weight=%i, x=%i, y=%i, z=%i") % i % len % w % x % y % z).str();
                std::cerr << error_msg.c_str() << std::endl;
                throw std::runtime_error(error_msg.c_str());
            }
            known[y][x] = 1;
            int coarse_idx = z / n_voxels_per_uint32;  // which int32 are we looking at in the z column
            boost::uint32_t* voxel = &(volume[y][x][coarse_idx]);
            for (short j = 0; j < w; ++j) {
                if (!((*voxel & bit_check_masks[z]) ==  bit_check_masks[z])) {  // do not mark if already at threshold
                    *voxel += mark_weights[z];  // add a mark
                    if ((*voxel & bit_check_masks[z]) == bit_check_masks[z]) {  // did we reach marking_threshold now?
                        column_counts[y][x] += 1;
                        break;  // no need to mark more
                    }
                }
                else break;  // no need to mark more
            }
        }
    }
    return numpy_markings;
}


numpy_boost<boost::int16_t, 2> get_voxels_implementation(const numpy_boost<boost::uint32_t, 3>& volume,
                                                         int n_voxels,
                                                         const numpy_boost<boost::uint8_t, 2>& column_counts,
                                                         unsigned short n_voxels_per_uint32,
                                                         boost::uint32_t *bit_check_masks) {
    /* Returns a n x 3 matrix with the (x, y, z) indices of every voxel that is marked in the given
       volume. Uses column_counts to speed up the search for the voxels

    volume: a uint32 matrix with the Y x X x 1+(n_voxels-1)/(32/n_bits_in_marking_threshold) 3d voxel volume that will be marked.
        z voxels are bits, each voxel occupying the number of bits needed to represent up to marking_threshold,
        so the z dimension of the matrix is 1+(n_voxels-1)/(32/n_bits_in_marking_threshold).
    n_voxels: number of voxels
    column_counts: the Y x X 2d array with the count of marked voxels in each column.
    */

    if (n_voxels > MAX_VOXELS) throw std::runtime_error((boost::format("Number of voxels %i too big (max is %i)") % n_voxels % MAX_VOXELS).str());
    if (n_voxels > volume.shape()[2] * n_voxels_per_uint32) throw std::runtime_error((boost::format("Number of voxels %i too big (max is %i)") % n_voxels % MAX_VOXELS).str());
    int tot_voxels = 0;
    short x, y;
    unsigned short coarse_z, voxel;
    for (x = 0; x < volume.shape()[1]; x++) {
        for (y = 0; y < volume.shape()[0]; y++) {
            tot_voxels += column_counts[y][x];
        }
    }

    numpy_boost<boost::int16_t, 2> numpy_voxels({ tot_voxels, 3 });
    int voxel_idx = 0;

    Py_BEGIN_ALLOW_THREADS

    for (x = 0; x < volume.shape()[1]; x++) {
        for (y = 0; y < volume.shape()[0]; y++) {
            if (column_counts[y][x] == 0) continue;
            for (coarse_z = 0; coarse_z < volume.shape()[2]; coarse_z++) {
                if (volume[y][x][coarse_z] == 0) continue;
                unsigned short z_origin = coarse_z * n_voxels_per_uint32;
                unsigned short last_voxel = std::min(n_voxels_per_uint32, (unsigned short) (n_voxels - z_origin));
                for (voxel = 0; voxel < last_voxel; voxel++) {
                    if ((volume[y][x][coarse_z] & bit_check_masks[voxel]) == bit_check_masks[voxel]) {
                        numpy_voxels[voxel_idx][0] = x;
                        numpy_voxels[voxel_idx][1] = y;
                        numpy_voxels[voxel_idx][2] = z_origin + voxel;
                        ++voxel_idx;
                    }
                }
            }
        }
    }

    Py_END_ALLOW_THREADS

    if (voxel_idx != tot_voxels) {
        throw std::runtime_error((boost::format("Total sum of column_counts %i does not match number of set voxels %i") % tot_voxels % voxel_idx).str());
    }
    return numpy_voxels;
}


#endif // _RAYTRACE_UTILS_

