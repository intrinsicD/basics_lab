//
// Created by alex on 17.02.21.
//

#ifndef BCG_GRAPHICS_BCG_VECTOR_MEDIAN_FILTER_DIRECTIONAL_H
#define BCG_GRAPHICS_BCG_VECTOR_MEDIAN_FILTER_DIRECTIONAL_H

#include "bcg_vector.h"
#include "math/matrix/bcg_matrix_map_eigen.h"
#include "math/matrix/bcg_matrix_pairwise_distances.h"

namespace bcg{

template<typename T, int D>
void vector_median_filter_directional(const std::vector<Vector<T, D>> &V){
    return vector_median_filter_directional(MapConst(V));
}

template<typename Derived>
Vector<typename Derived::Scalar, -1> vector_median_filter_directional(const Eigen::EigenBase<Derived> &V){
    Derived PA = (1.0 - pairwise_distance_squared(V.rowwise().normalized(), V.rowwise().normalized()).array() / 2.0).acos();
    Vector<typename Derived::Scalar, -1> sum = PA.rowwise().sum();
    auto median_idx = sum.minCoeff();
    return V.row(median_idx);
}

}

#endif //BCG_GRAPHICS_BCG_VECTOR_MEDIAN_FILTER_DIRECTIONAL_H
