#pragma once

#ifndef BCG_GRAPHICS_BCG_POINT_SET_BILATERAL_ZHENG_H
#define BCG_GRAPHICS_BCG_POINT_SET_BILATERAL_ZHENG_H

#include "point_cloud/bcg_point_cloud.h"
#include "kdtree/bcg_kdtree.h"

namespace bcg {

	void point_set_bilateral_zheng(vertex_container* vertices, kdtree_property<bcg_scalar_t>& index, int num_closest,
		bcg_scalar_t sigma_g, bcg_scalar_t sigma_f, size_t parallel_grain_size = 1024);

}

#endif //BCG_GRAPHICS_BCG_POINT_SET_BILATERAL_ZHENG_H