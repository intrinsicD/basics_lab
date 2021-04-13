#pragma once

#ifndef BCG_GRAPHICS_BCG_POINT_SET_BILATERAL_DIGNE_FRANCIS_H
#define BCG_GRAPHICS_BCG_POINT_SET_BILATERAL_DIGNE_FRANCIS_H

#include "point_cloud/bcg_point_cloud.h"
#include "kdtree/bcg_kdtree.h"

namespace bcg {

	enum class Point_UpdateType {
		point_position_update_digne_francis,
		point_position_update_zheng,
		__last__
	};

	std::vector<std::string>  point_position_update_names();

	void point_set_bilateral_digne_francis(vertex_container *vertices,
		kdtree_property<bcg_scalar_t> &index, bcg_scalar_t query_radius, bcg_scalar_t radius, bcg_scalar_t radius1, size_t parallel_grain_size = 1024);

	void point_set_bilateral_zheng_update(vertex_container* vertices,
		kdtree_property<bcg_scalar_t>& index, int num_closest, bcg_scalar_t sigma_g, bcg_scalar_t sigma_f, size_t parallel_grain_size = 1024);

}

#endif //BCG_GRAPHICS_BCG_POINT_SET_BILATERAL_DIGNE_FRANCIS_H