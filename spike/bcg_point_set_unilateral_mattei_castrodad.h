#pragma once

#ifndef BCG_GRAPHICS_BCG_POINT_SET_UNILATERAL_MATTEI_CASTRODAD_H
#define BCG_GRAPHICS_BCG_POINT_SET_UNILATERAL_MATTEI_CASTRODAD_H

#include "point_cloud/bcg_point_cloud.h"
#include "kdtree/bcg_kdtree.h"

namespace bcg {

	void point_set_unilateral_mattei_castrodad(vertex_container* vertices, kdtree_property<bcg_scalar_t>& index, int num_closest,
		bcg_scalar_t sigma_g, size_t parallel_grain_size = 1024);

}

#endif //BCG_GRAPHICS_BCG_POINT_SET_UNILATERAL_MATTEI_CASTRODAD_H