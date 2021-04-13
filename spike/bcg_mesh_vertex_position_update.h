#pragma once
#ifndef BCG_GRAPHICS_BCG_MESH_NORMAL_POSITION_UPDATE_H
#define BCG_GRAPHICS_BCG_MESH_NORMAL_POSITION_UPDATE_H

#include "geometry/mesh/bcg_mesh.h"

namespace bcg {
	enum class UpdateType {
		vertex_position_update_ohtake,
		__last__
	};

	std::vector<std::string>  vertex_position_update_names();

	void ohtake_vertex_position_update(halfedge_mesh &mesh, size_t parallel_grain_size = 1024);
}

#endif //BCG_GRAPHICS_BCG_MESH_NORMAL_POSITION_UPDATE_H