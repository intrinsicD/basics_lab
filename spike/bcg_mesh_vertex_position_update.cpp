#include "bcg_mesh_vertex_position_update.h"
#include "mesh/bcg_mesh_face_centers.h"
#include "mesh/bcg_mesh_face_normals.h"
#include "mesh/bcg_mesh_face_areas.h"
#include "mesh/bcg_mesh_vertex_normals.h"
#include "geometry/bcg_property_map_eigen.h"
#include "mesh/bcg_mesh_curvature_taubin.h"
#include "geometry/quadric/bcg_quadric.h"
#include "math/vector/bcg_vector_median_filter_directional.h"
#include "tbb/tbb.h"


namespace bcg {

	std::vector<std::string> vertex_position_update_names() {
		std::vector<std::string> names(static_cast<int>(UpdateType::__last__));
		names[static_cast<int>(UpdateType::vertex_position_update_ohtake)] = "vertex_position_update_ohtake";
		return names;
	}


	void ohtake_vertex_position_update(halfedge_mesh& mesh, size_t parallel_grain_size) {

		auto vertex_positions = mesh.vertices.get<VectorS<3>, 3>("v_position");
		auto normals = mesh.vertices.get<VectorS<3>, 3>("v_normal");
		auto f_normals_filtered = mesh.faces.get_or_add<VectorS<3>, 3>("f_normal_filtered");

		//Ohtake et al vertex updating: x'_i = x_i + 1/(sum(k) A_k) sum(k) A_k * n'_k * (n'_k dot (c_k - x_i))

		auto updated_vertex_positions = mesh.vertices.get_or_add<VectorS<3>, 3>("v_position_new");
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.vertices.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto v = vertex_handle(i);

				VectorS<3> delta = VectorS<3>::Zero();
				bcg_scalar_t sum_weights = 0;
				for (const auto& fv : mesh.get_faces(v)) {
					VectorS<3> dif_center_positions = face_center(mesh, fv) - vertex_positions[v];
					bcg_scalar_t weight = face_area(mesh, fv);
					sum_weights += weight;
					delta += weight * f_normals_filtered[fv] * dif_center_positions.dot(f_normals_filtered[fv]);
				}
				delta /= sum_weights;
				assert(isfinite(delta.x()) && isfinite(delta.y()) && isfinite(delta.z()));
				updated_vertex_positions[v] = vertex_positions[v] + delta;
			}
		}
		);
		Map(vertex_positions) = MapConst(updated_vertex_positions);

		face_normals(mesh, parallel_grain_size);
		vertex_normals(mesh, vertex_normal_area_angle, parallel_grain_size);
		vertex_positions.set_dirty();
		mesh.faces.remove(f_normals_filtered);

	}

}



