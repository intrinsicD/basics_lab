#include "bcg_mesh_unilateral_tasdizen.h"
#include "mesh/bcg_mesh_face_centers.h"
#include "mesh/bcg_mesh_face_normals.h"
#include "mesh/bcg_mesh_face_areas.h"
#include "mesh/bcg_mesh_vertex_normals.h"
#include "geometry/bcg_property_map_eigen.h"
#include "mesh/bcg_mesh_curvature_taubin.h"
#include "geometry/quadric/bcg_quadric.h"
#include "math/vector/bcg_vector_median_filter_directional.h"
#include "bcg_mesh_vertex_position_update.h"
#include "tbb/tbb.h"

namespace bcg {

	void mesh_unilateral_tasdizen(halfedge_mesh& mesh, bcg_scalar_t sigma_g, size_t parallel_grain_size) {

		auto f_normals_filtered = mesh.faces.get_or_add<VectorS<3>, 3>("f_normal_filtered", VectorS<3>::Zero());

		mesh_curvature_taubin(mesh, 1, true, parallel_grain_size);

		auto gauss_curvature = mesh.vertices.get_or_add<bcg_scalar_t, 1>("v_mesh_curv_gauss");

		bcg_scalar_t sigma_g_squared = sigma_g * sigma_g;
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.faces.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto f = face_handle(i);
			
				f_normals_filtered[f] = face_area(mesh, f) * face_normal(mesh, f);
				for (const auto& fh : mesh.get_halfedges(f)) {
					auto oh = mesh.get_opposite(fh);
					if (!mesh.is_boundary(oh)) {
						auto ff = mesh.get_face(oh);
						bcg_scalar_t x = (gauss_curvature[mesh.get_from_vertex(fh)] +
							gauss_curvature[mesh.get_to_vertex(fh)]) / 2.0;
						bcg_scalar_t weight = std::exp(-x * x / sigma_g_squared);
						f_normals_filtered[f] += weight * face_area(mesh, ff) * face_normal(mesh, ff);
					}
				}
				f_normals_filtered[f].normalize();
			}
		}
		);

		ohtake_vertex_position_update(mesh, parallel_grain_size);
	}

}