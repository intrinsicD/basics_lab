#include "bcg_mesh_unilateral_centin.h"
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

	void mesh_unilateral_centin(halfedge_mesh& mesh, bcg_scalar_t sigma_g, size_t parallel_grain_size) {

		auto f_normals_filtered = mesh.faces.get_or_add<VectorS<3>, 3>("f_normal_filtered", VectorS<3>::Zero());

		mesh_curvature_taubin(mesh, 1, true, parallel_grain_size);

		auto max_curvature = mesh.vertices.get_or_add<bcg_scalar_t, 1>("v_mesh_curv_max");

		tbb::atomic<bcg_scalar_t> l_avg = 0;

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.edges.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto e = edge_handle(i);
				l_avg = l_avg + mesh.get_length(e) / mesh.edges.size();
			}
		}
		);
		bcg_scalar_t sigma_g_square = sigma_g * sigma_g;
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.faces.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto f = face_handle(i);
				f_normals_filtered[f] = face_area(mesh, f) * face_normal(mesh, f);

				bcg_scalar_t k = 0;
				int count = 0;
				for (const auto& fh : mesh.get_halfedges(f)) {
					auto v = mesh.get_to_vertex(fh);
					k += max_curvature[v];
					++count;
				}
				k /= count;

				for (const auto& fh : mesh.get_halfedges(f)) {
					auto oh = mesh.get_opposite(fh);
					if (!mesh.is_boundary(oh)) {
						auto ff = mesh.get_face(oh);
						
						bcg_scalar_t x = k * l_avg;
						bcg_scalar_t weight = 1.0;
						if (std::abs(x) >= sigma_g) {
							bcg_scalar_t diff = x - sigma_g;
							weight =  sigma_g_square / (diff * diff + sigma_g_square);
						}

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