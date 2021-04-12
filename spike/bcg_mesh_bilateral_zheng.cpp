#include "bcg_mesh_bilateral_zheng.h"
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

	void mesh_bilateral_zheng(halfedge_mesh& mesh, bcg_scalar_t sigma_g, size_t parallel_grain_size) {

		face_normals(mesh, parallel_grain_size);
		auto f_normals = mesh.faces.get<VectorS<3>, 3>("f_normal");
		auto f_normals_filtered = mesh.faces.get_or_add<VectorS<3>, 3>("f_normal_filtered");
		auto e_g = mesh.edges.get_or_add<bcg_scalar_t, 1>("e_normal_filtering_g");
		auto fd = mesh.edges.get_or_add<bcg_scalar_t, 1>("e_face_distance_squared");

		tbb::atomic<bcg_scalar_t> sigma_d = 0;
		bcg_scalar_t two_sigma_g_squared = 2 * sigma_g * sigma_g;
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.edges.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto e = edge_handle(i);
				fd[e] = 0;
				if (!mesh.is_boundary(e)) {
					auto f0 = mesh.get_face(e, 0);
					auto f1 = mesh.get_face(e, 1);
					bcg_scalar_t d = (face_center(mesh, f0) - face_center(mesh, f1)).norm(); // centroid euclidian distance
					sigma_d = sigma_d + d; // average distance between neighboring faces and central face
					fd[e] = d * d; 
					bcg_scalar_t x = (face_normal(mesh, f0) - face_normal(mesh, f1)).norm();
					e_g[e] = std::exp(-x * x / two_sigma_g_squared);
				}
			}
		}
		);

		sigma_d = sigma_d / mesh.edges.size(); // average distance between neighboring faces and central face
		bcg_scalar_t two_sigma_d_squared = 2.0 * sigma_d * sigma_d;
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.faces.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto f = face_handle(i);
				f_normals_filtered[f] = face_normal(mesh, f) * face_area(mesh, f);

				for (const auto& fh : mesh.get_halfedges(f)) {
					auto oh = mesh.get_opposite(fh);
					if (!mesh.is_boundary(oh)) {
						auto ff = mesh.get_face(oh);
						auto e = mesh.get_edge(fh);
						bcg_scalar_t weight = e_g[e] * std::exp(-fd[e] / two_sigma_d_squared); // g and f weight
						f_normals_filtered[f] += weight * face_area(mesh, ff) * face_normal(mesh, ff);
					}
				}
				f_normals_filtered[f].normalize();
			}
		}
		);
		e_g.set_dirty();
		ohtake_vertex_position_update(mesh, parallel_grain_size);
	}

}