#include "bcg_mesh_unilateral_shen.h"
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

	void mesh_unilateral_shen(halfedge_mesh& mesh, bcg_scalar_t sigma_g, size_t parallel_grain_size) {

		auto f_normals_filtered = mesh.faces.get_or_add<VectorS<3>, 3>("f_normal_filtered", VectorS<3>::Zero());

		bcg_scalar_t sigma_g_squared = sigma_g * sigma_g;
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.faces.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto f = face_handle(i);
			
				f_normals_filtered[f].setZero();
				std::vector<VectorS<3>> normals;
				normals.push_back(face_normal(mesh, f));
				for (const auto& fh : mesh.get_halfedges(f)) {
					auto oh = mesh.get_opposite(fh);
					if (!mesh.is_boundary(oh)) {
						auto ff = mesh.get_face(oh);
						normals.push_back(face_normal(mesh, ff));
					}
				}
				VectorS<3> median = vector_median_filter_directional(normals); // argmin sum over angle between neigboring normals
				f_normals_filtered[f] = median * face_area(mesh, f);
				
				for (const auto& fh : mesh.get_halfedges(f)) {
					auto oh = mesh.get_opposite(fh);
					if (!mesh.is_boundary(oh)) {
						auto ff = mesh.get_face(oh);
						VectorS<3> n_j = face_normal(mesh, ff);
						bcg_scalar_t x = (median - n_j).norm();
						bcg_scalar_t weight = std::exp(-x * x / sigma_g_squared);
						f_normals_filtered[f] += weight * face_area(mesh, ff) * n_j;
					}
				}
				f_normals_filtered[f].normalize();
			}
		}
		);

		ohtake_vertex_position_update(mesh, parallel_grain_size);
	}

}