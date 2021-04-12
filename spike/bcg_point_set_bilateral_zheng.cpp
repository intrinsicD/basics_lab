#include "bcg_point_set_bilateral_zheng.h"
#include "point_cloud/bcg_point_cloud_vertex_pca.h"
#include "math/matrix/bcg_matrix_map_eigen.h"
#include "math/vector/bcg_vector_map_eigen.h"
#include "bcg_property_map_eigen.h"
#include "tbb/tbb.h"

namespace bcg {

	void point_set_bilateral_zheng(vertex_container* vertices, kdtree_property<bcg_scalar_t>& index, int num_closest,
		bcg_scalar_t sigma_g, bcg_scalar_t sigma_f, size_t parallel_grain_size) {


		auto positions = vertices->get<VectorS<3>, 3>("v_position");
		//auto normals = vertices->get_or_add<VectorS<3>, 3>("v_normal");
		auto normals = vertices->get<VectorS<3>, 3>("v_normal");
		auto v_normals_filtered = vertices->get_or_add<VectorS<3>, 3>("v_normal_filtered", VectorS<3>::Zero());

		if (!normals) {
			normals = vertices->get_or_add<VectorS<3>, 3>("v_normal");

			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0u, (uint32_t)vertices->size(), parallel_grain_size),
				[&](const tbb::blocked_range<uint32_t> & range) {
				for (uint32_t i = range.begin(); i != range.end(); ++i) {
					auto v = vertex_handle(i);
					auto result = index.query_knn(positions[v], num_closest);
					std::vector<VectorS<3>> V;
					for (size_t i = 0; i < result.indices.size(); ++i) {
						V.push_back(positions[result.indices[i]]);
					}
					auto pca = point_cloud_vertex_pca_least_squares_svd(MapConst(V), positions[v], false);
					normals[v] = pca.directions.col(2);
				}
			}
			);
		}

		bcg_scalar_t sigma_g_squared = sigma_g * sigma_g; // sigma_g between 0.1 and 0.5
		bcg_scalar_t sigma_f_squared = sigma_f * sigma_f; // sigma_f between 0.01 and 0.5
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)vertices->size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto v = vertex_handle(i);
				VectorS<3> n_i = normals[v];
				VectorS<3> p_i = positions[v];

				auto result = index.query_knn(positions[v], num_closest);
				for (const auto& idx : result.indices) {
					if (idx == v.idx) continue;
					VectorS<3> n_j = normals[idx];
					VectorS<3> p_j = positions[idx];
					bcg_scalar_t d = (p_i - p_j).norm();
					bcg_scalar_t x = (n_i - n_j).norm();
					bcg_scalar_t e_fd = std::exp(-d * d / 2 * sigma_f_squared);
					bcg_scalar_t weight = std::exp(-x * x / 2* sigma_g_squared) * e_fd;
					v_normals_filtered[v] += weight * n_j;

				}

				v_normals_filtered[v].normalize();
			}
		}
		);

		normals.set_dirty();
		v_normals_filtered.set_dirty();
		//positions.set_dirty(); // set property dirty so that it is uploaded to the gpu and displayed.
		//point_set_bilateral_digne_francis(vertices, index, num_closest, sigma_g, parallel_grain_size);//vertex position update 
	}

}