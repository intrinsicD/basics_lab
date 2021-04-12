#include "bcg_point_set_bilateral_digne_francis.h"
#include "point_cloud/bcg_point_cloud_vertex_pca.h"
#include "math/matrix/bcg_matrix_map_eigen.h"
#include "math/vector/bcg_vector_map_eigen.h"
#include "bcg_property_map_eigen.h"
#include "tbb/tbb.h"

namespace bcg {

	std::vector<std::string> point_position_update_names() {
		std::vector<std::string> names(static_cast<int>(Point_UpdateType::__last__));
		names[static_cast<int>(Point_UpdateType::point_position_update_digne_francis)] = "point_position_update_digne_francis";
		return names;
	}

	void point_set_bilateral_digne_francis(vertex_container* vertices,
		kdtree_property<bcg_scalar_t>& index, bcg_scalar_t query_radius, bcg_scalar_t radius, size_t parallel_grain_size) {

		auto v_normals_filtered = vertices->get_or_add<VectorS<3>, 3>("v_normal_filtered", VectorS<3>::Zero());
		auto positions = vertices->get<VectorS<3>, 3>("v_position");
		auto normals = vertices->get_or_add<VectorS<3>, 3>("v_normal");
		//auto e_fd = vertices->get_or_add<bcg_scalar_t, 1>("normal_filtering_g");
		//auto fd = vertices->get_or_add<bcg_scalar_t, 1>("normal_filtering_fd");

		auto updated_point_position = vertices->get_or_add<VectorS<3>, 3>("v_updated_point_position");

		bcg_scalar_t sigma_d = radius * 1 / 3;
		bcg_scalar_t sigma_g = radius * 1 / 3;
		bcg_scalar_t sigma_d_squared = sigma_d * sigma_d;
		bcg_scalar_t sigma_g_squared = sigma_g * sigma_g;



		/*tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)vertices->size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto v = vertex_handle(i);
				VectorS<3> p_i = positions[v];
				fd[v] = 0;
				e_fd[v] = 0;

				auto result = index.query_radius(positions[v], query_radius);
				for (const auto& idx : result.indices) {
					if (idx == v.idx) continue;
					VectorS<3> p_j = positions[idx];
					bcg_scalar_t d = (p_j - p_i).norm();
					fd[v] = d * d;
					e_fd[v] = std::exp(-fd[v]/ 2 * sigma_g_squared);


				}

			}
		}
		);*/

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)vertices->size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				bcg_scalar_t delta_p = 0;
				bcg_scalar_t sum_weights = 0;
				auto v = vertex_handle(i);
				VectorS<3> p_i = positions[v];

				auto result = index.query_radius(positions[v], query_radius);
				for (const auto& idx : result.indices) {
					if (idx == v.idx) continue;
					VectorS<3> p_j = positions[idx];
					bcg_scalar_t d = (p_i - p_j).norm();
					bcg_scalar_t d_sqaured = d * d;
					bcg_scalar_t e_fd = std::exp(-d_sqaured / 2 * sigma_g_squared);
					bcg_scalar_t x = v_normals_filtered[v].dot(p_j - p_i);
					bcg_scalar_t weight = std::exp(-x * x / 2 * sigma_g_squared) * e_fd;
					delta_p = delta_p + weight * x;
					sum_weights += weight;
				}
				delta_p /= sum_weights;
				updated_point_position[v] = positions[v] + delta_p * v_normals_filtered[v];

			}
		}
		);

		Map(positions) = MapConst(updated_point_position);
		index.build(positions);
		positions.set_dirty();
		//normals.set_dirty();

	}
}