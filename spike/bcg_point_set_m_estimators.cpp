

#include "bcg_point_set_m_estimators.h"
#include "point_cloud/bcg_point_cloud_vertex_pca.h"
#include "math/matrix/bcg_matrix_map_eigen.h"
#include "math/vector/bcg_vector_map_eigen.h"
#include "bcg_property_map_eigen.h"
#include "tbb/tbb.h"

namespace bcg {

	std::vector<std::string> m_estimator_names_point() {
		std::vector<std::string> names(static_cast<int>(M_ESTIMATORTYPE_POINT::__last__));
		names[static_cast<int>(M_ESTIMATORTYPE_POINT::l2_norm)] = "l2_norm";
		names[static_cast<int>(M_ESTIMATORTYPE_POINT::truncated_l2_norm)] = "truncated_l2_norm";
		names[static_cast<int>(M_ESTIMATORTYPE_POINT::l1_norm)] = "l1_norm";
		names[static_cast<int>(M_ESTIMATORTYPE_POINT::truncated_l1_norm)] = "truncated_l1_norm";
		names[static_cast<int>(M_ESTIMATORTYPE_POINT::hubers_minimax)] = "hubers_minimax";
		names[static_cast<int>(M_ESTIMATORTYPE_POINT::lorentzian_norm)] = "lorentzian_norm";
		names[static_cast<int>(M_ESTIMATORTYPE_POINT::gaussian_norm)] = "gaussian_norm";
		names[static_cast<int>(M_ESTIMATORTYPE_POINT::tukeys_norm)] = "tukeys_norm";
		return names;
	}

	std::vector<std::string> distances_names_point() {
		std::vector<std::string> names(static_cast<int>(DISTANCETYPE_POINT::__last__));
		names[static_cast<int>(DISTANCETYPE_POINT::euclidean)] = "euclidean";
		names[static_cast<int>(DISTANCETYPE_POINT::angle)] = "angle";
		names[static_cast<int>(DISTANCETYPE_POINT::arccos)] = "arccos";
		return names;
	}

	bcg_scalar_t EuclideanFunctor_Point::operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const
	{
		return (n_i - n_j).norm();
	}

	bcg_scalar_t AngleFunctor_Point::operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const
	{
		return acos(1.0 - (n_i - n_j).squaredNorm() / 2.0);
	}

	bcg_scalar_t ArccosFunctor_Point::operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const
	{
		return acos(n_i.dot(n_j));
	}




	MEstimatorFunctor_Point::MEstimatorFunctor_Point(bcg_scalar_t _sigma) : sigma(_sigma) {}

	bcg_scalar_t L2NormFunctor_Point::operator()(bcg_scalar_t x) const
	{
		return 2.0* x / x;
	}

	bcg_scalar_t TruncatedL2NormFunctor_Point::operator()(bcg_scalar_t x) const
	{
		bcg_scalar_t weight = 0.0;
		if (std::abs(x) < sigma) {
			weight = 2.0 * x / x;
		}
		return weight;
	}

	bcg_scalar_t L1NormFunctor_Point::operator()(bcg_scalar_t x) const
	{
		return 1.0 / x;
	}

	bcg_scalar_t TruncatedL1NormFunctor_Point::operator()(bcg_scalar_t x) const
	{
		bcg_scalar_t weight = 0.0;
		if (std::abs(x) < sigma) {
			weight = 1 / x;
		}
		return weight;
	}

	bcg_scalar_t HubersMinimaxFunctor_Point::operator()(bcg_scalar_t x) const
	{
		bcg_scalar_t weight = 1 / x;
		if (std::abs(x) < sigma) {
			weight = 1 / sigma;
		}
		return weight;
	}

	bcg_scalar_t LorentzianNormFunctor_Point::operator()(bcg_scalar_t x) const
	{
		return 2 * x / (x * x + 2 * sigma * sigma);
	}

	bcg_scalar_t GaussianNormFunctor_Point::operator()(bcg_scalar_t x) const
	{
		return std::exp(-x * x / sigma * sigma);
	}

	bcg_scalar_t TukeysNormFunctor_Point::operator()(bcg_scalar_t x) const
	{
		bcg_scalar_t weight = 0;
		if (std::abs(x) < sigma) {
			weight = 1.0 / 2.0 * (1 - (x / sigma) * (x / sigma)) * (1 - (x / sigma) * (x / sigma));
		}
		return weight;
	}

	void unilateral_filter(vertex_container* vertices, kdtree_property<bcg_scalar_t>& index, int num_closest, DistanceFunctor_Point const& distance_op, MEstimatorFunctor_Point const& g_op, bcg_scalar_t sigma_g, size_t parallel_grain_size)
	{
		auto positions = vertices->get<VectorS<3>, 3>("v_position");
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

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)vertices->size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto v = vertex_handle(i);

				VectorS<3> n_i = normals[v];
				VectorS<3> p_i = positions[v];
				v_normals_filtered[v] = n_i;

				auto result = index.query_knn(positions[v], num_closest);
				for (const auto& idx : result.indices) {
					if (idx == v.idx) continue;
						VectorS<3> n_j = normals[idx];

						bcg_scalar_t x = distance_op(n_i, n_j);
						bcg_scalar_t weight = g_op(x);
						v_normals_filtered[v] += weight * n_j;
					}

				v_normals_filtered[v].normalize();

				}
			
			}
		
		);

		v_normals_filtered.set_dirty();
	}

	void bilateral_filter(vertex_container* vertices, kdtree_property<bcg_scalar_t>& index, int num_closest, DistanceFunctor_Point const& distance_g_op, MEstimatorFunctor_Point const& g_op, bcg_scalar_t sigma_g, DistanceFunctor_Point const& distance_f_op, MEstimatorFunctor_Point& f_op, bcg_scalar_t sigma_f, size_t parallel_grain_size)
	{
		auto positions = vertices->get<VectorS<3>, 3>("v_position");
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

					bcg_scalar_t d = distance_f_op(p_i, p_j);
					bcg_scalar_t x = distance_g_op(n_i, n_j);
					bcg_scalar_t e_fd = g_op(x);
					bcg_scalar_t weight = f_op(d) * e_fd;
					v_normals_filtered[v] += weight * n_j;

				}
				v_normals_filtered[v].normalize();

				}
			}
			);
		
		normals.set_dirty();
		v_normals_filtered.set_dirty();
	}
}
