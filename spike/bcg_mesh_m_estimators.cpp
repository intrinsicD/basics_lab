#include "bcg_mesh_m_estimators.h"
#include "mesh/bcg_mesh_face_centers.h"
#include "mesh/bcg_mesh_face_normals.h"
#include "mesh/bcg_mesh_face_areas.h"
#include "mesh/bcg_mesh_vertex_normals.h"
#include "bcg_property_map_eigen.h"
#include "mesh/bcg_mesh_curvature_taubin.h"
#include "geometry/quadric/bcg_quadric.h"
#include "math/vector/bcg_vector_median_filter_directional.h"
#include "tbb/tbb.h"

namespace bcg {

	std::vector<std::string> m_estimator_names() {
		std::vector<std::string> names(static_cast<int>(M_ESTIMATORTYPE::__last__));
		names[static_cast<int>(M_ESTIMATORTYPE::l2_norm)] = "l2_norm";
		names[static_cast<int>(M_ESTIMATORTYPE::truncated_l2_norm)] = "truncated_l2_norm";
		names[static_cast<int>(M_ESTIMATORTYPE::l1_norm)] = "l1_norm";
		names[static_cast<int>(M_ESTIMATORTYPE::truncated_l1_norm)] = "truncated_l1_norm";
		names[static_cast<int>(M_ESTIMATORTYPE::hubers_minimax)] = "hubers_minimax";
		names[static_cast<int>(M_ESTIMATORTYPE::lorentzian_norm)] = "lorentzian_norm";
		names[static_cast<int>(M_ESTIMATORTYPE::gaussian_norm)] = "gaussian_norm";
		names[static_cast<int>(M_ESTIMATORTYPE::tukeys_norm)] = "tukeys_norm";
		return names;
	}

	std::vector<std::string> distances_names() {
		std::vector<std::string> names(static_cast<int>(DISTANCETYPE::__last__));
		names[static_cast<int>(DISTANCETYPE::euclidean)] = "euclidean";
		names[static_cast<int>(DISTANCETYPE::angle)] = "angle";
		names[static_cast<int>(DISTANCETYPE::arccos)] = "arccos";
		return names;
	}

	bcg_scalar_t EuclideanFunctor::operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const
	{
		return (n_i - n_j).norm();
	}

	bcg_scalar_t AngleFunctor::operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const
	{
		bcg_scalar_t x = 1.0 - (n_i - n_j).squaredNorm() / 2.0;
		return acos(x);
	}

	bcg_scalar_t ArccosFunctor::operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const
	{
		return acos(n_i.dot(n_j));
	}

	MEstimatorFunctor::MEstimatorFunctor(bcg_scalar_t _sigma) : sigma(_sigma) {}

	bcg_scalar_t L2NormFunctor::operator()(bcg_scalar_t x) const
	{
		return 2.0* x / x;
	}

	bcg_scalar_t TruncatedL2NormFunctor::operator()(bcg_scalar_t x) const
	{
		bcg_scalar_t weight = 0.0;
		if (std::abs(x) < std::sqrt(sigma)) {
			weight = 2.0 * x / x;
		}
		return weight;
	}

	bcg_scalar_t L1NormFunctor::operator()(bcg_scalar_t x) const
	{
		return 1.0 / x;
	}

	bcg_scalar_t TruncatedL1NormFunctor::operator()(bcg_scalar_t x) const
	{
		bcg_scalar_t weight = 0.0;
		if (std::abs(x) < sigma) {
			weight = 1 / x;
		}
		return weight;
	}

	bcg_scalar_t HubersMinimaxFunctor::operator()(bcg_scalar_t x) const
	{
		bcg_scalar_t weight = 1 / x;
		if (std::abs(x) < sigma) {
			weight = 1 / sigma;
		}
		return weight;
	}

	bcg_scalar_t LorentzianNormFunctor::operator()(bcg_scalar_t x) const
	{
		return 2  / (x * x + 2 * sigma * sigma);
	}

	bcg_scalar_t GaussianNormFunctor::operator()(bcg_scalar_t x) const
	{
		return std::exp(-x * x / sigma * sigma);
	}

	bcg_scalar_t TukeysNormFunctor::operator()(bcg_scalar_t x) const
	{
		bcg_scalar_t weight = 0.0;
		if (std::abs(x) < sigma) {
			weight = 1.0 / 2.0 * (1 - (x / sigma) * (x / sigma)) * (1 - (x / sigma) * (x / sigma));
		}
		return weight;
	}

	const bcg_scalar_t eps = 1e-6;

	void unilateral_filter(halfedge_mesh & mesh, DistanceFunctor const& distance_op, MEstimatorFunctor const& g_op, bcg_scalar_t sigma_g, size_t parallel_grain_size)
	{
		auto f_normals_filtered = mesh.faces.get_or_add<VectorS<3>, 3>("f_normal_filtered", VectorS<3>::Zero());

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.faces.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto f = face_handle(i);

				VectorS<3> n_i = face_normal(mesh, f);
				f_normals_filtered[f] = face_area(mesh, f) * n_i;

				for (const auto& fh : mesh.get_halfedges(f)) {
					auto oh = mesh.get_opposite(fh);
					if (!mesh.is_boundary(oh)) {
						auto ff = mesh.get_face(oh);

						VectorS<3> n_j = face_normal(mesh, ff);
						bcg_scalar_t x = distance_op(n_i, n_j);
						x = std::max(eps, x);
						bcg_scalar_t weight = g_op(x);
						assert(isfinite(weight));
						f_normals_filtered[f] += weight * face_area(mesh, ff) * n_j;
					}
				}
				f_normals_filtered[f].normalize();
			}
		}
		);

		f_normals_filtered.set_dirty();
	}

	void bilateral_filter(halfedge_mesh & mesh, DistanceFunctor const& distance_g_op, MEstimatorFunctor const& g_op, DistanceFunctor const& distance_f_op, MEstimatorFunctor & f_op, bool guided, size_t parallel_grain_size)
	{
		face_normals(mesh, parallel_grain_size);
		auto f_normals = mesh.faces.get<VectorS<3>, 3>("f_normal");
		auto f_normals_filtered = mesh.faces.get_or_add<VectorS<3>, 3>("f_normal_filtered");
		auto e_g = mesh.edges.get_or_add<bcg_scalar_t, 1>("e_normal_filtering_g");
		auto fd = mesh.edges.get_or_add<bcg_scalar_t, 1>("e_face_distance_squared");

		if (guided)
		{
			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.faces.size(), parallel_grain_size),
				[&](const tbb::blocked_range<uint32_t> & range) {
				for (uint32_t i = range.begin(); i != range.end(); ++i) {
					auto f = face_handle(i);
					VectorS<3> n_i = face_normal(mesh, f);
					f_normals_filtered[f] = n_i * face_area(mesh, f);

					for (const auto& fh : mesh.get_halfedges(f)) {
						auto oh = mesh.get_opposite(fh);
						if (!mesh.is_boundary(oh)) {
							auto ff = mesh.get_face(oh);
							VectorS<3> n_j = face_normal(mesh, ff);
							bcg_scalar_t x = acos(1.0 - (n_i - n_j).squaredNorm() / 2.0);
							x = std::max(eps, x);
							if (x < g_op.sigma) {
								f_normals_filtered[f] += n_j * face_area(mesh, ff);
							}
						}
					}
					f_normals_filtered[f].normalize();
				}
			}
			);
		}
		else
		{
			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.faces.size(), parallel_grain_size),
				[&](const tbb::blocked_range<uint32_t> & range) {
				for (uint32_t i = range.begin(); i != range.end(); ++i) {
					auto f = face_handle(i);
					f_normals_filtered[f] = face_normal(mesh, f).normalized();
				}
			}
			);
		}

		tbb::atomic<bcg_scalar_t> fd_avg = 0;
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0u, (uint32_t)mesh.edges.size(), parallel_grain_size),
			[&](const tbb::blocked_range<uint32_t> & range) {
			for (uint32_t i = range.begin(); i != range.end(); ++i) {
				auto e = edge_handle(i);
				fd[e] = 0;
				if (!mesh.is_boundary(e)) {
					auto f0 = mesh.get_face(e, 0);
					auto f1 = mesh.get_face(e, 1);
					bcg_scalar_t face_distance = distance_f_op(face_center(mesh, f0), face_center(mesh, f1));
					face_distance = std::max(eps, face_distance);
					fd_avg = fd_avg + face_distance;
					fd[e] = face_distance;
					bcg_scalar_t x = distance_g_op(f_normals_filtered[f0], f_normals_filtered[f1]);
					x = std::max(eps, x);
					e_g[e] = g_op(x);
				}
			}
		}
		);

		fd_avg = fd_avg / mesh.edges.size();
		bcg_scalar_t two_fd_avg_squared = 2.0 * fd_avg * fd_avg;
		f_op.sigma = fd_avg;
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
						bcg_scalar_t weight = e_g[e] * f_op(fd[e]); //std::exp(-fd[e] / two_fd_avg_squared);
						f_normals_filtered[f] += weight * face_area(mesh, ff) * face_normal(mesh, ff);
					}
				}
				f_normals_filtered[f].normalize();
			}
		}
		);
		e_g.set_dirty();

	}
}