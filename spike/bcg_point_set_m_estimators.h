#pragma once

#ifndef BCG_GRAPHICS_BCG_POINT_SET_M_ESTIMATORS_H
#define BCG_GRAPHICS_BCG_POINT_SET_M_ESTIMATORS_H

#include "point_cloud/bcg_point_cloud.h"
#include "kdtree/bcg_kdtree.h"

namespace bcg {

	enum class M_ESTIMATORTYPE_POINT {
		l2_norm,
		truncated_l2_norm,
		l1_norm,
		truncated_l1_norm,
		hubers_minimax,
		lorentzian_norm,
		gaussian_norm,
		tukeys_norm,
		__last__
	};

	enum class DISTANCETYPE_POINT {
		euclidean,
		angle,
		arccos,
		__last__
	};

	std::vector<std::string> m_estimator_names_point();
	std::vector<std::string> distances_names_point();

	struct DistanceFunctor_Point
	{
		virtual bcg_scalar_t operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const = 0;
	};

	struct EuclideanFunctor_Point : public DistanceFunctor_Point
	{
		bcg_scalar_t operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const;
	};

	struct AngleFunctor_Point : public DistanceFunctor_Point
	{
		bcg_scalar_t operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const;
	};

	struct ArccosFunctor_Point : public DistanceFunctor_Point
	{
		bcg_scalar_t operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const;
	};

	struct MEstimatorFunctor_Point
	{
		bcg_scalar_t sigma;
		MEstimatorFunctor_Point(bcg_scalar_t _sigma);

		virtual bcg_scalar_t operator()(bcg_scalar_t x) const = 0;
	};

	struct L2NormFunctor_Point : public MEstimatorFunctor_Point
	{
		MEstimatorFunctor_Point::MEstimatorFunctor_Point;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct TruncatedL2NormFunctor_Point : public MEstimatorFunctor_Point
	{
		MEstimatorFunctor_Point::MEstimatorFunctor_Point;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct L1NormFunctor_Point : public MEstimatorFunctor_Point
	{
		MEstimatorFunctor_Point::MEstimatorFunctor_Point;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct TruncatedL1NormFunctor_Point : public MEstimatorFunctor_Point
	{
		MEstimatorFunctor_Point::MEstimatorFunctor_Point;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct HubersMinimaxFunctor_Point : public MEstimatorFunctor_Point
	{
		MEstimatorFunctor_Point::MEstimatorFunctor_Point;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct LorentzianNormFunctor_Point : public MEstimatorFunctor_Point
	{
		MEstimatorFunctor_Point::MEstimatorFunctor_Point;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct GaussianNormFunctor_Point : public MEstimatorFunctor_Point
	{
		MEstimatorFunctor_Point::MEstimatorFunctor_Point;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct TukeysNormFunctor_Point : public MEstimatorFunctor_Point
	{
		MEstimatorFunctor_Point::MEstimatorFunctor_Point;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	void unilateral_filter(vertex_container* vertices, kdtree_property<bcg_scalar_t>& index, int num_closest, DistanceFunctor_Point const& distance_op, MEstimatorFunctor_Point const& g_op, bcg_scalar_t sigma_g, size_t parallel_grain_size);

	void bilateral_filter(vertex_container* vertices, kdtree_property<bcg_scalar_t>& index, int num_closest, DistanceFunctor_Point const& distance_g_op, MEstimatorFunctor_Point const& g_op, bcg_scalar_t sigma_g, DistanceFunctor_Point const& distance_f_op, MEstimatorFunctor_Point& f_op, bcg_scalar_t sigma_f, size_t parallel_grain_size);

}

#endif //BCG_GRAPHICS_BCG_POINT_SET_M_ESTIMATORS_H

