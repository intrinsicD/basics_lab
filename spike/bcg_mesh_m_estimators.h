#pragma once

#ifndef BCG_GRAPHICS_BCG_MESH_M_ESTIMATORS_H
#define BCG_GRAPHICS_BCG_MESH_M_ESTIMATORS_H

#include "mesh/bcg_mesh.h"

namespace bcg {

	enum class M_ESTIMATORTYPE {
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

	enum class DISTANCETYPE {
		euclidean,
		angle,
		arccos,
		__last__
	};

	std::vector<std::string> m_estimator_names();
	std::vector<std::string> distances_names();

	struct DistanceFunctor
	{
		virtual bcg_scalar_t operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const = 0;
	};

	struct EuclideanFunctor : public DistanceFunctor
	{
		bcg_scalar_t operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const ;
	};

	struct AngleFunctor : public DistanceFunctor
	{
		bcg_scalar_t operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const ;
	};

	struct ArccosFunctor : public DistanceFunctor
	{
		bcg_scalar_t operator()(VectorS<3> const& n_i, VectorS<3> const& n_j) const;
	};

	struct MEstimatorFunctor
	{
		bcg_scalar_t sigma;
		MEstimatorFunctor(bcg_scalar_t _sigma);

		virtual bcg_scalar_t operator()(bcg_scalar_t x) const = 0;
	};

	struct L2NormFunctor : public MEstimatorFunctor
	{
		MEstimatorFunctor::MEstimatorFunctor;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct TruncatedL2NormFunctor : public MEstimatorFunctor
	{
		MEstimatorFunctor::MEstimatorFunctor;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct L1NormFunctor : public MEstimatorFunctor
	{
		MEstimatorFunctor::MEstimatorFunctor;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct TruncatedL1NormFunctor : public MEstimatorFunctor
	{
		MEstimatorFunctor::MEstimatorFunctor;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct HubersMinimaxFunctor : public MEstimatorFunctor
	{
		MEstimatorFunctor::MEstimatorFunctor;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct LorentzianNormFunctor : public MEstimatorFunctor
	{
		MEstimatorFunctor::MEstimatorFunctor;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct GaussianNormFunctor : public MEstimatorFunctor
	{
		MEstimatorFunctor::MEstimatorFunctor;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	struct TukeysNormFunctor : public MEstimatorFunctor
	{
		MEstimatorFunctor::MEstimatorFunctor;
		bcg_scalar_t operator()(bcg_scalar_t x) const;
	};

	void unilateral_filter(halfedge_mesh& mesh, DistanceFunctor const& distance_op, MEstimatorFunctor const& g_op, bcg_scalar_t sigma_g, size_t parallel_grain_size);

	void bilateral_filter(halfedge_mesh& mesh, DistanceFunctor const& distance_g_op, MEstimatorFunctor const& g_op, DistanceFunctor const& distance_f_op, MEstimatorFunctor& f_op, bool guided, size_t parallel_grain_size);

}

#endif //BCG_GRAPHICS_BCG_MESH_M_ESTIMATORS_H
