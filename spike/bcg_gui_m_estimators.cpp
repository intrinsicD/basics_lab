/*
#include "bcg_gui_m_estimators.h"
#include "libs/bcg_opengl/bcg_viewer_state.h"
#include "bcg_mesh_normal_filtering.h"
#include "libs/bcg_opengl/guis/bcg_gui_point_cloud_vertex_noise.h"
#include "bcg_mesh_m_estimators.h"

namespace bcg {

	std::vector<std::string> filter() {
		std::vector<std::string> names(static_cast<int>(FilterType::__last__));
		names[static_cast<int>(FilterType::unilateral_filter)] = "unilateral_normal_filter";
		names[static_cast<int>(FilterType::bilateral_filter)] = "bilateral_normal_filter";
		return names;
	}
	std::vector<std::string> m_estimators() {
		std::vector<std::string> names(static_cast<int>(EstimatorType::__last__));
		names[static_cast<int>(EstimatorType::l2_norm)] = "l2_norm";
		names[static_cast<int>(EstimatorType::truncated_l2_norm)] = "truncated_l2_norm";
		names[static_cast<int>(EstimatorType::l1_norm)] = "l1_norm";
		names[static_cast<int>(EstimatorType::truncated_l1_norm)] = "truncated_l2_norm";
		names[static_cast<int>(EstimatorType::hubers_minimax)] = "hubers_minimax";
		names[static_cast<int>(EstimatorType::lorentzian_norm)] = "lorentzian_norm";
		names[static_cast<int>(EstimatorType::gaussian_norm)] = "gaussian_norm";
		names[static_cast<int>(EstimatorType::tukeys_norm)] = "tukeys_norm";
		return names;
	}

	std::vector<std::string> distances() {
		std::vector<std::string> names(static_cast<int>(InputType::__last__));
		names[static_cast<int>(InputType::euclidean)] = "euclidean";
		names[static_cast<int>(InputType::angle)] = "angle";
		names[static_cast<int>(InputType::arccos)] = "arccos";
		return names;
	}

	void gui_m_estimators(viewer_state* state) {
		gui_point_cloud_vertex_noise(state);
		ImGui::Separator();
		static float sigma_g = 0.1;
		static int iterations = 1;

		static auto filter_names = filter();
		static auto estimator_names = m_estimators();
		static auto distance_names = distances();

		static int e = 0;
		static int f = 0;
		static int g = 0;
		static int h = 0;
		static int i = 0;
		
		draw_combobox(&state->window, "Filter", e, filter_names);
		if (static_cast<FilterType>(e) == FilterType::unilateral_filter) {
		
			
			draw_combobox(&state->window, "M-Estimator", f, estimator_names);

			draw_combobox(&state->window, "Input", g, distance_names);

		}
		if (static_cast<FilterType>(e) == FilterType::bilateral_filter) {

			draw_combobox(&state->window, "M-Estimator", f, estimator_names);

			draw_combobox(&state->window, "Input", g, distance_names);
		
			draw_combobox(&state->window, "M-Estimator", h, estimator_names);

			draw_combobox(&state->window, "Input", i, distance_names);

		}

		ImGui::InputFloat("sigma_g", &sigma_g);
		ImGui::InputInt("iterations", &iterations);

		static bool update_every_frame = false;
		static int count = 0;
		static int max_count = 0;
		ImGui::InputInt("max_count", &max_count);
		if (ImGui::Checkbox("update_every_frame", &update_every_frame)) {
			if (update_every_frame) {
				count = 0;
			}
		}

		if (max_count > 0 && count > max_count) update_every_frame = false;

		
		if (update_every_frame || ImGui::Button("Compute Filtering")) {
			auto id = state->picker.entity_id;
			if (state->scene.valid(id) && state->scene.has<halfedge_mesh>(id)) {
				auto& mesh = state->scene.get<halfedge_mesh>(id);
				mesh_nf_parameters params{bcg_scalar_t(sigma_g), iterations };

				std::shared_ptr<DistanceFunctor>distance_op;
				switch (static_cast<InputType>(g)) {
				case InputType::euclidean: {
					distance_op = std::make_shared<EuclideanFunctor>(EuclideanFunctor());
					break;
				}

				case InputType::angle: {
					distance_op = std::make_shared<AngleFunctor>(AngleFunctor());
					break;
				}


				case InputType::arccos: {
					distance_op = std::make_shared<ArccosFunctor>(ArccosFunctor());
					break;
				}
				}

				std::shared_ptr<MEstimatorFunctor>g_op;
				switch (static_cast<EstimatorType>(f)) {
				case EstimatorType::l2_norm: {
					g_op = std::make_shared<L2NormFunctor>(L2NormFunctor(sigma_g));
					break;
				}

				case EstimatorType::truncated_l2_norm: {
					g_op = std::make_shared<TruncatedL2NormFunctor>(TruncatedL2NormFunctor(sigma_g));
					break;
				}


				case EstimatorType::l1_norm: {
					g_op = std::make_shared<L1NormFunctor>(L1NormFunctor(sigma_g));
					break;
				}

				case EstimatorType::truncated_l1_norm: {
					g_op = std::make_shared<TruncatedL1NormFunctor>(TruncatedL1NormFunctor(sigma_g));
					break;
				}

				case EstimatorType::hubers_minimax: {
					g_op = std::make_shared<HubersMinimaxFunctor>(HubersMinimaxFunctor(sigma_g));
					break;
				}


				case EstimatorType::lorentzian_norm: {
					g_op = std::make_shared<LorentzianNormFunctor>(LorentzianNormFunctor(sigma_g));
					break;
				}

				case EstimatorType::gaussian_norm: {
					g_op = std::make_shared<GaussianNormFunctor>(GaussianNormFunctor(sigma_g));
					break;
				}

				case EstimatorType::tukeys_norm: {
					g_op = std::make_shared<TukeysNormFunctor>(TukeysNormFunctor(sigma_g));
					break;
				}
				}

				std::shared_ptr<DistanceFunctor>distance_f_op;
				switch (static_cast<InputType>(i)) {
				case InputType::euclidean: {
					distance_f_op = std::make_shared<EuclideanFunctor>(EuclideanFunctor());
					break;
				}

				case InputType::angle: {
					distance_f_op = std::make_shared<AngleFunctor>(AngleFunctor());
					break;
				}


				case InputType::arccos: {
					distance_f_op = std::make_shared<ArccosFunctor>(ArccosFunctor());
					break;
				}
				}

				std::shared_ptr<MEstimatorFunctor>f_op;
				switch (static_cast<EstimatorType>(h)) {
				case EstimatorType::l2_norm: {
					f_op = std::make_shared<L2NormFunctor>(L2NormFunctor(sigma_g));
					break;
				}

				case EstimatorType::truncated_l2_norm: {
					f_op = std::make_shared<TruncatedL2NormFunctor>(TruncatedL2NormFunctor(sigma_g));
					break;
				}


				case EstimatorType::l1_norm: {
					f_op = std::make_shared<L1NormFunctor>(L1NormFunctor(sigma_g));
					break;
				}

				case EstimatorType::truncated_l1_norm: {
					f_op = std::make_shared<TruncatedL1NormFunctor>(TruncatedL1NormFunctor(sigma_g));
					break;
				}

				case EstimatorType::hubers_minimax: {
					f_op = std::make_shared<HubersMinimaxFunctor>(HubersMinimaxFunctor(sigma_g));
					break;
				}


				case EstimatorType::lorentzian_norm: {
					f_op = std::make_shared<LorentzianNormFunctor>(LorentzianNormFunctor(sigma_g));
					break;
				}

				case EstimatorType::gaussian_norm: {
					f_op = std::make_shared<GaussianNormFunctor>(GaussianNormFunctor(sigma_g));
					break;
				}

				case EstimatorType::tukeys_norm: {
					f_op = std::make_shared<TukeysNormFunctor>(TukeysNormFunctor(sigma_g));
					break;
				}
				}

				switch (static_cast<FilterType>(e)) {
				case FilterType::unilateral_filter: {
					unilateral_filter(mesh, *distance_op, *g_op, sigma_g, state->config.parallel_grain_size);
					break;
				}
				case FilterType::bilateral_filter: {
					bilateral_filter(mesh, *distance_op, *g_op, *distance_f_op, *f_op, sigma_g, state->config.parallel_grain_size); 
					break; 
				}
				
				}
			}
			++count;
		}

		ImGui::LabelText("iterations", "%d", count);

	}

}

*/