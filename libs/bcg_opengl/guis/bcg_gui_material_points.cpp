//
// Created by alex on 16.11.20.
//

#include "bcg_gui_material_points.h"
#include "bcg_viewer_state.h"
#include "bcg_gui_property_selector.h"
#include "renderers/points_renderer/bcg_events_points_renderer.h"

namespace bcg{

void gui_material_points(viewer_state *state, material_points *material, entt::entity id){
    if (!material) return;
    ImGui::PushID("point_material");
    auto *vertices = state->get_vertices(id);
    auto &position = material->attributes[0];
    draw_label(&state->window, (material->vao.name + " vao id").c_str(), std::to_string(material->vao.handle));
    if (ImGui::CollapsingHeader("Captured Attributes")) {
        for (const auto &item : material->vao.captured_attributes) {
            draw_label(&state->window, std::to_string(item.first).c_str(), item.second);
        }
    }
    if (gui_property_selector(state, vertices, {3}, position.shader_attribute_name.c_str(),
                                   position.property_name)) {
        if (position.property_name.empty()) {
            position.property_name = "v_position";
        }
        state->dispatcher.trigger<event::points_renderer::set_position_attribute>(id, position);
    }
    auto &color = material->attributes[1];
    if (gui_property_selector(state, vertices, {1, 3}, color.shader_attribute_name, color.property_name)) {
        if (color.property_name.empty()) {
            material->use_uniform_color = true;
        } else {
            state->dispatcher.trigger<event::points_renderer::set_color_attribute>(id, color);
        }
    }
    auto &point_size = material->attributes[2];
    if (gui_property_selector(state, vertices, {1}, point_size.shader_attribute_name, point_size.property_name)) {
        if (point_size.property_name.empty()) {
            material->use_uniform_size = true;
        } else {
            state->dispatcher.trigger<event::points_renderer::set_point_size_attribute>(id, point_size);
        }
    }
    if (ImGui::Checkbox("use_uniform_color", &material->use_uniform_color)) {
        if (material->use_uniform_color) {
            color.property_name = "";
        }
    }
    if (ImGui::Checkbox("use_uniform_point_size", &material->use_uniform_size)) {
        if (material->use_uniform_size) {
            point_size.property_name = "";
        }
    }
    draw_coloredit(&state->window, "uniform_color", material->uniform_color);
    ImGui::InputFloat("alpha", &material->alpha);
    ImGui::PopID();
}

}