//
// Created by alex on 28.10.20.
//

#include "bcg_graph_system.h"
#include "bcg_viewer_state.h"
#include "aligned_box/bcg_aligned_box.h"
#include "bcg_entity_info.h"
#include "bcg_property_map_eigen.h"
#include "renderers/picking_renderer/bcg_events_picking_renderer.h"
#include "renderers/graph_renderer/bcg_events_graph_renderer.h"
#include "graph/bcg_graph_vertex_pca.h"

namespace bcg{

graph_system::graph_system(viewer_state *state) : system("graph_system", state){
    state->dispatcher.sink<event::graph::setup>().connect<&graph_system::on_setup>(this);
    state->dispatcher.sink<event::graph::vertex::pca>().connect<&graph_system::on_vertex_pca>(this);
}

void graph_system::on_setup(const event::graph::setup &event){
    auto &graph = state->scene.get<halfedge_graph>(event.id);

    state->dispatcher.trigger<event::transform::add>(event.id);

    aligned_box3 aabb(graph.positions.vector());
    state->scene.emplace<entity_info>(event.id, event.filename, "graph", aabb.center(), aabb.halfextent().maxCoeff());

    Map(graph.positions) =
            (MapConst(graph.positions).rowwise() - aabb.center().transpose()) / aabb.halfextent().maxCoeff();

    state->dispatcher.trigger<event::mesh::vertex_normals::area_angle>(event.id);
    state->dispatcher.trigger<event::mesh::face::centers>(event.id);
    state->dispatcher.trigger<event::graph::edge::centers>(event.id);
    state->dispatcher.trigger<event::aligned_box::add>(event.id);
    state->scene.emplace_or_replace<event::picking_renderer::enqueue>(event.id);
    state->scene.emplace_or_replace<event::graph_renderer::enqueue>(event.id);
    state->picker.entity_id = event.id;
    std::cout << graph << "\n";
}

void graph_system::on_vertex_pca(const event::graph::vertex::pca &event){
    if(!state->scene.valid(event.id)) return;
    if(!state->scene.has<halfedge_graph>(event.id)) return;
    halfedge_graph &graph = state->scene.get<halfedge_graph>(event.id);

    switch(event.type){
        case PcaType::svd : {
            graph_local_pcas(graph, graph_local_pca_least_squares_svd, event.compute_mean, state->config.parallel_grain_size);
            break;
        }
        case PcaType::weighted_svd : {
            graph_local_pcas(graph, graph_local_pca_weighted_least_squares_svd, event.compute_mean, state->config.parallel_grain_size);
            break;
        }
        case PcaType::eig : {
            graph_local_pcas(graph, graph_local_pca_least_squares_eig, event.compute_mean, state->config.parallel_grain_size);
            break;
        }
        case PcaType::weighted_eig : {
            graph_local_pcas(graph, graph_local_pca_weighted_least_squares_eig, event.compute_mean, state->config.parallel_grain_size);
            break;
        }
    }
}

}