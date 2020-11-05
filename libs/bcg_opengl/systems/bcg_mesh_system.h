//
// Created by alex on 28.10.20.
//

#ifndef BCG_GRAPHICS_BCG_MESH_SYSTEM_H
#define BCG_GRAPHICS_BCG_MESH_SYSTEM_H

#include "bcg_systems.h"

namespace bcg{

struct mesh_system : public system{
    explicit mesh_system(viewer_state *state);

    void on_make_triangle(const event::mesh::make_triangle &event);

    void on_setup_mesh(const event::mesh::setup &event);
};

}

#endif //BCG_GRAPHICS_BCG_MESH_SYSTEM_H
