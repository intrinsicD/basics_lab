#version 330
in vec3 f_normal;
in vec3 f_color;
in vec4 f_position;

struct Material{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    float alpha;
    bool use_uniform_color;
};

uniform Material material;

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

out vec4 final_color;

vec4 phong_shading(Light light, Material material){
    //ambient
    vec3 ambient = light.ambient * material.ambient;

    // diffuse
    vec3 normal = normalize(f_normal);
    vec3 light_dir = normalize(light.position - f_position.xyz);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.diffuse * (diff * material.diffuse);

    // specular
    vec3 view_dir = normalize(- f_position.xyz);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * material.specular);

    vec4 color;
    if (!material.use_uniform_color){
        color = vec4(ambient + f_color + specular, material.alpha);
    } else {
        color = vec4(ambient + diffuse + specular, material.alpha);
    }
    return color;
}

void main() {
    Light light;
    light.position = vec3(0, 0, 0);
    light.ambient = vec3(0.2, 0.2, 0.2);
    light.diffuse = vec3(0.5, 0.5, 0.5);
    light.specular = vec3(1, 1, 1);

    final_color = phong_shading(light, material);
}
