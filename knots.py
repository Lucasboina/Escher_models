import numpy as np
import pyvista as pv

# --- Parâmetros Configuráveis ---
NUM_POINTS = 5000             # Número de pontos na curva central (mais pontos = mais suave).
TUBE_RADIUS = 2           # Raio do tubo que forma o nó. (Valor aumentado para um nó mais grosso)
TUBE_SIDES = 30               # Número de lados do tubo (resolução da seção transversal).
SCALE_FACTOR = 4.0            # Fator de escala geral para aumentar o tamanho do nó.


# 1. Gerar a Curva Central (Centerline)
def set_points():
    print("Gerando esqueleto...")
    
    t = np.linspace(0, 2 * np.pi, NUM_POINTS)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    points = np.vstack((x, y, z)).T * SCALE_FACTOR
    return points

# 2. Criar a Malha do Tubo (corrigido)
def apply_mesh(points):
    print("Gerando malha...")
    
    centerline = pv.lines_from_points(points)
    knot_mesh = centerline.tube(radius=TUBE_RADIUS, n_sides=TUBE_SIDES)
    return knot_mesh

# 3. Aplicar o Degradê de Cores para simular o brilho
def apply_colors(knot_mesh):
    print("Aplicando as cores...")
    
    vertices = knot_mesh.points
    y_coords = vertices[:, 1]
    min_y, max_y = y_coords.min(), y_coords.max()
    normalized_y = (y_coords - min_y) / (max_y - min_y)

    color_dark_blue = np.array([0, 100, 255])
    color_light_blue = np.array([200, 240, 255])
    colors = np.array([
        (color_dark_blue * (1 - val)) + (color_light_blue * val)
        for val in normalized_y
    ]).astype(np.uint8)

    knot_mesh.point_data['colors'] = colors
    return knot_mesh

# 4. Visualizar a Cena
def render_model(knot_mesh):
    print("Renderizando o modelo...")
    
    plotter = pv.Plotter(window_size=[800, 800])
    plotter.add_mesh(
        knot_mesh,
        scalars='colors',
        rgb=True,
        smooth_shading=True
    )
    plotter.set_background('black')
    plotter.enable_anti_aliasing('fxaa')
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.2)
    plotter.show()
    
if __name__ == "__main__":
    print("Iniciando a geração do Nó de Trevo (Trefoil Knot)...")
    points = set_points()
    knot_mesh  =apply_mesh(points)
    knot_mesh = apply_colors(knot_mesh)
    
    render_model(knot_mesh)
    print("Geração finalizada.")