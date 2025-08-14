import numpy as np
import pyvista as pv
import vtk # <-- ADICIONAR ESTA LINHA

# --- Parâmetros Configuráveis ---
BASE_SPHERE_RADIUS = 1.0       # Raio da casca esférica mais interna.
RIND_THICKNESS = 0.05          # Raio menor dos toros (espessura das cascas).
NUM_CONCENTRIC_SHELLS = 4      # Número de cascas concêntricas a gerar.
SHELL_SPACING = 0.5            # Distância incremental entre os raios de cascas concêntricas sucessivas.
TORUS_MAJOR_SECTIONS = 128     # Resolução do círculo maior do toro (número de segmentos).
TORUS_MINOR_SECTIONS = 64      # Resolução do círculo menor do toro (número de segmentos).

ADD_CENTRAL_SPHERE = True      # Se True, adiciona uma esfera sólida no centro.
CENTRAL_SPHERE_RADIUS = 0.6    # Raio da esfera central. Deve ser menor que BASE_SPHERE_RADIUS.
APPLY_COLOR_GRADIENT = True    # Se True, aplica o degradê de cores.

APPLY_BOOLEAN_UNION = True     # Se True, todas as cascas serão unidas em uma única malha.
SHOW_VISUALIZATION = True      # Se True, o modelo será visualizado interativamente.


def apply_radial_gradient(mesh, color_stops_rgb):
    """
    Aplica um degradê de cores a uma malha com base na distância de cada vértice ao centro.
    """
    vertices = mesh.points
    if len(vertices) == 0:
        return mesh

    radii = np.linalg.norm(vertices, axis=1)
    min_radius, max_radius = np.min(radii), np.max(radii)

    if max_radius == min_radius:
        normalized_radii = np.zeros_like(radii)
    else:
        normalized_radii = (radii - min_radius) / (max_radius - min_radius)

    color_stops_array = np.array(color_stops_rgb)
    interp_points = np.linspace(0, 1, len(color_stops_rgb))

    r = np.interp(normalized_radii, interp_points, color_stops_array[:, 0])
    g = np.interp(normalized_radii, interp_points, color_stops_array[:, 1])
    b = np.interp(normalized_radii, interp_points, color_stops_array[:, 2])

    vertex_colors = np.vstack([r, g, b]).T.astype(np.uint8)
    mesh.point_data['colors'] = vertex_colors
    
    print(" -> Degradê de cores radial aplicado com sucesso.")
    return mesh


def get_great_circle_normals():
    """Define os vetores normais para os 9 grandes círculos (simetria octaédrica)."""
    normals = [
        np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
        np.array([1, 1, 0]), np.array([1, -1, 0]), np.array([1, 0, 1]),
        np.array([1, 0, -1]), np.array([0, 1, 1]), np.array([0, 1, -1])
    ]
    return [n / np.linalg.norm(n) for n in normals]

# -----------------------------------------------------------------------------
# *** FUNÇÃO TOTALMENTE REESCRITA PARA MÁXIMA COMPATIBILIDADE ***
# -----------------------------------------------------------------------------
def create_single_rind(major_radius, minor_radius, target_normal):
    """Cria um único anel (toro) e o rotaciona para a orientação correta de forma robusta."""
    
    # Passo 1: Criar a fonte paramétrica VTK para o toro
    source = vtk.vtkParametricTorus()
    source.SetRingRadius(major_radius)       # Define o raio principal
    source.SetCrossSectionRadius(minor_radius) # Define o raio do tubo

    # Passo 2: Criar a malha PyVista a partir da fonte VTK
    rind_mesh = pv.surface_from_para(
        source,
        u_res=TORUS_MAJOR_SECTIONS, # Mapeia a resolução para os parâmetros corretos
        v_res=TORUS_MINOR_SECTIONS
    )

    # O resto da lógica de rotação, que já está correta, permanece a mesma.
    initial_normal = np.array([0, 0, 1])
    target_normal = target_normal / np.linalg.norm(target_normal)
    dot_product = np.dot(initial_normal, target_normal)

    if np.allclose(dot_product, 1):
        return rind_mesh
    
    if np.allclose(dot_product, -1):
        rind_mesh.rotate_x(180, inplace=True)
        return rind_mesh

    rotation_axis = np.cross(initial_normal, target_normal)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    rind_mesh.rotate_vector(rotation_axis, np.degrees(angle_rad), inplace=True)

    return rind_mesh

def generate_geometry():
    """Gera todas as peças geométricas (esfera central e anéis)."""
    print("1. Gerando a geometria das cascas e da esfera central...")
    all_meshes = []
    if ADD_CENTRAL_SPHERE:
        print(f" -> Gerando esfera central com raio: {CENTRAL_SPHERE_RADIUS}")
        central_sphere = pv.Sphere(radius=CENTRAL_SPHERE_RADIUS, phi_resolution=60, theta_resolution=60)
        all_meshes.append(central_sphere)

    great_circle_normals = get_great_circle_normals()
    for i in range(NUM_CONCENTRIC_SHELLS):
        current_radius = BASE_SPHERE_RADIUS + i * SHELL_SPACING
        print(f" -> Gerando Casca {i+1}/{NUM_CONCENTRIC_SHELLS} com Raio: {current_radius:.2f}")
        for j, target_normal in enumerate(great_circle_normals):
            rind_mesh = create_single_rind(current_radius, RIND_THICKNESS, target_normal)
            all_meshes.append(rind_mesh)
    return all_meshes

def combine_meshes(meshes_list):
    """Combina uma lista de malhas em uma única malha final."""
    if not meshes_list:
        return None
    
    print("2. Combinando todas as partes em uma única malha...")
    if APPLY_BOOLEAN_UNION:
        try:
            print(" -> Iniciando união booleana (pode levar algum tempo)...")
            final_mesh = meshes_list[0].copy()
            total = len(meshes_list)
            for i in range(1, total):
                print(f"      - Unindo malha {i+1}/{total}...", end='\r')
                final_mesh = final_mesh.boolean_union(meshes_list[i], progress_bar=False)
            print("\n -> União booleana concluída com sucesso.              ")
        except Exception as e:
            print(f"\n -> Erro na união booleana: {e}. Recorrendo à concatenação (merge).")
            final_mesh = pv.merge(meshes_list)
    else:
        final_mesh = pv.merge(meshes_list)
        print(" -> Concatenação de malhas (merge) concluída.")
    return final_mesh

def apply_colors(mesh):
    """Aplica o degradê de cores à malha final."""
    if not APPLY_COLOR_GRADIENT or not mesh:
        return mesh
    
    print("3. Aplicando o degradê de cores...")
    color_gradient = [
        [255, 255, 0],   # Amarelo (interno)
        [255, 100, 0],   # Laranja
        [255, 0, 255],   # Magenta
        [128, 0, 128]    # Roxo (externo)
    ]
    return apply_radial_gradient(mesh, color_gradient)

def render_model(mesh):
    """Renderiza e exibe o modelo 3D final."""
    if not SHOW_VISUALIZATION or not mesh:
        return
    
    print("4. Exibindo visualização do modelo...")
    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.add_mesh(
        mesh,
        scalars='colors' if APPLY_COLOR_GRADIENT else None,
        rgb=APPLY_COLOR_GRADIENT,
        smooth_shading=True,
    )
    plotter.set_background('black')
    plotter.enable_anti_aliasing('fxaa')
    plotter.show()
    print(" -> Visualização concluída.")

if __name__ == "__main__":
    print("Iniciando a geração das 'Cascas Concêntricas' de Escher (versão PyVista)...")

    list_of_meshes = generate_geometry()
    final_mesh = combine_meshes(list_of_meshes)
    final_mesh = apply_colors(final_mesh)
    render_model(final_mesh)

    print("\nGeração finalizada.")