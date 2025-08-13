import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

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
    :param mesh: O objeto de malha trimesh a ser colorido.
    :param color_stops_rgb: Uma lista de cores [R, G, B] no formato 0-255.
    """
    vertices = mesh.vertices
    if len(vertices) == 0:
        return mesh

    # Calcula a distância (raio) de cada vértice ao centro (0,0,0)
    radii = np.linalg.norm(vertices, axis=1)
    min_radius, max_radius = np.min(radii), np.max(radii)

    # Evita divisão por zero se todos os raios forem iguais
    if max_radius == min_radius:
        normalized_radii = np.zeros_like(radii)
    else:
        # Normaliza os raios para o intervalo [0, 1]
        normalized_radii = (radii - min_radius) / (max_radius - min_radius)

    # Prepara os canais de cores para interpolação
    color_stops_array = np.array(color_stops_rgb)
    red_channel = color_stops_array[:, 0]
    green_channel = color_stops_array[:, 1]
    blue_channel = color_stops_array[:, 2]

    # Pontos de interpolação (distribuídos igualmente de 0 a 1)
    interp_points = np.linspace(0, 1, len(color_stops_rgb))

    # Interpola cada canal de cor com base nos raios normalizados
    r = np.interp(normalized_radii, interp_points, red_channel)
    g = np.interp(normalized_radii, interp_points, green_channel)
    b = np.interp(normalized_radii, interp_points, blue_channel)

    # Combina os canais de volta em cores de vértice e atribui à malha
    vertex_colors = np.vstack([r, g, b]).T.astype(np.uint8)
    mesh.visual.vertex_colors = vertex_colors
    
    print("Degradê de cores radial aplicado com sucesso.")
    return mesh


def get_great_circle_normals():
    """
    Define os vetores normais para os 9 grandes círculos (simetria octaédrica).
    """
    normals = [
        np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
        np.array([1, 1, 0]), np.array([1, -1, 0]), np.array([1, 0, 1]),
        np.array([1, 0, -1]), np.array([0, 1, 1]), np.array([0, 1, -1])
    ]
    return [n / np.linalg.norm(n) for n in normals]

def calculate_rotation_matrix(initial_normal, target_normal):
    """Calcula a matriz de rotação 4x4 para alinhar vetores."""
    r, _ = Rotation.align_vectors(target_normal[np.newaxis, :], initial_normal[np.newaxis, :])
    transform_matrix_4x4 = np.eye(4)
    transform_matrix_4x4[:3, :3] = r.as_matrix()
    return transform_matrix_4x4

def create_single_rind(major_radius, minor_radius, transform_matrix):
    """Cria um único anel (toro) e aplica a transformação."""
    rind_mesh = trimesh.creation.torus(
        major_radius=major_radius, minor_radius=minor_radius,
        major_sections=TORUS_MAJOR_SECTIONS, minor_sections=TORUS_MINOR_SECTIONS
    )
    rind_mesh.apply_transform(transform_matrix)
    return rind_mesh

def generate_geometry():
    """Gera todas as peças geométricas (esfera central e anéis)."""
    print("1. Gerando a geometria das cascas e da esfera central...")
    all_meshes = []
    if ADD_CENTRAL_SPHERE:
        print(f"Gerando esfera central com raio: {CENTRAL_SPHERE_RADIUS}")
        central_sphere = trimesh.creation.icosphere(subdivisions=5, radius=CENTRAL_SPHERE_RADIUS)
        all_meshes.append(central_sphere)

    great_circle_normals = get_great_circle_normals()
    initial_torus_normal = np.array([0, 0, 1])
    for i in range(NUM_CONCENTRIC_SHELLS):
        current_radius = BASE_SPHERE_RADIUS + i * SHELL_SPACING
        print(f" -> Gerando Casca {i+1}/{NUM_CONCENTRIC_SHELLS} com Raio: {current_radius:.2f}")
        for j, target_normal in enumerate(great_circle_normals):
            rotation_matrix = calculate_rotation_matrix(initial_torus_normal, target_normal)
            rind_mesh = create_single_rind(current_radius, RIND_THICKNESS, rotation_matrix)
            all_meshes.append(rind_mesh)
    return all_meshes

def combine_meshes(meshes_list):
    """Combina uma lista de malhas em uma única malha final."""
    if not meshes_list:
        return None
    
    print("2. Combinando todas as partes em uma única malha...")
    if APPLY_BOOLEAN_UNION:
        try:
            # A união booleana cria uma superfície contínua, ideal para impressão 3D
            final_mesh = trimesh.boolean.union(meshes_list, engine='manifold')
            print(" -> União booleana concluída com sucesso.")
        except Exception as e:
            print(f" -> Erro na união booleana: {e}. Recorrendo à concatenação.")
            final_mesh = trimesh.util.concatenate(meshes_list)
    else:
        # A concatenação apenas junta as malhas, mais rápido mas pode ter faces internas
        final_mesh = trimesh.util.concatenate(meshes_list)
        print(" -> Concatenação de malhas concluída.")
    return final_mesh

def apply_colors(mesh):
    """Aplica o degradê de cores à malha final."""
    if not APPLY_COLOR_GRADIENT or not mesh:
        return mesh
    
    print("3. Aplicando o degradê de cores...")
    color_gradient = [
        [255, 255, 0],   # Amarelo (interno)
        [255, 165, 0],   # Laranja
        [255, 0, 255],   # Magenta
        [128, 0, 128]    # Roxo (externo)
    ]
    return apply_radial_gradient(mesh, color_gradient)

def render_model(mesh):
    """Renderiza e exibe o modelo 3D final."""
    if not SHOW_VISUALIZATION or not mesh:
        return
    
    print("4. Exibindo visualização do modelo...")
    # Define a cor de fundo para preto (RGBA: 0, 0, 0, 255)
    viewer_kwargs = {
        'background': [0, 0, 0, 255]
    }
    scene = trimesh.Scene(mesh)
    scene.show(**viewer_kwargs)
    print(" -> Visualização concluída. Feche a janela para continuar.")

# --- Execução Principal ---
if __name__ == "__main__":
    print("Iniciando a geração das 'Cascas Concêntricas' de Escher...")

    # 1. Gerar todas as peças individuais
    list_of_meshes = generate_geometry()

    # 2. Unir as peças em uma única malha
    final_mesh = combine_meshes(list_of_meshes)

    # 3. Aplicar o esquema de cores
    final_mesh = apply_colors(final_mesh)
    
    # 4. Mostrar o resultado
    render_model(final_mesh)

    print("\nGeração finalizada.")