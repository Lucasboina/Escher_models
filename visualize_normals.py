import pyvista as pv
import numpy as np

def get_great_circle_normals():
    """
    Define os vetores normais para os 9 grandes círculos (simetria octaédrica).
    """
    normals = [
        np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
        np.array([1, 1, 0]), np.array([1, -1, 0]), np.array([1, 0, 1]),
        np.array([1, 0, -1]), np.array([0, 1, 1]), np.array([0, 1, -1])
    ]
    # Normaliza os vetores para que todos tenham comprimento 1
    return [n / np.linalg.norm(n) for n in normals]

def plot_vectors():
    """
    Cria uma cena 3D para visualizar os vetores normais.
    """
    normals = get_great_circle_normals()
    
    plotter = pv.Plotter(window_size=[800, 800])
    
    # Adiciona um marcador de eixos para orientação
    plotter.add_axes()
    
    # Ponto de origem para todas as setas
    origin = np.array([0, 0, 0])
    
    # Adiciona cada vetor como uma seta na cena
    for i, vec in enumerate(normals):
        plotter.add_arrows(origin, vec, mag=1.0, color='cyan', label=f'{vec.round(2)}')

    plotter.add_point_labels([v * 1.1 for v in normals], [f'[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]' for v in normals], font_size=12)
    
    plotter.set_background('black')
    plotter.camera_position = 'iso'
    plotter.show()

if __name__ == "__main__":
    plot_vectors()