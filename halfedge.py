import numpy as np
from scipy import sparse
import pyvista as pv

NULL_HALF_EDGE = -1
BOUNDARY_HALF_EDGE = -1
BOUNDARY_FACE = -1

class HalfEdgeMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """
        Assume that faces are consistently oriented
        """
        self.vertices = vertices
        self.faces = faces
        n_vertices = vertices.shape[0]
        n_faces = faces.shape[0]
        he_index = np.arange(faces.size).reshape(faces.shape)
        he_face = np.arange(n_faces)[:, None] * np.ones((1, 3), dtype=int)
        self.he_face = he_face.flatten()
        self.he_tip = np.roll(faces, -1, axis=1).flatten()
        self.he_next = np.roll(he_index, -1, axis=1).flatten()
        self.he_prev = np.roll(he_index, 1, axis=1).flatten()
        self.he_pair = self._init_pairs(he_index, n_vertices)

        self.vertex_to_he = NULL_HALF_EDGE * np.ones(n_vertices, dtype=int)
        self.vertex_to_he[faces.flatten()] = he_index.flatten()

        self._init_boundary()

    @staticmethod
    def _init_pairs(he_index: np.ndarray, n_vertices: int) -> np.ndarray:
        n_half_edges = he_index.size
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        indices = np.array([], dtype=int)
        for i in range(3):
            rows = np.concatenate((rows, faces[:, i]))
            cols = np.concatenate((cols, faces[:, (i+1) % 3]))
            indices = np.concatenate((indices, he_index[:, i]))
        ## NOTE: Need to 1-index the sparse matrix so half edge 0 doesn't disappear
        he_pair_mat_transpose = sparse.coo_matrix((indices+1, (rows, cols)),
                                         shape=(n_vertices, n_vertices)).T.tocsr()
        he_pair = BOUNDARY_HALF_EDGE * np.ones(n_half_edges, int)
        he_pair[indices] = he_pair_mat_transpose[rows, cols] - 1
        return he_pair

    def _init_boundary(self) -> None:
        interior_boundary_he = np.where(self.he_pair == BOUNDARY_HALF_EDGE)[0]
        n_half_edges = len(self.he_pair)
        n_new_he = len(interior_boundary_he)
        self.he_face = np.concatenate((self.he_face, BOUNDARY_FACE*np.ones(n_new_he)))
        exterior_boundary_he = n_half_edges + np.arange(n_new_he)
        self.he_pair = np.concatenate((self.he_pair, interior_boundary_he)) # exterior pair to interior
        self.he_pair[interior_boundary_he] = exterior_boundary_he # interior pair to exterior

        x = self.he_tip[interior_boundary_he]
        self.vertex_to_he[x] = exterior_boundary_he
        y = self.he_tip[self.he_prev[interior_boundary_he]]
        self.he_next = np.concatenate((self.he_next, self.vertex_to_he[y]))
        self.he_tip = np.concatenate((self.he_tip, y))
        self.he_prev = np.concatenate((self.he_prev, np.zeros(n_new_he, dtype=int)))
        self.he_prev[self.vertex_to_he[y]] = exterior_boundary_he

    def get_vertex_neighbors(self, index: int) -> list[int]:
        neighbors = []
        he_start = self.vertex_to_he[index]
        if he_start == NULL_HALF_EDGE:
            return neighbors
        he = he_start
        for _ in range(len(self.he_face)):
            neighbors.append(self.he_tip[he])
            # he = self.he_next[self.he_pair[he]] # Walks clockwise
            he = self.he_pair[self.he_prev[he]] # Walks counter-clockwise
            if he == he_start:
                break
        return neighbors

    def get_boundary_vertices(self) -> np.ndarray:
        interior_boundary_he = np.where(self.he_face == BOUNDARY_FACE)[0]
        boundary_vertices = self.he_tip[interior_boundary_he]
        return boundary_vertices

    def get_euler_chacteristic(self) -> int:
        """
        # vertices - # edges + #faces
        """
        if len(self.get_boundary_vertices() > 0):
            raise ValueError("Cannot get Euler characteristic of mesh that is not watertight")
        n_vertices = len(self.vertex_to_he)
        n_edges = len(self.he_pair) // 2
        n_faces = self.faces.shape[0]
        return n_vertices - n_edges + n_faces

    def pick_neighbor(self):
        plotter = pv.Plotter()
        packed_faces = 3 * np.ones((self.faces.shape[0], 4), dtype=int)
        packed_faces[:, 1:] = self.faces
        mesh = pv.PolyData(self.vertices, faces=packed_faces)
        plotter.add_mesh(mesh, color="lightblue", show_edges=False, line_width=3)

        def point_picker_callback(_, index):
            coord = mesh.points[index]
            # plotter.add_point_labels(coord, label_text, point_size=20, font_size=16, name="picked_label")
            neighbors = self.get_vertex_neighbors(index)
            for i, n in enumerate(neighbors):
                line = pv.Line(self.vertices[index], self.vertices[n])
                plotter.add_mesh(line, color="red", line_width=5, name=f"neighbor{i}")

        boundary_points = pv.PolyData(self.vertices[self.get_boundary_vertices()])
        plotter.add_mesh(boundary_points, render_points_as_spheres=True,
                   point_size=8,  # size in pixels
                   color="green",
                   show_scalar_bar=False)

        plotter.enable_point_picking(
            callback=point_picker_callback,
            use_mesh=True,
            left_clicking=True,
            show_point=True,
            point_size=15,
            color="magenta"
        )

        plotter.add_axes()
        plotter.show()


mesh = pv.read("meshes/genusthree.obj")
vertices = mesh.points
faces = mesh.faces.reshape((mesh.faces.size//4, 4))[:, 1:]
mesh = HalfEdgeMesh(vertices, faces)
print("Euler:", mesh.get_euler_chacteristic())
mesh.pick_neighbor()