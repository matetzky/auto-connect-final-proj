from scipy import optimize
from scipy.spatial.distance import hamming
from sklearn.cluster import SpectralClustering
from trimesh.base import Trimesh
from mesh_utils import *
from typing import Optional
import numpy as np
import math


class trimesh_handler:
    mesh: Trimesh  # original mesh (input)
    current_holder_T: Trimesh  # parameter T from the paper
    maximum_holder: Trimesh  # Holder holds all legal triangles
    is_convex_hull: bool  # is original mesh a convex_hull
    whole_mesh_holdability: float  # original mesh holdability
    alpha: float  # alpha parameter for free motions calculations
    intrinsic_free_motions: list  # computed internal free motion
    symmetry_planes: list  # list of symmetry planes of the mesh
    symmetry_sections: list  # list of symmetry sections of the mesh
    minimal_distances_of_triangles: list  # list of minimal distances of triangles center from any symmetry plane
    results: list  # list of holders
    constraints_input: list[dict]  # constraints from user received as input
    free_motions_input: list[list]  # free motions inserted by user
    ignored_triangles: list[int]  # list of triangles indexes to ignore (blocking)

    def calc_symmetry_planes(self):
        sp = []
        if self.mesh.symmetry:
            s0 = self.mesh.symmetry_section[0]
            s1 = self.mesh.symmetry_section[1]
            sections = [self.mesh.section(s1, s0), self.mesh.section(s0, s1)]
            for s in sections:
                p0 = extract_point(s.vertices, 0)
                p4 = extract_point(s.vertices, 4)
                p7 = extract_point(s.vertices, 7)

                v = p7 - p0
                u = p4 - p0
                a, b, c = n = np.cross(u, v)
                d = np.dot(n, p7)
                sp.append([a, b, c, d])
            self.symmetry_sections = sections
        else:
            self.symmetry_sections = None
        self.symmetry_planes = sp

    def init_shell(self, starting_point):
        mesh_size = len(self.mesh.triangles_center)
        random_vec = np.random.randint(0, 2, 3)
        rw1, rw2, rw3 = np.random.random_sample((3,))
        sum_of_weights = weighted_sum(rw1, rw2, rw3, starting_point, self, random_vec)
        shell_triangles = []
        sorted_triangles = []
        for triangle in range(mesh_size):
            if triangle not in self.ignored_triangles:
                weight = 0
                shell_triangles.append([triangle])
            else:
                weight = np.inf
            sorted_triangles.append([triangle, weight + sum_of_weights[triangle]])
        sorted_triangles.sort(key=lambda item: item[1])
        self.maximum_holder = self.mesh.submesh(shell_triangles, append=True)
        return sorted_triangles

    def pre_process_mesh(self, constraints):
        self.calc_symmetry_planes()
        self.minimal_distances_of_triangles = list[d_symm(self.mesh.triangles_center, self.symmetry_planes)]
        self.constraints_input = []
        self.current_holder_T = Trimesh
        self.maximum_holder = Trimesh
        intrinsic_free_motions(self)
        external_free_motions(self, constraints)
        self.ignored_triangles = find_triangles_to_ignore(self)

    def export(self, path):
        self.mesh.export(path)

    def __init__(self, mesh, constraints, convex_hull=False, alpha=0.5):
        self.mesh = mesh.convex_hull if convex_hull else mesh
        self.is_convex_hull = convex_hull
        self.alpha = alpha
        self.whole_mesh_holdability = 0
        self.pre_process_mesh(constraints)


def extract_point(v, i):
    return np.array((v[i][0], v[i][1], v[i][2]))


## 4.1 - Shell Computation ##

def calc_starting_point(mesh_h):
    if not mesh_h.symmetry_planes:
        res = mesh_h.mesh.triangles_center[np.random.randint(0, len(mesh_h.mesh.triangles_center) - 1)]
    else:
        plane = np.random.randint(0, len(mesh_h.symmetry_sections) - 1)
        point = np.random.randint(0, len(mesh_h.symmetry_sections[plane].vertices) - 1)
        res = mesh_h.symmetry_sections[plane].vertices[point]
    return res


def shell_computation(mesh_w, starting_point, holdability_th=0.1, weight_th=np.inf):
    ordered_triangles = mesh_w.init_shell(starting_point)

    whole_mesh_args = (mesh_w.mesh.triangles_center, mesh_w.mesh.face_normals)
    maximum_holdability = optimize.minimize(fun=subregion_blockage, x0=[0 for i in range(6)], method='COBYLA',
                                            args=whole_mesh_args, constraints=mesh_w.constraints_input)
    mesh_w.whole_mesh_holdability = maximum_holdability.fun

    num_of_triangles = len(ordered_triangles)
    shell_triangles = []
    shell_vectors = [0 for i in range(num_of_triangles)]
    for i in range(num_of_triangles):

        if i in [0.25 * num_of_triangles, 0.5 * num_of_triangles, 0.75 * num_of_triangles]:
            print(
                f'\t\t   Used {i / num_of_triangles:0.2f}% of triangles: current normalized holdability={normalized_holdability_value:.4f}')

        if ordered_triangles[i][1] >= weight_th:
            print(
                f'\t\t > Reached infinite weight using {i} triangles: normalized holdability={normalized_holdability_value:.4f} [didn`t reach {holdability_th}]')
            break

        t = ordered_triangles[i][0]
        shell_triangles.append([t])
        shell_vectors[t] = 1
        mesh_w.current_holder_T = mesh_w.mesh.submesh(shell_triangles, append=True)

        optimization_args = (mesh_w.current_holder_T.triangles_center, mesh_w.current_holder_T.face_normals)
        current_holdability = optimize.minimize(fun=subregion_blockage, x0=[0 for i in range(6)], method='COBYLA',
                                                args=optimization_args, constraints=mesh_w.constraints_input)
        normalized_holdability_value = current_holdability.fun / mesh_w.whole_mesh_holdability

        if normalized_holdability_value > epsilon:
            print(f'\t\t > Reached non-zero holdability for shell using {i} out of {num_of_triangles} triangles')

        if normalized_holdability_value >= holdability_th:
            print(
                f'\t\t > Reached holdability threshold using {i} triangles: normalized holdability={normalized_holdability_value:.4f}')
            break

    return shell_vectors


# The Geodesic distance between centers of triangles t, s
def d_geod(centers, seed_center):
    return np.array([points_distance(centers[i], seed_center) for i in range(len(centers))])


# The distance between triangle t and symmetry plane Pk
def d_symm(centers, symmetry_planes):
    n = len(centers)
    if not symmetry_planes:
        minimal_distances = [0 for i in range(n)]
    else:
        m = len(symmetry_planes)
        minimal_distances = []
        for t in range(n):
            min_distance = np.inf
            for s in range(m):
                symmetry_plane = symmetry_planes[s]
                triangle_center = centers[t]
                min_distance = min(min_distance, min_dis_p2plane(triangle_center, symmetry_plane))
            minimal_distances.append(min_distance)
    return np.array(minimal_distances)


# The angle between vector normal of triangle 't' and global direction N
def d_norm(normals, N):
    return np.array([get_angle(N, normals[i]) for i in range(len(normals))])


def weighted_sum(w1, w2, w3, starting_point, mesh_w, global_vec):
    w_geod = w1 * d_geod(mesh_w.mesh.triangles_center, starting_point)
    w_symm = w2 * d_symm(mesh_w.mesh.triangles_center, mesh_w.symmetry_planes)
    w_norm = w3 * d_norm(mesh_w.mesh.face_normals, global_vec)
    return w_geod + w_symm + w_norm


## 4.2 - Holdability Criterion ##

# calculate b(p, phi): measures the amount of blockage experienced
def contact_blockage(point, normal, phi, b_th=0):
    assert type(phi) in [np.ndarray, list], f'Phi is {type(phi)}: Phi={phi}'
    identity_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
    x, y, z = point
    point_matrix = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])
    matrix = np.concatenate((np.transpose(point_matrix), identity_matrix), axis=1)
    try:
        b = float(np.dot(normal, np.dot(matrix, np.transpose(phi))))
    except:
        print(f'Normal:\n{normal}\n')
        print(f'Matrix:\n{matrix}\n')
        print(f'Phi   :\n{phi} \n')
        print(f'Shpaes: N:{normal.shape}, Matrix:{matrix.shape}, Phi:{phi.shape}')
        exit(9)
    if b > b_th:
        return b
    return 0

#calculate B(T, phi)
def subregion_blockage(phi,centers,normals,):
    res = 0
    for i in range(len(centers)):
        res += contact_blockage(point=centers[i], normal=normals[i], phi=phi)
    return res


## 4.3 - Free Motions ##

def function_for_constraint(phi, x, cone_factor):
    return (math.degrees(get_angle(x, phi)) - cone_factor)



def intrinsic_free_motions(mesh_w, cone_factor= 30):

    def sumsq_minus_one(input):
        return (np.sum(np.square(input)) - 1)

    def one_minus_sumsq(input):
        return (1 - np.sum(np.square(input)))

    intrinsic_free_motions = []
    constraints = [{'type': 'ineq', 'fun': one_minus_sumsq}, {'type': 'ineq', 'fun': sumsq_minus_one}]
    args = (mesh_w.mesh.triangles_center, mesh_w.mesh.face_normals)
    min_val = 0
    while min_val == 0:
        optimization = optimize.minimize(
            fun=subregion_blockage,
            x0=[0 for i in range(6)],
            method='COBYLA',
            args=args,
            constraints=constraints)
        min_val = optimization.fun
        if min_val == 0:
            intrinsic_free_motions.append(optimization.x)
            constraints.append(
                {
                    'type': 'ineq',
                    'fun': function_for_constraint,
                    'args': (optimization.x, cone_factor)
                }
            )
    mesh_w.intrinsic_free_motions = intrinsic_free_motions
    mesh_w.constraints_input += constraints

    return min_val

def find_triangles_to_ignore(mesh_w):
    normals = mesh_w.mesh.face_normals
    centers = mesh_w.mesh.triangles_center
    res = []
    for triangle_index in range(len(centers)):
        for phi in mesh_w.free_motions_input:
            if contact_blockage(centers[triangle_index], normals[triangle_index], phi):
                res.append(triangle_index)
    return res


def external_free_motions(mesh_w, user_free_motions, cone_factor = 30):
    user_constraints = []
    for phi_free in user_free_motions:
        assert type(phi_free) in [np.array, np.ndarray], f'type of external phi: {type(phi_free)}'
        user_constraints.append({
            'type': 'ineq',
            'fun': function_for_constraint,
            'args': (phi_free, cone_factor)
        })

    mesh_w.free_motions_input = [user_free_motions]
    mesh_w.constraints_input += user_constraints






## 4.5 - Providing Diverse Subset

def clustering_fit(vectors, results):
    n = len(vectors)
    similarity = np.zeros((n, n))
    for i in range(n):
        assert vectors[i] != None, f'vectors[{i}] = None'
        for j in range(i, n):
            assert vectors[j] != None, f'vectors[{j}] = None'
            assert len(vectors[i]) == len(vectors[j]), f'unmatched len: i={len(vectors[i])}   j={len(vectors[i])}'
            similarity[i, j] = similarity[j, i] = (1 - hamming(vectors[i], vectors[j]))
    clustering = SpectralClustering(
        n_clusters=results,
        assign_labels='kmeans',
        random_state=0,
        affinity='precomputed'
    )
    return clustering.fit(similarity)
