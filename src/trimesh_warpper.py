import trimesh
# from holder import x,y,z

class Trimesh_wrapper:

    mesh:               trimesh
    convex_hull:        bool
    # symmetry_planes:    list
    minimal_distances:  list
    starting_point:     int # index in surfaces list of starting point


    def calc_minimal_distances(self) -> None:
        lst = []
        val = 0 # To Delete
        n = len (self.mesh.faces )
        for i in range(n):
            # val = use linear computation from different mod
            lst.append(val)
        self.minimal_distances = lst
        pass

    def calc_starting_point(self) -> None:
        lst = self.calc_symmetry_planes()
        starting_point_idx = 0
        if lst:
            pass
        else:
            pass
            # return get_rand_point()
        self.starting_point = starting_point_idx
        

    def calc_symmetry_planes(self) -> list:
        sp = []
        # get simetry planes of object
        return sp

    def pre_process_mesh(self) -> None:
        pass

    def __init__( self, mesh ,
                  convex_hull: bool=False) -> None:
        self.mesh = mesh
        self.convex_hull = convex_hull
        self.pre_process_mesh()
        # self.calc_symmetry_planes()
        # self.calc_starting_point()