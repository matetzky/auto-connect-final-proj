import trimesh

class Trimesh_handler:

    mesh:               trimesh
    convex_hull:        bool
    minimal_distances:  list
    starting_point:     int 


    def calc_minimal_distances(self):
        lst = []
        val = 0 
        n = len (self.mesh.faces )
        for i in range(n):
            lst.append(val)
        self.minimal_distances = lst
        pass

    def calc_starting_point(self):
        lst = self.calc_symmetry_planes()
        starting_point_idx = 0
        if lst:
            pass
        else:
            pass
        self.starting_point = starting_point_idx
        

    def calc_symmetry_planes(self):
        sp = []

        return sp

    def pre_process_mesh(self):
        pass

    def __init__( self, mesh ,
                  convex_hull: bool=False) -> None:
        self.mesh = mesh
        self.convex_hull = convex_hull
        self.pre_process_mesh()
