import os
import numpy as np
import trimesh
import os.path as path
import configargparse
from trimesh_handler import *


#--------------------------------------------------------#
#------------------------ PARSER ------------------------#
#--------------------------------------------------------#

def get_parser():
    parser = configargparse.get_argument_parser()

    parser.add_argument('--input', '-in', type=str,
                        help='object name to use as input from inputs folder, input file MUST be in inputs folder')
    parser.add_argument('--input-type', '-t', type=str, default='ply',
                        help='input file type, defauld is ply')
    parser.add_argument('--iterations', '-i', type=int, default=1,
                        help='number of iterations to run, default is a single iteration')
    parser.add_argument('--results', '-r', type=int, default=1,
                        help='number of results to provide, must be maximum as number of iterations')
    parser.add_argument('--constraints', '-c', nargs='+', type=int, required=True,
                        help='a list of constraints chosen from 0 to 5: [0-2]:Rotational, [3-5]:XYZ')
    parser.add_argument('--convex-hull', '-cv', action='store_true', required=False,
                        help='a flag to use input`s convex hull')
    parser.add_argument('--message', '-m', type=str, default='',
                        help='a message to log in autoconnect.txt file in output folder')

    parsed = parser.parse_args()
    parsed.constraints = np.array([[1.0 if i in parsed.constraints else 0.0 for i in range(6)]])
    parsed.clustering = parsed.results < parsed.iterations
    assert parsed.iterations >= 0 and parsed.results >= 0 , 'iteration must be bigger than 0'
    assert parsed.results <= parsed.iterations, 'iterations must be bigger than results'

    return parsed


#------------------------------------------------------#
#------------------------ MAIN ------------------------#
#------------------------------------------------------#

def input_path(filename: str, file_type= 'ply'):
    file_full_path = f'{path.curdir}{path.sep}{"inputs"}{path.sep}{filename}.{file_type}'
    if path.exists(file_full_path):
        return f'{path.curdir}{path.sep}{"inputs"}{path.sep}{filename}.{file_type}'
    else:
        raise ValueError('Given input file doesn't exist, please rerun with an existing file or add the wanted file into inputs directory')



def main():
    vectors = []
    results = []

    args = get_parser()
    mesh_data = Trimesh_handler(
        mesh=trimesh.load(input_path(args.input, args.input_type)),
        constraints=args.constraints,
        convex_hull=args.convex_hull
    )

    # Folders preparing
    c = ''.join([str(i) for i in range(6) if args.constraints[0][i]])
    out_folder = f'outputs{p.sep}{args.input}_{c}{p.sep}'
    all_folder = f'{out_folder}all'
    os.makedirs(all_folder, exist_ok=True)


    for iteration in range(args.iterations):
        print(f'\n Staring computing for {args.input} ')
        print(f'\n\t - Computing iteration: #{iteration}')
        starting_point = calc_starting_point(mesh_data)
        curr_shell  = shell_computation(mesh_data, starting_point)
        results.append(mesh_data.current_holder)
        vectors.append(curr_shell)
        out_path = f'{all_folder}{p.sep}{args.input}_{iteration}.obj'
        mesh_data.current_holder.export(out_path)
        print(f'\n\t\t ~ Exporting: {out_path}')
        
    if args.clustering:
        clustered_folder = f'{out_folder}clustered'
        os.makedirs(clustered_folder, exist_ok=True)

        print(f'\n\t Running Clustering {args.iterations}->{args.results}:')
        clustering = clustering_fit(vectors, args.results)
        clustered = [False for i in range(args.results)]
        
        for i in range(len(vectors)):
            for j in range(i, len(vectors)):
                if clustering.labels_[j] == i:
                    if not clustered[i]:
                        out_path = f'{clustered_folder}{p.sep}{args.input}_{i}.obj'
                        results[i].export(out_path)
                        print(f'\t ~ Export: {out_path}')
                        clustered[i] = True

    print(f'\nFinished running Auto-Connect for {args.input} \n')
    with open(f'{out_folder}Run_Results.txt', 'w') as info:
        info.write(args.info)


if __name__ == "__main__":
    main()



