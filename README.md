# Auto-Connect Final Project
This project is aiming to implement the first part of the auto-connect algorithm as explained in the following article: 
https://koyama.xyz/project/AutoConnect/autoconnect.pdf &nbsp;&nbsp;&nbsp;

# Description
The article is describing an algorithm to produce a 3D printable holder for a free form object with another connecting target object.
In this project we implemented only phase 1 of the article and produce an holder for a free form without any target object ot be connected too. 

1. We analyze the input mesh.
2. Generates a costume shell based on the shape and constrains given with the input object.
3. Iterate over a variety of options until getting a divers collection of holder matching the requirements for the user to choose from. 


# How to run: 
1. Clone this repo.
2. Install all the requirement `pip install -r requirements.txt`
3. Download and install https://git-lfs.github.com/
4. place your object file in Inputs directory, we support ply files.
5. Using the terminal change directory into src dir: `cd src`
5. Run command: `main.py` argument should be provided followed by this description:

* <b>`--input, -in`</b> &nbsp; <span style="color:red"> required </span>  
   object name to use as input from inputs folder, input file MUST be in inputs folder
* <b>`--input-type, -t`</b> &nbsp; default= `ply`  
   input file type, default is ply   
* <b>`--iterations, -i`</b>  
   number of iterations to run, default is a single iteration
* <b>`--results, -r`</b>  
    number of results to provide, must be maximum as number of iterations
* <b>`--constraints, -c`</b> &nbsp; <span style="color:red"> required </span>   
    a list of constraints chosen from 0 to 5:  
    [0-2] : Rotational  
    [3-5] : XYZ (default: None)
* <b>`--convex_hull, -cv`</b>  
    a flag to use input`s convex hull, False by default 


# Products
The output objects are not printable by default and trying to pint them naively will fail, the outputted object width is too thin.
Using fusion 360 we were able to give the boundaries of the output object a printable width.
 We will present in class printed holder for some chosen objects. 


# Supporting article: 
https://koyama.xyz/project/AutoConnect/supplemental.pdf &nbsp;