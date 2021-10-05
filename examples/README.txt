///////////////////////////////////////////////////////////////////////////////////////////////////////
#   PRELIMINARY OPERATIONS   # 



This environment supports the kernels 'Native' and 'Embree' in the RTXhandler (see 
line 108 in the maven_test example). 
The installation of Embree v3 is more complicate and will be released in the next 
future.


///////////////////////////////////////////////////////////////////////////////////////////////////////
#    EXAMPLES DESCRIPTION    # 

- visualize_pixel_plane_sphere.py
This example shows how to use the ray tracer to compute the intersection points on a simple sphere and visually inspect the output. 
Multiple options are available to the user to decide what to visualize. 

- maven_test.py
This example shows the general pipeline for performing a simple raytracing task

- maven_test_packets.py
Same as maven_test but employing the 'multiple packets' approach to subdivide the sun rays in multiple
sectors. This is done to avoid the segmentation fault that happens when ray tracing with too many rays 


- lro_srp_complete.py
Show the capability of the object-oriented part of the code by computing the SRP on LRO
based on the mission SPICE kernels. 
NOTE: to run this example you need LRO kernels. To download them run
>> python download_lro_kernels.py

- generate_crossection_lut.py
Generate a lookup table of the cross-section given the spacecraft shape

