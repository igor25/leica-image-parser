# Cluster images, combine and make animations from arbitrary Leica confocal slices

![colony2](animations/colony_00_ani.gif)
![colony2](animations/colony_02_ani.gif)

We start by knowing the number of colonies we're recording and having:

1. ``Mark_and_Find_XXX/`` folder
2. ``xypositions.maf`` XML file with all xy positions

## Cluster, stitch, merge, sort images

Python script [stitch_merge_sort.py](stitch_merge_sort.py) stitches xy positions, combines red and green color channels, sort images by z slices, then sort by time points. Image groups corresponding to pictures for a each colony to be stitched are identified using K-means clustering where we specify k in advance since we know how many colonies we're recording. xyz coordinates are extracted from the XML file. Alternatively, they're available in the ``MetaData`` folder for each image in ``Mark_and_Find_XXX`` folder.

Run it for example:

```
python stitch_merge_sort.py /Temp/Mark_and_Find_001/ 3 xypositions.maf /Temp/confocal_2016-06-27_stitched/
```

This creates a folder ``/Temp/confocal_2016-06-27_stitched/`` and within it a separate folder for each colony (image cluster) and within each separate folder for each time point. Each time point folder contains one image for each z slice.

## Build 3D projections

Next, ImageJ script [imagej_macro.ijm](imagej_macro.ijm) uses [3D viewer](http://3dviewer.neurofly.de/) to make 3D projections from these images and then saves a specific projection (for now just the top). In order for JVM not to keep eating up more and more RAM, configure ImageJ with a G1 garbage collector, i.e. add ``-XX:+UseG1GC`` to ``C:\Program Files\ImageJ\ImageJ.cfg``.

To do: write the code to be able to run from a batch script from command line.

## Making animations

Upload a series of time point images from ImageJ to [gifmaker.me](http://gifmaker.me) then resize if needed.
