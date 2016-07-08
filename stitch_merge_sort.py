# -*- coding: utf-8 -*-
"""

Stitch, channel merge and sort confocal image slices using Leica FP8 Mark and Find feature
------------------------------------------------------------------------------------------


Given a list of images at different x,y positions and a number of clusters, perform KMeans
clustering to figure out which image is in which cluster, then for each cluster stitch
the images together using the provided meta data, combine red and green channels and output
images sorted first by the time taken and then by the z height.

I gave up trying to output image sequences


Required Python libraries: SciKit Learn, PIL, numpy, ElementTree

Usage: python stitch_merge_sort.py <input_mark_and_find_folder> <number_of_colonies> <leica_maf_xml> <output_folder>

Input: 1. input_mark_and_find_folder: folder named like Mark_and_Find_001 from Leica Mark and Find feature
       2. number_of_colonies: number of


Output: A folder structure with images:

        colony_00: (each colony separate)
         |
         +---- time_000 (each time point separate)
                |
                +---- 000.tif (each z position separate)
                +---- 001.tif
                +---- ...


I use the output from this script to create 3D projections in ImageJ 3D viewer.


* Download precompiled .dll on PC, then drag and drop it to C:\Python27\Lib\site-packages\skimage\io\_plugins.


Created on Mon Jun 27 16:13:42 2016

@author: Igor
"""

# Remove all variables
# import sys
# sys.modules[__name__].__dict__.clear()

from PIL import Image
from math import ceil
from operator import itemgetter
from sklearn.cluster import KMeans

import os, sys, string
import numpy as np
import xml.etree.ElementTree as etree


def leica_image_filename (xyid, time, z, channel='00', xypad=3, tpad=3, zpad=2, format='tif'):
    """ Output the filename for xy position xyid, time z and specific channel. We can adjust
        zero padding for each xy position (xypad), time (tpad) or z position (zpad).
    """
    return 'Position'+str(xyid)+str(xyid).zfill(xypad)+'/Position'+str(xyid)+str(xyid).zfill(xypad)+ \
        '_t'+str(time).zfill(tpad)+'_z'+str(z).zfill(zpad)+'_ch'+str(channel)+'.'+format


def xy_from_xml (xml_filename='xypositions.maf'):
    """ Extract x,y stage positions from Leica XML file (.maf) and output
        values in micro meters.
    """
    tree = etree.parse(xml_filename)
    root = tree.getroot()
    return [[float(child.attrib['StageYPos'])*1e6, -float(child.attrib['StageXPos'])*1e6] for child in root]


def main ():

    # Input variables (required):
    if len(sys.argv) < 5:
        print __doc__
        exit()

    # Folder with images from Mark and Find
    image_folder = sys.argv[1]
    output_folder = sys.argv[4]

    # Specify number of clusters (grouped xy positions) to find using KMeans
    no_clusters = sys.argv[2]

    # XML file with meta data
    xml_filename = sys.argv[3]

    # Input variables (optional; good defaults assumed)

    # Channel indices in the order R,G,B (we usually have R and G)
    channels = ['01', '00']

    # These are good defaults, stitch in binary mode then later merge channels
    stitched_mode = 'L'

    # Extract conversion from pixels to um from the first image
    im = Image.open(image_folder+leica_image_filename(1, 0, 0))
    im_size_um = [round(25400*px/dpi,1) for dpi,px in zip(im.info['dpi'], im.size)]
    xy_px_to_um = [25400/dpi for dpi in im.info['dpi']]
    xy_um_to_px = [1/x for x in xy_px_to_um]
    im.close()
    del im

    # Use string translation table to remove characters from time points, z points etc.
    # This allows using numbers padded with arbitrary number of zeros in file names.
    allchars = ''.join(chr(i) for i in xrange(256))
    identity = string.maketrans('', '')
    nondigits = allchars.translate(identity, string.digits)

    # Extract xy positions
    xy_positions_from_id = xy_from_xml(xml_filename)

    # Figure out the image clusters using Kmeans and known number of clusters
    # actual image ID is going to be just array index + 1
    image_groups = KMeans(no_clusters, random_state=13).fit_predict(xy_positions_from_id)

    for group in set(image_groups): # Do this for each image group

        # Extract time points
        ts = set([int(filename.split('_')[1].translate(identity, nondigits))
            for filename in os.listdir(image_folder+os.listdir(image_folder)[0]) if filename != 'MetaData'])

        # Extract z positions
        zs = set([int(filename.split('_')[2].translate(identity, nondigits))
            for filename in os.listdir(image_folder+os.listdir(image_folder)[0]) if filename != 'MetaData'])

        # Extract only imageid and their xy coordinates for this specific group
        xys = [(i+1, xy[0], xy[1]) for i, (gg, xy) in enumerate(zip(image_groups, xy_positions_from_id)) if gg==group]

        # Find x and y min to calculate relative coordiantes
        xy_min = [min(xys, key=itemgetter(1))[1], min(xys, key=itemgetter(2))[2]]

        # Convert xy position to relative coordinates
        xys = [(im_id, x-xy_min[0], y-xy_min[1]) for im_id, x, y in xys]
        xys_px = {im_id: (int(x*xy_um_to_px[0]), int(y*xy_um_to_px[1])) for im_id, x, y in xys}

        # Now find maximum to calculate the size for the stitched image
        xy_max = [max(xys, key=itemgetter(1))[1], max(xys, key=itemgetter(2))[2]]
        stitched_size = [int(ceil((xy_max[i]+im_size_um[i])*xy_um_to_px[i])) for i in range(len(xy_max))]

        for t in ts: # Do this for each t (iterate over z)

            print 'colony: '+str(group).zfill(2)+', time point: '+str(t).zfill(3)+' stitching... '
            sys.stdout.flush()

            # Create a big numpy 3D array to hold all frames
            # 3 is for RGB
            for z in zs: # Iterate through different z slices

                stitched_im_channels = []
                for ch_id, ch in enumerate(channels): # Iterate through different color channels

                    # Generate a blank image
                    stitched_im = Image.new(stitched_mode, stitched_size)

                    # Now for every image from the image_list find their pixel coordinates in this new image
                    for im_id, xy_px in xys_px.items():
                        im_filename = image_folder+leica_image_filename(im_id, t, z, channel=ch)
                        stitched_im.paste(Image.open(im_filename), xy_px)

                    output_image_folder = output_folder+'colony_'+str(group).zfill(2)+'/time_'+str(t).zfill(3)+'/'

                    # Create folders if they don't exist
                    if not os.path.exists(output_image_folder):
                        os.makedirs(output_image_folder)

                    stitched_im_channels.append(stitched_im)

                # Add blank blue channel, then merge
                stitched_im_channels.append(Image.new(stitched_mode, stitched_size))
                stitched_im_rgb = Image.merge('RGB', stitched_im_channels)

                # There some bug in Image.save() if we use Image.merge() so I convert the image to np array
                # then back into the Image. Then it works fine and doesn't throw some cryptic error.
                stitched_im_array = np.array(stitched_im_rgb)
                stitched_im_rgb_new = Image.fromarray(stitched_im_array, 'RGB')
                stitched_im_rgb_new.info['compression'] = 'tiff_lzw'
                stitched_im_rgb_new.save(output_image_folder+str(z).zfill(3)+'.tif')


if __name__ == '__main__':
    main()