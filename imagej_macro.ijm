/*
	ImageJ macro script that loads stitched images from Python script
	then makes a 3D projection and saves 3D projection snapshot for
	each time point.

	Author: Igor Segota
	Date: 2016-07-06
*/

// directories
work_dir = "\\Temp\\confocal_2016-06-27_stitched\\colony_00\\";
output_dir = "\\Temp\\confocal_2016-06-27_stitched\\colony_00_3d_top_crop\\";

File.makeDirectory(output_dir)

/* cropping area for colony 02 (2016-06-27)
crop_x = 564
crop_y = 886
crop_w = 1112
crop_h = 896
*/

/* cropping area for colony 01 (2016-06-27)
crop_x = 32
crop_y = 306
crop_w = 1932
crop_h = 1742
*/

// cropping area for colony 00 (2016-06-27)
crop_x = 286;
crop_y = 392;
crop_w = 1866;
crop_h = 1598;

// extract time directories
time_dirs = getFileList(work_dir);

// iterate through time points
for (i=1; i<=time_dirs.length; i=i+1) {
	// open image stack
	time_string = substring(time_dirs[i-1], 0, 8);
	run("Image Sequence...", "open="+work_dir+time_string+"\\000.png sort");
	res_x = getWidth();
	res_y = getHeight();

	// run 3D viewer, take a snapshot. Actually open it in advance
	run("3D Viewer");
	call("ij3d.ImageJ3DViewer.setCoordinateSystem", "false");
	call("ij3d.ImageJ3DViewer.add", time_string, "None", time_string, "0", "true", "true", "false", "2", "0");
	call("ij3d.ImageJ3DViewer.snapshot", res_x, res_y);
	selectWindow("Snapshot");
	makeRectangle(crop_x, crop_y, crop_w, crop_h);
	run("Crop");
	saveAs("PNG", output_dir+time_string+".png");
	close();

	// delete projection from 3D viewer
	call("ij3d.ImageJ3DViewer.select", time_string);
	call("ij3d.ImageJ3DViewer.delete");

	// close image stack
	selectWindow(time_string);
	close();
	
	// Close 3D viewer
	call("ij3d.ImageJ3DViewer.close");
	// perform garbage collection
	call("java.lang.System.gc");

	// wait
	wait(500);
}

