// Tile ROI coordinate exporter
// for QuPath 0.1.X
// Written by James Dolezal

import qupath.lib.gui.QuPathGUI
import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.PathROIToolsAwt
import qupath.lib.scripting.QPEx

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.geom.AffineTransform
import java.awt.image.AffineTransformOp
import java.awt.image.DataBufferByte
import java.nio.file.Paths

def tessellate = false
def augment = false
def extract_um = 280
def tile_px = 512

def qupath = QuPathGUI.getInstance()
def project = qupath.getProject()
if (project == null) {
    print("ERROR: No project open, please create a project and try again.")
    return
}
def root_dir = project.getBaseDirectory()

setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049 ", "Stain 2" : "Eosin", "Values 2" : "0.2159 0.8012 0.5581 ", "Background" : " 255 255 255 "}');
if (tessellate) {
    selectAnnotations();
    runPlugin('qupath.lib.algorithms.TilerPlugin', String.format('{"tileSizeMicrons": %d,  "trimToROI": false,  "makeAnnotations": true,  "removeParentAnnotation": false}', extract_um));
}

def imageData = QPEx.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
def server = imageData.getServer()

def name = server.getShortServerName()
def tile_dir = new File(root_dir, name)
def roi_dir = new File(root_dir, "ROI")
QPEx.mkdirs(roi_dir.getAbsolutePath())
def tile_file = new File(roi_dir, String.format("Tile_coords_%s.txt", name))
if (tessellate) {
    QPEx.mkdirs(tile_dir.getAbsolutePath())
    tile_file.text = ''
}
def roi_file = new File(roi_dir, String.format("%s.csv", name))
roi_file.text = ''
roi_file << "ROI_Name,X_base,Y_base" << System.lineSeparator()

def roi_count = 0

for (obj in annotations) {
    if (obj.isAnnotation()) {
        def roi = obj.getROI()
        def roi_name = "ROI_" + roi_count
        roi_count = roi_count + 1

        // Ignore empty annotations
        if (roi == null) {
            continue
        }

        print("Working on " + roi_name)
        points = roi.getPolygonPoints()
        for (point in points) {
            p_x = point.getX()
            p_y = point.getY()
            roi_file << roi_name + "," + p_x + "," + p_y << System.lineSeparator()
        }
    }
}
print("Finished.")