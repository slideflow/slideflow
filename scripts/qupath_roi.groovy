// Tile ROI coordinate exporter
// for QuPath 0.2.X - 0.3.X
// Written by James Dolezal

import qupath.lib.gui.QuPathGUI
import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.geom.AffineTransform
import java.awt.image.AffineTransformOp
import java.awt.image.DataBufferByte
import java.nio.file.Paths

def qupath = QuPathGUI.getInstance()
def project = qupath.getProject()
def server = getCurrentServer()
if (project == null) {
    print("ERROR: No project open, please create a project and try again.")
    return
}
def root_dir = project.getBaseDirectory()

print("Project images:")
print(project.getImageList())

for (entry in project.getImageList()) {
    
    def hierarchy = entry.readHierarchy()
    def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
    def name = GeneralTools.getNameWithoutExtension(entry.getImageName())
    def roi_dir = new File(root_dir, "ROI")
    QPEx.mkdirs(roi_dir.getAbsolutePath())

    // Count total ROIs first (in case this is a duplicate, but empty/non-annotated, slide
    def total_roi = 0
    for (obj in annotations) {
        if (obj.isAnnotation()) {
            total_roi = total_roi + 1
        }
    }
    
    if (total_roi == 0) {
        print(String.format("Skipping image with no annotations: %s", name))
        continue
    } else {
        // Create annotations file
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
                
                // Export annotation
                points = roi.getAllPoints()
                for (point in points) {
                    p_x = point.getX() + server.boundsX
                    p_y = point.getY() + server.boundsY
                    roi_file << roi_name + "," + p_x + "," + p_y << System.lineSeparator()
                }
            }
        }
        print(String.format("Finished %s", name))
    }
 }