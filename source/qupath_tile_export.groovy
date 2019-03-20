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

def imageData = QPEx.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
def server = imageData.getServer()

def name = server.getShortServerName()
def home_dir = "/Users/james/thyroid/svs/tiles/" + name
QPEx.mkdirs(home_dir)
def path = buildFilePath(home_dir, String.format("Tile_coords_%s.txt", name))
def file = new File(path)
file.text = ''

def downsample = 1.0

for (obj in annotations) {
    if (obj.isAnnotation()) {
        def roi = obj.getROI()

        // Ignore empty or complex annotations
        if ((roi == null) || (roi.getClass() != qupath.lib.roi.RectangleROI)) {
            continue
        }
               
        file << roi.getPolygonPoints() << System.lineSeparator()
        
        def region = RegionRequest.createInstance(server.getPath(), downsample, roi)
        
        String tile_name = String.format('%s_(%.2f,%d,%d)',
            name,
            region.getDownsample(),
            region.getX(),
            region.getY(),
        )
        
        def img = server.readBufferedImage(region)
        /*int w = img.getWidth()
        int h = img.getHeight()

        // Ops for flipping/rotating
        AffineTransform scaleTransform = new AffineTransform();
        scaleTransform.rotate(Math.PI/2, w/2, h/2)
        AffineTransformOp scaleOp = new AffineTransformOp(scaleTransform, AffineTransformOp.TYPE_BILINEAR)
        BufferedImage rot_img = scaleOp.filter(img, null)*/

        def fileImage = new File(home_dir, tile_name + ".jpg")
        // def rotFileImage = new File(home_dir, tile_name + "_rot90.jpg")
        print("Writing image tiles for tile " + tile_name)
        ImageIO.write(img, "JPG", fileImage)
        // ImageIO.write(rot_img, "JPG", rotFileImage)
    }
}
print("Finished!")