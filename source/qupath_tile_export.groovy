import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.PathROIToolsAwt
import qupath.lib.scripting.QPEx

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

def imageData = QPEx.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
def server = imageData.getServer()

def name = server.getShortServerName()
def home_dir = "/home/shawarma/thyroid/svs/tiles/" + name
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
        def fileImage = new File(home_dir, tile_name + ".jpg")
        print("Writing image tile " + tile_name)
        ImageIO.write(img, "JPG", fileImage)
    }
}
print("Finished!")