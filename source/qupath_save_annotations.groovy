import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.PathROIToolsAwt
import qupath.lib.scripting.QPEx

def imageData = QPEx.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
def server = imageData.getServer()

def name = server.getShortServerName()
def home_dir = "/home/shawarma/thyroid/svs"
QPEx.mkdirs(home_dir)
def path = buildFilePath(home_dir, String.format("%s.qptxt", name))
def file = new File(path)
file.text = ''

def downsample = 1.0

for (obj in annotations) {
    if (obj.isAnnotation()) {
        def roi = obj.getROI()
        // Ignore empty annotations
        if (roi == null) {
            continue
        }
        print("Name: " + obj.name)
        points = roi.getPolygonPoints()
        file << "Name: " + obj.name << System.lineSeparator()
        for (point in points) {
            p_x = point.getX()
            p_y = point.getY()
            point_string = p_x + ", " + p_y
            file << point_string << System.lineSeparator()
        }
        file << "end" << System.lineSeparator()
    }
}
print("Finished!")