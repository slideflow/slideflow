# Please note that we can make no guarantees regarding the use of this patch
# and provide no warranties, explicit or implied.
# It has been tested for pixman version 0.38 in Ubuntu 18.04 and 20.04
#
# Preparation:
# 1. ensure a deb-src source is uncommented
# sudo nano /etc/apt/sources.list
if grep -q ^deb-src /etc/apt/sources.list*; then
    echo "deb-src found. proceeding..."
else
    echo "Ensure a deb-src is available in your sources.list"
    exit 1
fi

# 2. ensure dch is available for patching stuff
if ! command -v dch &> /dev/null
then
    echo "COMMAND dch not be found, trying to install devscripts..."
    apt install devscripts
else
	echo "dch found. proceeding..."
fi

echo "Creating pixman dir within /opt/ ..."
cd /opt/
mkdir pixman
chown $(whoami):$(whoami) pixman
cd pixman

echo "Getting sources..."
apt install libpixman-1-0=0.38.4-0ubuntu1
apt-get source libpixman-1-0=0.38.4-0ubuntu1
cd ..
chown -R $(whoami):$(whoami) pixman
cd pixman/pixman-0.38.4

echo "Trying to patch..."
cd pixman
patch --verbose --ignore-whitespace -b <<EOF
--- pixman-bits-image.c.old 2021-08-03 10:51:55.099218082 +0200
+++ pixman-bits-image.c 2021-08-03 10:55:40.979818850 +0200
@@ -432,29 +432,38 @@

     case PIXMAN_FILTER_CONVOLUTION:
     if (wide)
+    {
         bits_image_fetch_pixel_convolution (image, x, y,
                         get_pixel, out,
                         accum_float,
                         reduce_float);
+    }
     else
+    {
         bits_image_fetch_pixel_convolution (image, x, y,
                         get_pixel, out,
                         accum_32, reduce_32);
+    }
     break;

     case PIXMAN_FILTER_SEPARABLE_CONVOLUTION:
     if (wide)
+    {
         bits_image_fetch_pixel_separable_convolution (image, x, y,
                               get_pixel, out,
                               accum_float,
                               reduce_float);
+    }
     else
+    {
         bits_image_fetch_pixel_separable_convolution (image, x, y,
                               get_pixel, out,
                               accum_32, reduce_32);
+        }
         break;

     default:
+        assert (0);
         break;
     }
 }

EOF

patch -b <<EOF
--- pixman-inlines.h.old        2021-08-03 10:55:54.011621896 +0200
+++ pixman-inlines.h    2021-08-03 10:56:34.175014448 +0200
@@ -231,7 +231,7 @@
     argb_t r;

     distxy = distx * disty;
-    distxiy = distx - (1.f - distxy);
+    distxiy = distx * (1.f - distxy);
     distixy = (1.f - distx) * disty;
     distixiy = (1.f - distx) * (1.f - disty);
EOF
# Alternatively:
# nano pixman/pixman-inlines.h
# ... and ...
# nano pixman/pixman-bits-image.c
# to change files as in this commit:
# https://gitlab.freedesktop.org/pixman/pixman/commit/8256c235d9b3854d039242356905eca854a890ba

cd ..
dch -n "user-patch of commit 8256c235"

echo "Trying to build & install patched pixman..."
apt -y build-dep pixman
debuild
cd ..
dpkg -i libpixman-1-0_0.38.4-0ubuntu1.1_amd64.deb
