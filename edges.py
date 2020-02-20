#!/usr/bin/env python
import pylab as pl
from matplotlib.widgets import RectangleSelector
import numpy as np
import os
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import chan_vese
from skimage.morphology import skeletonize
from skimage.util import invert
from optparse import OptionParser
from skimage import measure, io
import auxfunc as af
from PIL import Image


def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


def main():
    """ """
    version = 0.0
    # Parsing command line options via 'optparse' module
    parser = OptionParser(usage="%prog [OPTION] val", version="%prog " + str(version))
    parser.add_option("-f", "--files", action="callback", callback=get_comma_separated_args, dest="files", type="string", default="", help="Tiff-files to be analysed")
    parser.add_option("-d", "--dir", dest="dir", type="string", default="", help="Tiff-files to be analysed")
    parser.add_option("-v", "--view", action="store_true", dest="view", default=False, help="View centerline + surface")
    parser.add_option("--view_all", action="store_true", dest="view_all", default=False, help="View centerline + surface")
    parser.add_option("--printfile", action="store_true", dest="prnt", default=False, help="print filnames")
    parser.add_option("-t", "--tol", dest="tol", default=0.0, type="float", help="tolerance")
    parser.add_option("-p", "--pix", dest="pix", default='0.9144', type="string", help="(x,y)-pixel distance in [um]")
#    parser.add_option("-p", "--pix", action="callback", callback=get_comma_separated_args, dest="pix", default='0.9114', type="string", help="(x,y)-pixel distance in [um]")
    (options, args) = parser.parse_args()  # options stored here
    default_pix = af.splitter(options.pix, float)
    if len(default_pix) == 1:
        default_pix.append(default_pix[-1])

    cwd = os.getcwd()
    newdir = os.getcwd()
    # Loop through files provided and append to
    if options.dir:
        newdir = "{}/{}".format(cwd, options.dir)
        print("Change directory to:", newdir)
        os.chdir(newdir)
        lastname = newdir.split('/')[-1]
        files = getfiles(newdir, options.prnt)
    elif options.files:
        files = getfile(options.files, options.prnt)
        lastname = "fileset"
    else:
        lastname = newdir.split('/')[-1]
        files = getfiles(newdir, options.prnt)
    print(lastname, "zstack files:", [x[0] for x in files])

    diam_avgs = list()
    for fil, pix in files:
        img = io.imread(fil)
        # Feel free to play around with the parameters to see how they impact the result
        cv = chan_vese(img, mu=0.3, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
                       dt=0.4, init_level_set="checkerboard", extended_output=True)
        chan = np.array(cv[0], dtype=np.int)
        if chan[0, 0] == 1:
            chan = 2+invert(chan)
        skel = skeletonize(chan)
        #skel, chan = properSkel(img, chan)   # skeletonize(chan) #
        xycrop = crop_image(img)
        print("file + pixel-to-micron", fil, pix)
        dist = distance_transform_edt(chan, pix)

        # Crop image
        cimg, cchan, cskel, cdist = crop(xycrop, img, chan, skel, dist)
        diam = 2 * cdist * cskel  # 2: as skeleton provides radius
        if options.view_all:
            f, axes = pl.subplots(2, 2)
            ax = axes.flatten()
            ax[0].imshow(cimg)
            ax[0].set_title('image')
            ax[1].imshow(cchan)  # , alpha=0.5)
            ax[1].imshow(diam, alpha=0.5)
            ax[1].set_title('chan-vese (white=True) + diameter (inside the vessel)')
            ax[2].imshow(cskel, cmap='gray')
            ax[2].set_title('skeleton (white=True)')
            ax[3].imshow(cdist)
            ax[3].set_title('distance map')

            print("cskel:", cskel[0,0], cskel.min(), cskel.max(), cdist.min(), cdist.max())

            pl.show()
        #print("Diam", diam)
        if not options.view:
            f, axes = pl.subplots(1, 2)
            ax = axes.flatten()
            ax[0].imshow(cchan, alpha=0.75)
            ax[0].imshow(cskel, alpha=0.25)
            dax = ax[1].imshow(diam)
            #ax[1].imshow(cimg, alpha=0.5)
            for a in ax:
                a.set_xlabel('$\\mu$m')
                a.set_ylabel('$\\mu$m')
            maxd = int(diam.max())
            tks = int(maxd / 4)
            cbar = f.colorbar(dax, ticks=pl.linspace(0, tks * 4, 4 + 1))
            cbar.set_label("Diameter in [$\\mu$m]")
            pl.tight_layout()
            pl.show()
        diam_vals = diam[np.nonzero(diam)]
        #print(diam_vals)
        print("mean diameter of selected area: {:.3f} um, SD: {:.3f} um (from {})".format(diam_vals.mean(), diam_vals.std(), fil))
        diam_avgs.append([diam_vals.mean(), diam_vals.std(), fil])
    f = open("cap_diam_{}.txt".format(lastname), "w")
    f.write("# AvgDiameter [um], StdDiameter [um], file\n")
    for it in sorted(diam_avgs, key=lambda num: num[2]):
        f.write("{}, {}, {}\n".format(it[0], it[1], it[2]))
    f.close()


def properSkel(img, chan):
    skel = skeletonize(chan)  # (2+invert(chan))
    while True:
        fig, a = pl.subplots()
        a.imshow(img)  # , alpha = 0.3)
        a.imshow(skel, alpha=0.4)
        pl.show()
        #print("skel   ddf ", skel[:3])
        proper = input('Is "skeleton" proper? [y,n]\n')
        if proper == 'y':
            return skel, chan
        elif proper == 'n':
            chan = 2 + invert(chan)  # invert(chan)
            skel = skeletonize(chan)
        else:
            print("You must write 'y' or 'n' fool")


def avg_diameter(diam):
    msk = diam[np.nonzero(diam)]
    #print(np.nonzero(diam), msk)
    return [msk.mean(), msk.std()]


def crop(xycrop, orig, chan, skel, dist):
    ymin, ymax, xmin, xmax = xycrop
    norig = orig[int(ymin): int(ymax), int(xmin): int(xmax)]
    nchan = chan[int(ymin): int(ymax), int(xmin): int(xmax)]
    nskel = skel[int(ymin): int(ymax), int(xmin): int(xmax)]
    ndist = dist[int(ymin): int(ymax), int(xmin): int(xmax)]
    return norig, nchan, nskel, ndist


def crop_image(img):
    f, ax = pl.subplots()
    ax.imshow(img)
    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    pl.connect('key_press_event', toggle_selector)
    pl.show()
    xmin, xmax, ymin, ymax = toggle_selector.RS.extents
    return int(ymin), int(ymax), int(xmin), int(xmax)
    #print("extents", toggle_selector.RS.extents)


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    #print(eclick)
    #print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    #print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (eclick.x, eclick.y, erelease.x, erelease.y))
    return [x1, y1], [x2, y2]


def get_tiff_resolution(fil):
    im = Image.open(fil)
    width, height = im.size
    xres, yres = im.info['resolution']
    im.close()
    return 1.0/float(xres), 1.0/float(yres)


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


def getfile(files, prnt=False):
    fs = list()
    cwd = os.getcwd()
    for fil in files:
        splits = fil.split('/')
        filname = splits[-1]
        dirname = "/".join(splits[:-1])
        os.chdir(dirname)
        if "projection.tif" in filname:
            if prnt:
                print(fil)
            tmppix = get_tiff_resolution(filname)
            print("{}: pixel dimensions, x: {}, y: {} um".format(fil, tmppix[0], tmppix[1]))
            fs.append([fil, tmppix])
    os.chdir(cwd)
    return fs


def getfiles(newdir, prnt=False):
    files = list()
    fls = os.listdir(newdir)
    for fil in fls:
        if "projection.tif" in fil:
            if prnt:
                print(fil)
            tmppix = get_tiff_resolution(fil)
            print("{}: pixel dimensions, x: {}, y: {} um".format(fil, tmppix[0], tmppix[1]))
            files.append([fil, tmppix])
    return files


def get_contours(image, chan):
    fig, axes = pl.subplots(1, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, label='orig')
    ax[1].imshow(chan, label='chan.vese')
    cont = measure.find_contours(chan, 0)
    for a in ax:
        for c in cont:
            a.plot(c[:, 1], c[:, 0], label="contour")
            # a.plot(measure.approximate_polygon(c, tolerance=options.tol)[:,1],
            #       measure.approximate_polygon(c, tolerance=options.tol)[:, 0], label="contour")
        a.legend(frameon=False, loc=0)
    pl.tight_layout()
    pl.show()


if __name__=='__main__':
    main()
