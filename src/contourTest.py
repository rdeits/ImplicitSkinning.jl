from director import vtkNumpy as vnp
import vtk
import numpy as np
import lcm
import drake as lcmdrake
import threading
import ddapp.lcmUtils as lcmUtils


def numpyToVtkImage(data, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):

    assert len(data.shape) == 3

    image = vtk.vtkImageData()
    image.SetWholeExtent(0, data.shape[0]-1, 0, data.shape[1]-1, 0, data.shape[2]-1)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetExtent(image.GetWholeExtent())
    image.SetNumberOfScalarComponents(1)
    image.SetScalarType(vtk.VTK_DOUBLE)
    image.AllocateScalars()

    d = vnp.getNumpyFromVtk(image, 'ImageScalars')
    np.copyto(d, data.flatten())
    return image


def applyContourFilter(image, value):

    f = vtk.vtkContourFilter()
    f.SetInput(image)
    f.SetNumberOfContours(1)
    f.SetValue(0, value)
    f.Update()
    return f.GetOutput()


##########################

import director.visualization as vis
from director.consoleapp import ConsoleApp


def main():

    app = ConsoleApp()
    view = app.createView()

    vis_item = {"current": None}
    def handle_data(msg):
        # msg = lcmdrake.lcmt_viewer_geometry_data.decode(msg_data)
        side_length = int(round(np.power(len(msg.float_data), 1./3)))
        data = np.reshape(msg.float_data, (side_length, side_length, side_length))

        # convert to a vtkImageData
        image = numpyToVtkImage(data)

        # compute iso contour as value 0.5
        polyData = applyContourFilter(image, 0.0)

        # show data
        if vis_item["current"] is not None:
            vis_item["current"].removeFromAllViews()

        vis_item["current"] = vis.showPolyData(polyData, 'contour')
        view.resetCamera()

    lcmUtils.addSubscriber("FIELD_DATA", lcmdrake.lcmt_viewer_geometry_data, handle_data)

    # start app
    view.show()
    view.resetCamera()
    app.start()



    # # create a block of random numpy data
    # data = np.random.randn(50,50,50)



if __name__ == '__main__':
    main()

