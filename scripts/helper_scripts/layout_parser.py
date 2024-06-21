import layoutparser as lp
import cv2

image = cv2.imread("/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg")
image = image[..., ::-1]
    # Convert the image from BGR (cv2 default loading style)
    # to RGB

model = lp.models.Detectron2LayoutModel('lp://HJDataset/mask_rcnn_R_50_FPN_3x/config')
    # Load the deep layout model from the layoutparser API
    # For all the supported model, please check the Model
    # Zoo Page: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html

layout = model.detect(image)
    # Detect the layout of the input image

lp.draw_box(image, layout, box_width=3).show()
    # Show the detected layout of the input image