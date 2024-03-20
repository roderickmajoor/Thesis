import layoutparser as lp
import cv2

image = cv2.imread("data/Wisselbank voorbeelddata/WBMB00008000060.jpeg")
image = image[..., ::-1]
    # Convert the image from BGR (cv2 default loading style)
    # to RGB

model = lp.Detectron2LayoutModel("/home/roderickmajoor/.torch/iopath_cache/s/f3b12qc4hc0yh4m/config.yml?dl=1",
                                 "/home/roderickmajoor/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth",
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    # Load the deep layout model from the layoutparser API
    # For all the supported model, please check the Model
    # Zoo Page: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html

layout = model.detect(image)
    # Detect the layout of the input image

lp.draw_box(image, layout, box_width=3).show()
    # Show the detected layout of the input image