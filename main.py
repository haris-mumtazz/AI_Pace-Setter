import jetson.inference
import jetson.utils
import warnings

# Turn off GStreamer warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gi.repository.Gst")

filename = 'name_list.txt'

with open(filename, 'r') as file:
    lines = file.readlines()

namelist = [line.strip() for line in lines]
print(namelist)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.7)
display = jetson.utils.videoOutput("display://0")
# usb camera
# camera =  jetson.utils.videoSource("/dev/video0") 
# csi camera:  csi://0
camera =  jetson.utils.videoSource("csi://0")
font = jetson.utils.cudaFont(size=22)
warning = jetson.utils.loadImage("media/warning0.png")
left = jetson.utils.loadImage("media/left.png")
right = jetson.utils.loadImage("media/right.png")

while display.IsStreaming():
    img = camera.Capture()
    jetson.utils.cudaDrawLine(img, (img.width * 0.35, img.height), (img.width * 0.35, 0), (255, 0, 0), 1)
    jetson.utils.cudaDrawLine(img, (img.width * 0.65, img.height), (img.width * 0.65, 0), (255, 0, 0), 1)
    
    # Crop the image into left and right halves
    crop_l = (0, 0, img.width * 0.35, img.height)
    crop_r = (img.width * 0.65, 0, img.width, img.height)
    crop_m = (img.width * 0.35, 0, img.width * 0.65, img.height)

    # allocate the output image, with the cropped size
    imgL = jetson.utils.cudaAllocMapped(width=img.width * 0.35,
                                        height=img.height,
                                        format=img.format)
    imgR = jetson.utils.cudaAllocMapped(width=img.width * 0.35,
                                        height=img.height,
                                        format=img.format)
    imgM = jetson.utils.cudaAllocMapped(width=img.width * 0.3,
                                        height=img.height,
                                        format=img.format)
    jetson.utils.cudaCrop(img, imgL, crop_l)
    jetson.utils.cudaCrop(img, imgR, crop_r)
    jetson.utils.cudaCrop(img, imgM, crop_m)

    displayImage = jetson.utils.cudaAllocMapped(width=img.width,
                                                height=img.height,
                                                format=img.format)

    # Left
    detections_l = net.Detect(imgL)
    jetson.utils.cudaOverlay(imgL, displayImage, 0, 0)
    left_count = 0
    for detection_ in detections_l:
        class_name = net.GetClassDesc(detection_.ClassID)
        print(class_name)
        if class_name in namelist:
            print("!!left")
            left_count += 1
            jetson.utils.cudaOverlay(warning, displayImage, img.width * 0.15, img.height - 100)
            jetson.utils.cudaOverlay(right, displayImage, img.width * 0.15, 50)

    # Right
    detections_r = net.Detect(imgR)
    jetson.utils.cudaOverlay(imgR, displayImage, img.width * 0.65, 0)
    right_count = 0
    for detection_ in detections_r:
        class_name = net.GetClassDesc(detection_.ClassID)
        print(class_name)
        if class_name in namelist:
            print("!!right")
            right_count += 1
            jetson.utils.cudaOverlay(warning, displayImage, img.width * 0.8, img.height - 100)
            jetson.utils.cudaOverlay(left, displayImage, img.width * 0.8, 50)
            
    # Front
    detections_m = net.Detect(imgM)
    jetson.utils.cudaOverlay(imgM, displayImage, img.width * 0.35, 0)
    middle_count = 0
    for detection_ in detections_m:
        class_name = net.GetClassDesc(detection_.ClassID)
        print(class_name)
        if class_name in namelist:
            print("!!ahead")
            middle_count += 1
            jetson.utils.cudaOverlay(warning, displayImage, img.width * 0.45, img.height - 100)
            jetson.utils.cudaOverlay(left, displayImage, img.width * 0.45 - 5, 50)
            jetson.utils.cudaOverlay(right, displayImage, img.width * 0.45 + 70, 50)

    # Display the result
    font.OverlayText(displayImage, img.width, img.height, "Number of Objects: {}".format(left_count), 20, 10, (255, 0, 0), (0, 0, 0))
    font.OverlayText(displayImage, img.width, img.height, "Number of Objects: {}".format(middle_count), img.width // 2 - 150, 10, (255, 0, 0), (0, 0, 0))
    font.OverlayText(displayImage, img.width, img.height, "Number of Objects: {}".format(right_count), img.width // 2 + 250, 10, (255, 0, 0), (0, 0, 0))
    display.Render(displayImage)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

display.Close()
