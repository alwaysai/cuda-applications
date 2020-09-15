import time
import edgeiq
import cv2
import numpy as np
import os
"""
Instance segmenataiom used to count unique instances of water bottles
"""


def main():
    print("get labels")
    labelsPath = "models/object_detection_classes_coco.txt"
    LABELS = open(labelsPath).read().strip().split("\n")

    print("create colors")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    	dtype="uint8")

    # set paths to the Mask R-CNN model and configuration
    weightsPath = "models/frozen_inference_graph.pb"
    configPath = "models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

    print("load mask rcnn model and set CUDA backend")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("mask rcnn loaded sucessfully")

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()

                blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
                net.setInput(blob)
                (boxes, masks) = net.forward(["detection_out_final",
                    "detection_masks"])
                    # loop over the number of detected objects
                counter = 0
                for i in range(0, boxes.shape[2]):
                    # extract the class ID of the detection along with the
                    # confidence (i.e., probability) associated with the
                    # prediction
                    classID = int(boxes[0, 0, i, 1])
                    confidence = boxes[0, 0, i, 2]
                    if confidence > 0.5 and LABELS[classID] == "bottle":
                        # scale the bounding box coordinates back relative to the
                        # size of the frame and then compute the width and the
                        # height of the bounding box
                        (H, W) = frame.shape[:2]
                        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")
                        boxW = endX - startX
                        boxH = endY - startY
                        # extract the pixel-wise segmentation for the object,
                        # resize the mask such that it's the same dimensions of
                        # the bounding box, and then finally threshold to create
                        # a *binary* mask
                        mask = masks[i, classID]
                        mask = cv2.resize(mask, (boxW, boxH),
                            interpolation=cv2.INTER_CUBIC)
                        mask = (mask > 0.3)
                        # extract the ROI of the image
                        roi = frame[startY:endY, startX:endX][mask]
                        # grab the color used to visualize this particular class,
            			# then create a transparent overlay by blending the color
            			# with the ROI
                        counter += 1
                        color = COLORS[classID + counter]
                        blended = ((0.7 * color) + (0.4 * roi)).astype("uint8")
                        # store the blended ROI in the original frame
                        frame[startY:endY, startX:endX][mask] = blended
                        # draw the bounding box of the instance on the frame
                        # counter += 1
                        color = [int(c) for c in color]
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                            color, 2)
                        text = "{}: number {} ".format(LABELS[classID], counter)
                        cv2.putText(frame, text, (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        text = "Instance Segmenataiom"
                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
