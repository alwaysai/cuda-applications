import cv2
import time
import datetime
import edgeiq
"""
Application use the Nano's CSI camera interface JetsonVideoStream.
Information about JetsonVideoStream can befound in alwaysai documentation at this address:
https://alwaysai.co/docs/edgeiq_api/video_stream.html#edgeiq.edge_tools.JetsonVideoStream
Information on the Object Detection API found in alwaysai documentation at this address:
https://alwaysai.co/docs/edgeiq_api/object_detection.html
You can change the detected object(s) by altering OBJECT list found on line 15.
"""

def main():
    obj_detect = edgeiq.ObjectDetection("alwaysai/ssd_mobilenet_v1_coco_2018_01_28")
    obj_detect.load(engine=edgeiq.Engine.DNN_CUDA, accelerator=edgeiq.Accelerator.NVIDIA)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.JetsonVideoStream(cam=0,rotation=edgeiq.FrameRotation.ROTATE_180) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                frame = edgeiq.resize(frame, width=416)
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                frame = edgeiq.markup_image(
                        frame, results.predictions, show_confidences=False,
                        colors=obj_detect.colors)

                # Print date and time on frame
                current_time_date = str(datetime.datetime.now())
                (h, w) = frame.shape[:2]
                cv2.putText(
                        frame, current_time_date, (10, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                for prediction in results.predictions:
                    text.append("{}: {:2.2f}%".format(
                    prediction.label, prediction.confidence * 100))

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
