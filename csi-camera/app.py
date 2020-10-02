import cv2
import time
import datetime
import edgeiq
"""
Application detects bottles and counts them.  Application use the Nano's CSI
camera interface JetsonVideoStream.  Information about JetsonVideoStream can be
found in alwaysai documentation at this address:
https://alwaysai.co/docs/edgeiq_api/video_stream.html#edgeiq.edge_tools.JetsonVideoStream
"""

OBJECT = ["bottle"]


def main():
    obj_detect = edgeiq.ObjectDetection("alwaysai/yolo_v3_tiny")
    obj_detect.load(engine=edgeiq.Engine.DNN_CUDA, accelerator=edgeiq.Accelerator.NVIDIA)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))
    print("Labels:\n{}\n".format(obj_detect.labels))
    print("Detecting:\n{}\n".format(OBJECT))

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
                predictions = edgeiq.filter_predictions_by_label(
                        results.predictions, OBJECT)
                frame = edgeiq.markup_image(
                        frame, predictions, show_confidences=False,
                        colors=obj_detect.colors)

                # Print date and time on frame
                current_time_date = str(datetime.datetime.now())
                (h, w) = frame.shape[:2]
                cv2.putText(
                        frame, current_time_date, (10, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Count OBJECT
                counter = {obj: 0 for obj in OBJECT}

                for prediction in predictions:
                    # increment the counter of the detected object
                    counter[prediction.label] += 1

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Object counts:")

                for label, count in counter.items():
                    text.append("{}: {}".format(label, count))

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
