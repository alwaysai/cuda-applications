import time
import edgeiq
import cv2
"""
Real time classifier used to detect water bottles.  You can change the object
you want to detect and count by altering the object label on line 41. Information
for Classification APIs can found here https://alwaysai.co/docs/edgeiq_api/image_classification.html
"""


def main():
    classifier = edgeiq.Classification("alwaysai/googlenet")
    classifier.load(engine=edgeiq.Engine.DNN_CUDA, accelerator=edgeiq.Accelerator.NVIDIA)

    print("Engine: {}".format(classifier.engine))
    print("Accelerator: {}\n".format(classifier.accelerator))
    print("Model:\n{}\n".format(classifier.model_id))
    print("Labels:\n{}\n".format(classifier.labels))

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
                frame = edgeiq.resize(frame, width=224)
                results = classifier.classify_image(frame)

                # Generate text to display on streamer
                text = ["Model: {}".format(classifier.model_id)]
                text.append("Inference time: {:1.3f} s".format(results.duration))

                if results.predictions:
                    print(results.predictions[0].label)
                    if results.predictions[0].label == 'water bottle':
                        # label the frame
                        image_text = "Label: {}, {:.2f}".format(
                                results.predictions[0].label,
                                results.predictions[0].confidence)
                        cv2.putText(
                                frame, image_text, (5, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        for idx, prediction in enumerate(results.predictions[:5]):
                            text.append("{}. label: {}, confidence: {:.5}".format(
                                idx + 1, prediction.label, prediction.confidence))
                    else:
                        text.append("No water bottles detected.")

                else:
                    text.append("No water bottles detected.")

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
