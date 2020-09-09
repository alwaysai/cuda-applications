import time
import edgeiq
"""
Use semantic segmentation to determine a class for each pixel of an image.
The classes of objects detected can be changed by selecting different models.
This particular starter application uses a model trained on the Pascal VOC dataset
(http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""


def main():
    semantic_segmentation = edgeiq.SemanticSegmentation("alwaysai/enet")
    semantic_segmentation.load(engine=edgeiq.Engine.DNN_CUDA)

    print("Loaded model:\n{}\n".format(semantic_segmentation.model_id))
    print("Engine: {}".format(semantic_segmentation.engine))
    print("Accelerator: {}\n".format(semantic_segmentation.accelerator))
    print("Labels:\n{}\n".format(semantic_segmentation.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.FileVideoStream('toronto.mp4', play_realtime=True) as video_stream, \
                edgeiq.Streamer() as streamer:  # play_realtime simulates video feed from a camera
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while video_stream.more():
                frame = video_stream.read()
                results = semantic_segmentation.segment_image(frame)

                # Generate text to display on streamer
                text = ["Model: {}".format(semantic_segmentation.model_id)]
                text.append("Inference time: {:1.3f} s".format(results.duration))
                text.append("Legend:")
                text.append(semantic_segmentation.build_legend())

                mask = semantic_segmentation.build_image_mask(results.class_map)
                blended = edgeiq.blend_images(frame, mask, alpha=0.7)

                streamer.send_data(blended, text)

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
