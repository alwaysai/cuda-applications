import time
import edgeiq
import numpy as np
"""

"""


def main():
    semantic_segmentation = edgeiq.SemanticSegmentation("alwaysai/fcn_resnet18_pascal_voc_512x320")
    semantic_segmentation.load(engine=edgeiq.Engine.DNN_CUDA, accelerator=edgeiq.Accelerator.NVIDIA_FP16)

    print("Loaded model:\n{}\n".format(semantic_segmentation.model_id))
    print("Engine: {}".format(semantic_segmentation.engine))
    print("Accelerator: {}\n".format(semantic_segmentation.accelerator))
    print("Labels:\n{}\n".format(semantic_segmentation.labels))

    fps = edgeiq.FPS()

    class_list = ['bottle']

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                frame = edgeiq.resize(frame, width=320)
                results = semantic_segmentation.segment_image(frame)

                object_map = semantic_segmentation.build_object_map(results.class_map, class_list)

                object_mask = semantic_segmentation.build_image_mask(object_map)

                # object_mask[np.where((object_mask==[0,0,0]).all(axis=2))] = [255,255,255]

                # Generate text to display on streamer
                text = ["Model: {}".format(semantic_segmentation.model_id)]
                text.append("Inference time: {:1.3f} s".format(results.duration))
                text.append("Legend:")
                text.append(semantic_segmentation.build_legend())

                blended = edgeiq.blend_images(frame, object_mask, alpha=0.8)

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
