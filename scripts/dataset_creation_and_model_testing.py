from scripts.camera_dataset_creator.metamorfic_dataset_generator import MetamorphicDatasetCreator
from yolov8.yolov8_main import Yolov8ImageDetector
from imagenet.imagenet_main import ImageNetImageDetector

########################################################################################################################
#                                                                                                                      #
#   This script is used to create the metamorphic transformed dataset.                                                 #
#   The dataset is created from a video file.                                                                          #
#   COMMENT IF YOU WANT TEST MODELS ONLY                                                                               #
#                                                                                                                      #
########################################################################################################################
# input_video_path = './camera_dataset_creator/Mask.mp4'
# frames_output_folder = './camera_dataset_creator/movie_frames'
# augmented_images_folder = './metamorphic_transformed_dataset'
#
# metamorphic_creator = MetamorphicDatasetCreator(input_video_path, frames_output_folder, augmented_images_folder)
# # metamorphic_creator.trim_frames()
# metamorphic_creator.augment_images(rotation_range=[0, 10, 20],
#                                    zoom_range=[1.0, 1.5],
#                                    illumination_changes=[20, 40],
#                                    movement_offsets=[5, 10],
#                                    blurring_kernels=[3, 5],
#                                    scaling_factors=[0.75, 1.0])



########################################################################################################################
#                                                                                                                      #
#   This script is used to test the detection models on the metamorphic transformed dataset.                           #
#   The models are tested on the same dataset, so the results can be compared.                                         #
#                                                                                                                      #
########################################################################################################################
# Paths for YOLOv8
yolov8_weights_path = './nn_models/yolov8_best.pt'
yolov8_source_dir = './metamorphic_transformed_dataset'
yolov8_output_dir = './metamorphic_test_detection/yolov8'

# Paths for ImageNet
imagenet_model_path = './nn_models/imagenet.h5'
imagenet_source_dir = './metamorphic_transformed_dataset'
imagenet_output_dir = './metamorphic_test_detection/imagenet'

# Load models
# Creating instances of the detectors with the specified paths
yolov8_model = Yolov8ImageDetector(yolov8_weights_path, yolov8_source_dir, yolov8_output_dir)
imagenet_model = ImageNetImageDetector(imagenet_model_path, imagenet_source_dir, imagenet_output_dir)

# yolov8_model.run_detection()
imagenet_model.run_detection()