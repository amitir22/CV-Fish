#!/usr/bin/env python3

import cv2


def check_environment_for_cuda():
    """
    check_environment_for_cuda
    """
    # Get the build information
    build_info = cv2.getBuildInformation()
    
    # Check for CUDA and cuDNN in the build information
    cuda_enabled = "CUDA" in build_info and "YES" in build_info.split("CUDA")[1].split("\n")[0]  # = 'CUDA' and 'YES' are found in the same line
    cudnn_enabled = "cuDNN" in build_info and "YES" in build_info.split("cuDNN")[1].split("\n")[0]  # = 'cuDNN' and 'YES' are found in the same line

    # Display the result
    if cuda_enabled and cudnn_enabled:
        print("CUDA and cuDNN are enabled in OpenCV.")
    elif cuda_enabled:
        print("CUDA is enabled, but cuDNN is not enabled in OpenCV.")
    else:
        print("CUDA is not enabled in OpenCV.")


def main():
    """
    main
    """
    check_environment_for_cuda()


if __name__ == '__main__':
    main()
