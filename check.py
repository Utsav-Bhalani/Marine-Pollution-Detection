try:
    from torchcam.methods import GradCAM
    print("torchcam is installed and working!")
except ImportError:
    print("Error: torchcam is not installed!")
