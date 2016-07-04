"""
Full pipeline to convert a jpg to a caption.
Steps:
1. Pass the image through a pretrained network, getting a pickle file containing
the image features.
2. Pipe the pickle file through gen_caption.
"""

