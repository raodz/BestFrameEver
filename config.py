import torchvision.transforms as T

movie_path = "C:\\Users\\Admin\\Downloads\\WhatsApp Video 2024-11-15 at " "18.13.08.mp4"
movie_title = "The Movie"
actors_list = ["X", "Y"]
filter_every_nth_frame = 50
frameset_transforms = T.Compose(
    [T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)
