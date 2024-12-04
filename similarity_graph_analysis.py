import torch
import open_clip
from PIL import Image
from torch.nn.functional import cosine_similarity

def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer

model, transform, tokenizer = load_open_clip()

if __name__ == '__main__':

    model, transform, tokenizer = load_open_clip()

    men_image_paths = [
        "images/men-women-classification/men/00000001.jpg",
        "images/men-women-classification/men/00000002.jpg",
        "images/men-women-classification/men/00000003.jpg",
        "images/men-women-classification/men/00000004.jpg",
        "images/men-women-classification/men/00000005.jpg",
    ]

    women_image_paths = [
        "images/men-women-classification/women/00000001.jpg",
        "images/men-women-classification/women/00000002.jpg",
        "images/men-women-classification/women/00000003.jpg",
        "images/men-women-classification/women/00000005.jpeg",
        "images/men-women-classification/women/00000006.png",
    ]

    men_images = [Image.open(img) for img in men_image_paths]
    men_images = [transform(img).unsqueeze(dim=0) for img in men_images]
    men_image_encodings = [model.encode_image(img) for img in men_images]
    women_images = [Image.open(img) for img in women_image_paths]
    women_images = [transform(img).unsqueeze(dim=0) for img in women_images]
    women_image_encodings = [model.encode_image(img) for img in women_images]

    template = "a photo of a {c}"

    cls_strs = ["man", "woman"]

    cls_templates = [template.replace("{c}", s) for s in cls_strs]
    text_tokens = [tokenizer(s) for s in cls_templates]
    text_embeddings = [model.encode_text(s) for s in text_tokens]

    for men_img in men_image_encodings:
        predictions = []
        for idx, text_embedding in enumerate(text_embeddings):
            prediction = cosine_similarity(men_img, text_embedding, dim=1)
            predictions.append(prediction)
            print(prediction, cls_templates[idx])
        print("GT: Man")
        print()

    for women_img in women_image_encodings:
        predictions = []
        for idx, text_embedding in enumerate(text_embeddings):
            prediction = cosine_similarity(women_img, text_embedding, dim=1)
            predictions.append(prediction)
            print(prediction, cls_templates[idx])
        print("GT: Woman")
        print()

