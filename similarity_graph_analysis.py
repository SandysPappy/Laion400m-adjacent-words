import torch
import open_clip
from PIL import Image
from torch.nn.functional import cosine_similarity
from graph.laion_graph import load_graph_and_keymap, get_neighbors
import matplotlib.pyplot as plt

def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer

model, transform, tokenizer = load_open_clip()



def write_similarity_plot(similarities: list[float], class_names: list[str], colors: list[str], out_file_name: str):
    plt.figure(figsize=(15, 9))
    plt.bar(class_names, similarities, color=colors)
    plt.title('Class Similarities', fontsize=16)
    plt.xlabel('Class Label', fontsize=14)
    plt.ylabel('Similarity Score', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_file_name)
    plt.close()


def get_similarities(model, transform, tokenizer, img_path: str, cls_strs: tuple[str, str], cls_biases_1: list[str], cls_biases_2: list[str]) -> list[tuple[str, float]]:
    img = Image.open(img_path)
    img = transform(img).unsqueeze(dim=0)
    img_encoding = model.encode_image(img)

    text_strs = cls_strs + cls_biases_1 + cls_biases_2
    text_tokens = [tokenizer(s) for s in text_strs]
    text_embeddings = [model.encode_text(s) for s in text_tokens]

    similarities = [cosine_similarity(img_encoding, text_encoding, dim=1) for text_encoding in text_embeddings]
    similarities = [n.item() for n in similarities]

    retvals = []
    for idx, s in enumerate(cls_strs):
        retvals.append((s, similarities[idx]))
    for idx, s in enumerate(cls_biases_1):
        retvals.append((s, similarities[idx+len(cls_strs)]))
    for idx, s in enumerate(cls_biases_2):
        retvals.append((s, similarities[idx+len(cls_strs)+len(cls_biases_1)]))

    # normalize the first two image similarities so it looks nicer on the graph
    # x, y = similarities[0], similarities[1]

    # total = x+y
    # retvals[0] = (cls_strs[0], x/total)
    # retvals[1] = (cls_strs[1], y/total)

    return retvals



if __name__ == '__main__':

    model, transform, tokenizer = load_open_clip()

    cls_strs = ["man", "woman"]
    men_bias_terms = ['urdu', 'utd', 'hwpl', 'novels', 'lamotta', 'darwin', 'invincible', 'salah', 'tt', 'dapper', 'downey', 'stark', 'ip', 'pac', 'mins', 'trucks', 'kin', 'dpcw', 'macho', 'egress', 'arsenal', 'vitruvian', 'accessible', 'namal', 'ronaldo', 'manly', 'bvlgari', 'wheelie', 'method']
    women_bias_terms = ['lupin', 'indo', 'redvalentino', 'mujer', 'wedges', 'femme', 'rhinestone', 'manicure', 'lipstick', 'midi', 'gadot', 'veil', 'parasol', 'renoir', 'wedge', 'zapatos', '0no', 'ita', 'corset', 'eyelashes', 'degas', 'monet', 'flamenco', 'loretta', 'chiffon', 'ear-rings', 'glamorous', 'stiletto', 'petals', 'peep']
    colors1 = ["teal", "coral"]
    colors1 += ["blue"]*len(men_bias_terms)
    colors1 += ["pink"]*len(women_bias_terms)

    img1 = "images/men-women-classification/men/00000004.jpg"
    sims1 = get_similarities(model, transform, tokenizer, img1, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs1, similarities1 = zip(*sims1)
    write_similarity_plot(similarities1, cls_strs1, colors1, "plots/man1.png")
    ####
    img2 = "images/men-women-classification/men/00000002.jpg"
    sims2 = get_similarities(model, transform, tokenizer, img2, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs2, similarities2 = zip(*sims2)
    write_similarity_plot(similarities2, cls_strs2, colors1, "plots/man2.png")
    
    ###
    img3 = "images/men-women-classification/men/00000003.jpg"
    sims3 = get_similarities(model, transform, tokenizer, img3, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs3, similarities3 = zip(*sims3)
    write_similarity_plot(similarities3, cls_strs3, colors1, "plots/man3.png")

    img4 = "images/men-women-classification/men/00000004.jpg"
    sims4 = get_similarities(model, transform, tokenizer, img4, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs4, similarities4 = zip(*sims4)
    write_similarity_plot(similarities4, cls_strs4, colors1, "plots/man4.png")

    img5 = "images/men-women-classification/men/00000005.jpg"
    sims5 = get_similarities(model, transform, tokenizer, img5, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs5, similarities5 = zip(*sims5)
    write_similarity_plot(similarities5, cls_strs5, colors1, "plots/man5.png")

    
    img6 = "images/men-women-classification/women/00000001.jpg"
    sims6 = get_similarities(model, transform, tokenizer, img6, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs6, similarities6 = zip(*sims6)
    write_similarity_plot(similarities6, cls_strs6, colors1, "plots/woman1.png")


    img7 = "images/men-women-classification/women/00000002.jpg"
    sims7 = get_similarities(model, transform, tokenizer, img7, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs7, similarities7 = zip(*sims7)
    write_similarity_plot(similarities7, cls_strs7, colors1, "plots/woman2.png")


    img8 = "images/men-women-classification/women/00000003.jpg"
    sims8 = get_similarities(model, transform, tokenizer, img8, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs8, similarities8 = zip(*sims8)
    write_similarity_plot(similarities8, cls_strs8, colors1, "plots/woman3.png")



    img9 = "images/men-women-classification/women/00000005.jpeg"
    sims9 = get_similarities(model, transform, tokenizer, img9, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs9, similarities9 = zip(*sims9)
    write_similarity_plot(similarities9, cls_strs9, colors1, "plots/woman4.png")
    

    img10 = "images/men-women-classification/women/00000006.png"
    sims10 = get_similarities(model, transform, tokenizer, img10, cls_strs, men_bias_terms, women_bias_terms)
    cls_strs10, similarities10 = zip(*sims10)
    write_similarity_plot(similarities10, cls_strs10, colors1, "plots/woman5.png")


    cls_strs_food = ["donut", "salad"]
    donut_bias_terms = ['cake', 'pan', 'hair', 'baking']
    salad_bias_terms = ['healthy', 'grilled', 'woman', 'ranch']
    colors_food = ["brown", "teal"]
    colors_food += ["beige"]*len(donut_bias_terms)
    colors_food += ["green"]*len(salad_bias_terms)

    img_food1 = "images/food/donut.png"
    sims_food1 = get_similarities(model, transform, tokenizer, img_food1, cls_strs_food, donut_bias_terms, salad_bias_terms)
    cls_strs_food1, similarities_food1 = zip(*sims_food1)
    write_similarity_plot(similarities_food1, cls_strs_food1, colors_food, "plots/donut1.png")


    img_food2 = "images/food/salad.png"
    sims_food2 = get_similarities(model, transform, tokenizer, img_food2, cls_strs_food, donut_bias_terms, salad_bias_terms)
    cls_strs_food2, similarities_food2 = zip(*sims_food2)
    write_similarity_plot(similarities_food2, cls_strs_food2, colors_food, "plots/salad1.png")
