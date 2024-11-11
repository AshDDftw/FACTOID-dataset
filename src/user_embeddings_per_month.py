from utils.feature_computing import *
from dataset.reddit_user_dataset import RedditUserDataset
import os
import pickle as pkl
from argparse import ArgumentParser
from tqdm import tqdm
import glob
from constants import *
from sentence_transformers import SentenceTransformer
from utils.train_utils import process_tweet


def sentence_embeddings_model(bert_model):
    model = SentenceTransformer(bert_model).to(DEVICE)
    return model

# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--vocabs_dir", required=True, type=str)
parser.add_argument("--base_dataset", dest="base_dataset_path", default='../data/reddit_dataset/reddit_corpus_balanced_filtered.gzip', type=str)
parser.add_argument("--output_dir", default="../data/reddit_dataset/bert_embeddings/", type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        print("Creating directory {}".format(args.output_dir))
        os.mkdir(args.output_dir)

    # Load base dataset
    base_dataset = RedditUserDataset.load_from_file(args.base_dataset_path, compression='gzip')
    model = sentence_embeddings_model('all-mpnet-base-v2')

    # Sort files by month index
    files = glob.glob(os.path.join(args.vocabs_dir, '*.txt'))
    files = sorted(files, key=lambda file: int(file.split('.')[-2].split('_')[-1]))

    # Process each monthly file
    for i, file in tqdm(enumerate(files), desc="File Processing Progress"):
        user_texts = {}
        embeddings = {'desc': 'Bert Embeddings per month'}

        # Read each line of the file with UTF-8 encoding
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                temp = line.split('\t')
                assert len(temp) == 2, print(line)
                user = temp[0]
                text = temp[1]

                # Collect texts for each user
                texts = user_texts.get(user, [])
                texts.append(text.strip())
                user_texts[user] = texts

        # Generate embeddings for each user
        for user_id, texts in tqdm(user_texts.items(), desc="Embedding Progress"):
            texts = [process_tweet(text) for text in texts]
            output = model.encode(texts)
            embeddings[user_id] = torch.tensor(np.mean(output, axis=0))

        # Save embeddings to file
        pkl.dump(embeddings, open(os.path.join(args.output_dir, f'user_embeddings_{i}.pkl'), 'wb'))
