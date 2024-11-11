from dataset.reddit_user_dataset import RedditUserDataset
import datetime
from argparse import ArgumentParser
import os
import re

# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--base_dataset", dest="base_dataset", required=True, type=str)
parser.add_argument("--output_dir", default="../data/user_vocabs_per_month", type=str)

if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        print("Creating directory {}".format(args.output_dir))
        os.mkdir(args.output_dir)

    # Load dataset
    data_frame = RedditUserDataset.load_from_file(args.base_dataset, compression='gzip').data_frame
    print(data_frame.head())

    # Set range of dates in the dataset
    start_date = datetime.datetime(2020, 1, 1, 0, 0, 0) 
    end_date = datetime.datetime(2021, 4, 30, 23, 59, 59)
    SIZE = 16  # Total number of months

    # Initialize a list to hold documents per date
    docs_per_date = []
    for i in range(SIZE):
        docs_per_date.append(dict())
        
    # Iterate through each row in the data frame
    for index, row in data_frame.iterrows():
        docs = row['documents']
        user = row['user_id']
        
        # Process each document
        for doc in docs: 
            date = doc[2]
            text = doc[1]
            
            # Filter documents by date range
            if start_date <= date <= end_date:
                add = 0
                if date.year == 2021:
                    add = 12

                idx = date.month - 1 + add
                value = docs_per_date[idx].get(user, [])
                
                # Clean up whitespace in text
                text = re.sub(r"\s\s+", ' ', text)
                text = re.sub(r"\s+", ' ', text)
                value.append(text.strip())
                
                # Update the dictionary for the month
                docs_per_date[idx][user] = value

    # Write documents to files per month
    for date in range(len(docs_per_date)):
        filename = os.path.join(args.output_dir, 'user_vocab_' + str(date) + '.txt')
        
        # Write each user's documents to the file
        with open(filename, 'w', encoding='utf-8') as f:
            user_docs = docs_per_date[date]
            
            for user, docs in user_docs.items():
                for doc in docs:
                    f.write(user + '\t' + doc + '\n')