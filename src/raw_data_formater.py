# Import the necessary library
import ir_datasets

# Load the Cranfield dataset
corpus = ir_datasets.load("cranfield")

# Define the output file path
output_file = 'cranfield_dataset.txt'

# Open the text file in write mode
with open(output_file, mode='w', encoding='utf-8') as txtfile:
    # Write the dataset to the file
    for doc in corpus.docs_iter():
        line = f"ID: {doc.doc_id}\nTitle: {doc.title}\nAuthor: {doc.author}\nBibliography: {doc.bib}\nText: {doc.text}\n\n"
        txtfile.write(line)

print(f"Dataset has been written to {output_file}")