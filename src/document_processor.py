import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
# Download NLTK resources (only need to run once)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class DocumentProcessor:
    def __init__(self, file_path):
        self.raw_documents = []
        self.documents = self.read_and_tokenize(file_path)

    def preprocess_text(self, text):
        """Clean and tokenize the input text."""
        # Remove punctuation and make lowercase
        text = re.sub(r"[^\w\s]", "", text.lower())
        # Tokenize the text
        tokens = word_tokenize(text, language="english", preserve_line=True)
        # Lemmatize the tokens
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        return tokens

    def read_and_tokenize(self, file_path):
        """Read the file and tokenize the content of each document."""
        documents = []

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

            # Split the content into individual documents
            doc_strings = content.strip().split(
                "\n\n"
            )  # Assuming documents are separated by double newlines

            for doc in doc_strings:
                # Extract ID, Title, Author, Bibliography, and Content
                lines = doc.split("\n")
                doc_id = lines[0].replace("ID: ", "").strip()
                title = []
                author = []
                bibliography = []
                content_lines = []
                in_text = False

                for line in lines[1:]:
                    if line.startswith("Title: "):
                        title.append(line.replace("Title: ", "").strip())
                    elif line.startswith("Author: "):
                        author.append(line.replace("Author: ", "").strip())
                    elif line.startswith("Bibliography: "):
                        bibliography.append(line.replace(
                            "Bibliography: ", "").strip())
                    elif line.startswith("Text: "):
                        in_text = True
                        content_lines.append(
                            line.replace("Text: ", "").strip())
                    elif in_text:
                        content_lines.append(line.strip())
                    else:
                        title.append(line.strip())

                title = " ".join(title)
                author = " ".join(author)
                bibliography = " ".join(bibliography)
                content = " ".join(content_lines)

                self.raw_documents.append(
                    {
                        "id": doc_id,
                        "title": title,
                        "author": author,
                        "bibliography": bibliography,
                        "content": content,
                    }
                )

                # Tokenize the fields
                title_tokens = self.preprocess_text(title)
                author_tokens = self.preprocess_text(author)
                bibliography_tokens = self.preprocess_text(bibliography)
                content_tokens = self.preprocess_text(content)

                # Append the document as a dictionary
                documents.append(
                    {
                        "id": doc_id,
                        "title": title_tokens,
                        "author": author_tokens,
                        "bibliography": bibliography_tokens,
                        "content": content_tokens,
                    }
                )

        return documents
