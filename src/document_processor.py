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
        """
        Initialize the DocumentProcessor with the specified file path.

        Args:
            file_path (str): The path to the text file containing the documents.

        This constructor reads the documents from the specified file and tokenizes them,
        storing the raw documents and their processed versions in the instance variables.
        """
        self.raw_documents = []
        self.documents = self.read_and_tokenize(file_path)

    def preprocess_text(self, text):
        """
        Clean and tokenize the input text.

        This method performs several preprocessing steps on the input text:
        1. Converts the text to lowercase.
        2. Removes punctuation using a regular expression.
        3. Tokenizes the cleaned text into individual words using NLTK's word_tokenize.
        4. Lemmatizes each token to reduce it to its base form using WordNetLemmatizer.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            list: A list of cleaned and lemmatized tokens.

        Example:
            >>> processor = DocumentProcessor("documents.txt")
            >>> tokens = processor.preprocess_text("This is a sample text!")
            >>> print(tokens)
            ['this', 'is', 'a', 'sample', 'text']
        """
        # Remove punctuation and make lowercase
        text = re.sub(r"[^\w\s]", "", text.lower())
        # Tokenize the text
        tokens = word_tokenize(text, language="english", preserve_line=True)
        # Lemmatize the tokens
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        return tokens

    def read_and_tokenize(self, file_path):
        """
        Read the file and tokenize the content of each document.

        This method reads the specified file and processes its content to extract
        individual documents. Each document is expected to contain fields such as
        ID, Title, Author, Bibliography, and Text. The method tokenizes each of these
        fields and stores them in a structured format.

        Args:
            file_path (str): The path to the text file containing the documents.

        Returns:
            list: A list of dictionaries, each representing a processed document
                  with tokenized fields.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If there is an error while reading or processing the file.

        Example:
            >>> processor = DocumentProcessor("documents.txt")
            >>> documents = processor.read_and_tokenize("documents.txt")
            >>> print(documents)
            [{'id': '1', 'title': ['sample', 'title'], 'author': ['author', 'name'], ...}, ...]
        """
        documents = []

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

            # Split the content into individual documents
            # Assuming documents are separated by double newlines
            doc_strings = content.strip().split("\n\n")

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
