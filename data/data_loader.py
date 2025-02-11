import os
import re
import time
import requests
import pandas as pd
from io import BytesIO

def sanitize_filename(filename):
    # Replace Windows-invalid characters: <>:"/\|?* with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)
    # Remove hyphens completely
    sanitized = sanitized.replace("-", "")
    return sanitized

class FigshareCollection:
    def __init__(self, collection_id):
        self.collection_id = collection_id
        self.link = f"https://api.figshare.com/v2/collections/{collection_id}"
    
    def list_articles(self):
        articles_url = f"{self.link}/articles"
        response = requests.get(articles_url)
        response.raise_for_status()
        return response.json()
    
    def list_all_files(self):
        articles = self.list_articles()
        all_files = {}
        for article in articles:
            article_id = article["id"]
            files_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
            response = requests.get(files_url)
            response.raise_for_status()
            files = response.json()
            all_files[article_id] = files
        return all_files
    
    def import_file(self, article_index=0, file_index=0, rows_to_skip=None):
        articles = self.list_articles()
        if article_index < 0 or article_index >= len(articles):
            raise ValueError("Article index out of range.")
        
        selected_article = articles[article_index]
        article_id = selected_article["id"]
        
        # List files for the selected article
        files_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
        response = requests.get(files_url)
        response.raise_for_status()
        files = response.json()
        
        if file_index < 0 or file_index >= len(files):
            raise ValueError("File index out of range.")
        
        selected_file = files[file_index]
        filename = selected_file["name"]
        extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else None
        
        # Download the file content
        download_url = selected_file.get("download_url")
        if not download_url:
            raise ValueError("Download URL not found for the selected file.")
        
        download_response = requests.get(download_url)
        download_response.raise_for_status()
        
        # Load the file into a DataFrame based on file extension
        if extension == "csv":
            df = pd.read_csv(BytesIO(download_response.content), skiprows=rows_to_skip)
        elif extension == "xlsx":
            df = pd.read_excel(BytesIO(download_response.content), skiprows=rows_to_skip)
        else:
            raise ValueError("Unsupported file extension. Only CSV and XLSX files are supported.")
        
        return df

    def download_all_files(self, root_dir="raw"):
        # Create the root directory if it doesn't exist
        os.makedirs(root_dir, exist_ok=True)
        
        articles = self.list_articles()
        print(f"Found {len(articles)} articles in collection {self.collection_id}.")
        
        for article in articles:
            article_id = article["id"]
            # Sanitize the article title to remove invalid characters and replace spaces with underscores
            raw_title = article.get("title", f"article_{article_id}")
            article_title = sanitize_filename(raw_title).replace(" ", "_")
            
            print(f"\nProcessing Article ID: {article_id} - Title: {article_title}")
            
            # Retrieve files for the current article
            files_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
            response = requests.get(files_url)
            response.raise_for_status()
            files = response.json()
            
            if not files:
                print("  No files found for this article.")
                continue
            
            # Download each file in the article
            for file_info in files:
                file_name = file_info.get("name")
                download_url = file_info.get("download_url")
                
                if not download_url:
                    print(f"  No download URL for file: {file_name}. Skipping.")
                    continue
                
                print(f"  Downloading file: {file_name}")
                file_response = requests.get(download_url, stream=True)
                file_response.raise_for_status()
                
                file_path = os.path.join(root_dir, file_name)
                with open(file_path, "wb") as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                print(f"    Saved to {file_path}")
                
                # Optional: Pause briefly between downloads to be respectful to the API
                time.sleep(0.2)

if __name__ == "__main__":
    # Instantiate the FigshareCollection with the desired collection ID (e.g., 5312786)
    collection = FigshareCollection(collection_id=5312786)
    print("Collection Link:", collection.link)
    
    # 1. List all articles in the collection
    articles = collection.list_articles()
    print("\nArticles in the collection:")
    for i, article in enumerate(articles):
        print(f"{i}: Article ID: {article['id']}, Title: {article.get('title', 'No Title')}")
    
    # 2. List all files in the collection
    all_files = collection.list_all_files()
    print("\nFiles in each article of the collection:")
    for article_id, files in all_files.items():
        print(f"Article ID {article_id}:")
        for file in files:
            print(f"  - {file['name']}")
    
    # 3. Download and store all files into the "raw" folder
    print("\nStarting download of all files into the 'raw' folder...")
    collection.download_all_files(root_dir="data\\raw")
    print("\nAll files downloaded successfully!")