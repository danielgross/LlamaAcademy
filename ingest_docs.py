"""Load html from files, clean up, split, ingest into FAISS."""
import pickle
from typing import Any, List, Optional, Tuple
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.faiss import FAISS
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import contextlib
import lxml.html as LH
import lxml.html.clean as clean
import requests
import os
import re
import tqdm
import time
import luigi
import logging
from luigi.util import inherits

class APIReferenceLoader(WebBaseLoader):
    """
    Loader that uses Elinks and Selenium to load webpages.
    With customization to scrape the code.

    :param web_path: a string, the path of the website to be scraped
    :param header_template: an optional dictionary, for customizing request headers
    :param strategy: an optional string, specifies the scraping strategy,
                     defaults to "selenium_elinks"
    :param is_visible_scrape: a bool, whether to perform a visible content scraping,
                              defaults to False
    """

    def __init__(self, web_path: str, header_template: Optional[dict] = None, strategy: Optional[str] = "selenium_elinks", is_visible_scrape: bool = False):
        # Initialize the WebBaseLoader
        super().__init__(web_path=web_path, header_template=header_template)
        
        # Initialize the Firefox driver
        self.driver = self.init_firefox_driver()
        
        # Set the scraping strategy
        self.strategy = strategy
        
        # Set whether to perform a visible content scraping
        self.is_visible_scrape = is_visible_scrape

    def _scrape_bs4(self, url: str) -> Any:
        """
        Scrape the webpage using BeautifulSoup4.

        :param url: a string, the URL of the webpage to be scraped
        :return: a BeautifulSoup object
        """
        # Send a GET request to the URL
        html_doc = self.session.get(url)
        
        # Create a BeautifulSoup object from the HTML text
        soup = BeautifulSoup(html_doc.text, "html.parser")
        
        return soup

    def load(self) -> List[Document]:
        """
        Load data into document objects.

        :return: a List of Document objects containing the scraped content
        """
        # Implement different scraping strategies
        if self.strategy == "bs4":
            soup = self._scrape_bs4(self.web_path)
            text = soup.get_text()
        elif self.strategy == "selenium_elinks":
            text = self._scrape_SelElinks(self.web_path)
        else:
            raise ValueError("Strategy not supported")

        metadata = {"source": self.web_path}
        return [Document(page_content=text, metadata=metadata)]

    def find_common_words(self, s, t):
        """
        Find the common words between two strings.

        :param s: a string
        :param t: another string
        :return: a List of common words between s and t
        """

        # Split input strings into words
        s_words = s.split()
        t_words = t.split()

        # Find common words
        common_words = [word for word in s_words if word in t_words]

        return common_words

    def insert_missing_words(self, s, t, common_words):
        """
        Insert missing words from visible text into the target text (structured elements).

        :param s: a string, the source string
        :param t: a string, the target string
        :param common_words: a List of common words between s and t
        :return: a string, the target string with missing words inserted
        """
        s_words = s.split()
        t_words = t.split()
        missing_words = []

        for i in range(len(common_words)-1):
            start, end = common_words[i], common_words[i+1]
            start_idx = s_words.index(start)
            end_idx = s_words.index(end)
            missing_words.extend(s_words[start_idx+1:end_idx])

        for word in missing_words:
            if word not in t_words:
                t_words.insert(t_words.index(common_words[-1])+1, word)

        t_new = " ".join(t_words)
        return t_new

    def init_firefox_driver(self):
        """
        Initialize a headless Firefox browser.

        :return: a webdriver.Firefox object, representing the headless browser
        """
        options = Options()
        options.headless = True
        options.binary = FirefoxBinary("/usr/bin/firefox")
        service = FirefoxService(executable_path="geckodriver")
        driver = webdriver.Firefox(service=service, options=options)
        return driver

    def scrape_visible_elements(self, url):
        """
        Scrape the visible elements of the page using a headless browser and Selenium.

        :param url: a string, the URL of the webpage to be scraped
        :return: a string, the scraped visible content of the webpage
        """
        ignore_tags = ('style')
        with contextlib.closing(self.driver) as browser:
            browser.get(url)  # Load page
            time.sleep(10)
            content = browser.page_source
            cleaner = clean.Cleaner()
            content = cleaner.clean_html(content)
            doc = LH.fromstring(content)
            texts = []
            for elt in doc.iterdescendants():
                if elt.tag in ignore_tags:
                    continue
                text = elt.text or ''
                tail = elt.tail or ''
                words = ' '.join((text, tail)).strip()
                if words:
                    texts.append(words)
            return " ".join(texts)

    def scrape_structured_elements(self, url):
        """
        Scrape the structured elements of the page using text-based web browser Elinks.

        :param url: a string, the URL of the webpage to be scraped
        :return: a string, the scraped structured content of the webpage
        """
        response = self.session.get(url)
        with open("/tmp/struct.html", "w") as f:
            f.write(response.text)
        os.system("elinks --dump /tmp/struct.html > /tmp/struct.txt")
        with open("/tmp/struct.txt", "r") as f:
            lines = f.readlines()
        text = "".join(lines)
        return text

    def _scrape_SelElinks(self, url):
        """
        Combine the best from both worlds: visual and structured elements.

        :param url: a string, the URL of the webpage to be scraped
        :return: a string, the combined scraped content of the webpage
        """
        struct_text = self.scrape_structured_elements(url)
        struct_text = self.clean_text(struct_text)
        
        if self.is_visible_scrape:
            vis_text = self.scrape_visible_elements(url)
            t_joint = self.insert_missing_words(
                vis_text, struct_text, self.find_common_words(vis_text, struct_text))
            with open("/tmp/debug_vis.txt", "w") as f:
                f.write(self.clean_text(t_joint))
            return self.clean_text(t_joint)
        return struct_text

    def clean_text(self, text):
        """
        Clean up the text.

        :param text: a string, the text to be cleaned
        :return: a string, the cleaned text
        """
        delete_str = "Visible links"
        index = text.find(delete_str)
        if index != -1:
            text = text[:index]
        text = re.sub(r'\n\s*\n', '\n', text.strip())
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'-{3,}', '--', text)
        text = re.sub(r'═{3,}', '==', text)
        text = re.sub(r'_', '', text)
        text = re.sub(r'`', '', text)
        text = re.sub(r'Link: \[\d+\]prefetch','', text)
        text = re.sub(r'Link: \[\d+\]preload','', text)
        text = re.sub(r'Link: \[\d+\]preconnect','', text)
        text = re.sub(r'Link: \[\d+\]canonical','', text)
        text = re.sub(r'Link: \[\d+\]alternate','', text)
        text = re.sub(r'\[\d+\]', '', text)
        return text
    
    def clean_table_content(self, text):
        pass

def hierarchy_links(url_docs: str, recursive_depth: int = 1, current_depth: int = 1) -> List[str]:
    """
    Get all links from a web page up to a specified recursion depth.

    :param url_docs: a string, the URL of the web page 
    :param recursive_depth: an optional integer, the maximum recursion depth, defaults to 1
    :param current_depth: an optional integer, the current recursion depth, defaults to 1
    :return: a List of strings, the URLs of documents collected from the web page
    """

    # Check if we have reached the maximum recursion depth
    if current_depth > recursive_depth and recursive_depth != 0:
        return []
    elif recursive_depth == 0:
        return [url_docs]

    # Send a GET request to the provided URL
    reqs = requests.get(url_docs)
    # Create a BeautifulSoup object from the HTML content
    soup = BeautifulSoup(reqs.text, 'html.parser')
    # Initialize the list for collected document links
    docs_link = list()
    # Iterate over all the links in the web page
    for link in soup.find_all('a'):
        # Create an absolute URL by joining the base URL and the href attribute
        ref_link = urljoin(url_docs, link.get('href'))  
        # Check if the URL is valid, not equal to the base URL, and not already in the list
        if url_docs in ref_link and ref_link is not None and url_docs != ref_link:
            docs_link.append(ref_link)
            # Recursively collect links if maximum depth is not yet reached
            if current_depth < recursive_depth:
                docs_link.extend(
                    hierarchy_links(ref_link, recursive_depth, current_depth + 1)
                )
    
    # Return the list of collected document links
    return docs_link

def ingest_docs(url_docs: str, recursive_depth: int = 1, return_summary: bool = True, logger=None) -> Tuple[List, List]:
    """
    Get documents from web pages.

    :param url_docs: a string, the URL of the web page
    :param recursive_depth: an optional integer, the maximum recursion depth for getting links, defaults to 1
    :param return_summary: an optional bool, whether to return a summary of documents, defaults to True
    :param logger: an optional logging object, for logging progress and information, defaults to None
    :return: a Tuple with two Lists,
             first, the list of documents collected from the web pages,
             second, the list of documents used for summary (if return_summary is True)
    """
    embeddings = OpenAIEmbeddings()
    # Get links from the web page, up to the specified recursion depth
    docs_link = set(hierarchy_links(url_docs, recursive_depth))
    # Initialize the lists for collected documents and document summaries
    documents = list()
    docs_for_summary = list()
    logger.info(f"Crawling {docs_link} ...")
    # Iterate over the collected links
    for link in tqdm.tqdm(docs_link):
        # Initialize an APIReferenceLoader with the option to scrape visible content
        loader = APIReferenceLoader(link, is_visible_scrape=True)
        # Load the raw documents
        raw_documents = loader.load()
        # Initialize text splitters for documents and summaries
        text_splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=150)
        text_splitter_sum = TokenTextSplitter(chunk_size=3100, chunk_overlap=300)
        # Split documents for summary and document lists
        if return_summary:
            docs_for_summary.extend(text_splitter_sum.split_documents(raw_documents))
        documents.extend(text_splitter.split_documents(raw_documents))
    logger.info("Number of documents: {}".format(len(documents)))
    
    logger.info("Saving vectorstore into assets/vectorstore.pkl")
    vectorstore = FAISS.from_documents(documents, embeddings)
    with open("assets/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    return documents, docs_for_summary


class IngestDocumentsTask(luigi.Task):
    url_docs = luigi.Parameter()
    recursive_depth = luigi.IntParameter(default=1)
    logger = logging.getLogger(__name__)

    def output(self):
        return {
                'documents': luigi.LocalTarget("assets/documents_{}.pkl".format(self.task_id), format=luigi.format.Nop),
                'docs_for_summary': luigi.LocalTarget("assets/docs_for_summary_{}.pkl".format(self.task_id), format=luigi.format.Nop)
                } 

    def run(self):
        self.task_id  = luigi.task.task_id_str(self.get_task_family(), self.to_str_params())
        self.logger.info(self.task_id)
        docs, docs_for_summary = ingest_docs(self.url_docs, self.recursive_depth, logger=self.logger)
        with self.output()['documents'].open("wb") as f:
            pickle.dump(docs, f)
        with self.output()['docs_for_summary'].open("wb") as f:
            pickle.dump(docs_for_summary, f)


@inherits(IngestDocumentsTask)
class SaveVectorStoreTask(luigi.Task):
    logger = logging.getLogger(__name__)

    def requires(self):
        return self.clone(IngestDocumentsTask)

    def output(self):
        return luigi.LocalTarget("assets/vectorstore_{}.pkl".format(self.task_id), format=luigi.format.Nop)
                
    def run(self):
        self.logger.info(self.task_id)
        with self.input()['documents'].open("rb") as f:
            docs = pickle.load(f)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        with self.output().open("wb") as f:
            pickle.dump(vectorstore, f)


if __name__ == "__main__":
    luigi.build([SaveVectorStoreTask("https://developers.notion.com/reference/create-a-token", recursive_depth=0)], local_scheduler=True)
